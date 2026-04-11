#!/usr/bin/env python3
"""
Phase 4 ablation postprocess runner.

Completes the remaining S4-8 work after the six-scenario ablation eval JSONs
have been generated:
1. Resolve anchor-full + ablation variants into a dedicated manifest
2. Aggregate checkpoint-level and policy-level rows
3. Build anchor-vs-ablation comparison tables
4. Write a focused QC report for the ablation chain

Usage:
    conda activate env_isaaclab
    python scripts/phase_morl/run_phase4_ablation_postprocess.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from aggregate_phase4_results import (  # noqa: E402
    COMMAND_KEYS,
    OBJECTIVE_KEYS,
    SCENARIO_META_KEYS,
    SUPPLEMENTAL_KEYS,
    TRACKING_KEYS,
    build_checkpoint_level_rows,
    build_policy_level_rows,
    load_phase4_summary_rows,
)
from phase4_manifest import DEFAULT_PHASE4_MAIN_MANIFEST, Phase4ManifestEntry, load_phase4_manifest  # noqa: E402
from run_morl_ablation import load_phase4_ablation_manifest  # noqa: E402
from run_phase4_ablation_eval import DEFAULT_ABLATION_MANIFEST, load_eval_targets_from_ablation_manifest  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY_DIR = PROJECT_ROOT / "logs" / "eval" / "phase_morl_v2"
DEFAULT_RUN_ROOT = PROJECT_ROOT / "logs" / "rsl_rl" / "unitree_go1_rough"
DEFAULT_OUTPUT_DIR = DEFAULT_SUMMARY_DIR / "ablation"
MEAN_CMD_VX_PROXY_TOL = 1e-4
SEED_OUTLIER_REL_DEV_THRESH = 0.5
COMPARISON_KEYS = TRACKING_KEYS + OBJECTIVE_KEYS + ("success_rate", "mean_timeout_rate", "recovery_time")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "resolved_manifest_json": output_dir / "resolved_ablation_manifest.json",
        "checkpoint_level_csv": output_dir / "checkpoint_level.csv",
        "policy_level_ablation_csv": output_dir / "policy_level_ablation.csv",
        "ablation_comparison_csv": output_dir / "ablation_comparison.csv",
        "qc_report_md": output_dir / "qc_report.md",
    }


def _unique_in_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values))


def _population_std(values: list[float]) -> float:
    mean_value = _mean(values)
    return float(math.sqrt(sum((value - mean_value) ** 2 for value in values) / len(values)))


def _write_resolved_manifest(path: Path, *, entries: list[Phase4ManifestEntry], metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "anchor_policy_id": metadata["anchor_policy_id"],
        "variant_policy_ids": metadata["variant_policy_ids"],
        "seed_list": metadata["seed_list"],
        "entries": [asdict(entry) for entry in entries],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_resolved_ablation_entries(
    *,
    main_manifest_path: Path,
    ablation_manifest_path: Path,
    run_root: Path,
    seeds: set[int] | None = None,
    ablation_ids: set[str] | None = None,
    include_anchor_full: bool = True,
) -> tuple[list[Phase4ManifestEntry], dict]:
    ablation_manifest = load_phase4_ablation_manifest(ablation_manifest_path, project_root=PROJECT_ROOT)
    main_entries = load_phase4_manifest(main_manifest_path)

    anchor_policy_id = str(ablation_manifest.get("anchor_policy_id", "P10"))
    manifest_seeds = [int(seed) for seed in ablation_manifest.get("training_protocol", {}).get("training_seeds", [42, 43, 44])]
    selected_seeds = sorted(seeds) if seeds is not None else manifest_seeds

    variant_targets = load_eval_targets_from_ablation_manifest(
        ablation_manifest_path,
        project_root=PROJECT_ROOT,
        run_root=run_root,
        seeds=set(selected_seeds),
        entry_ids=ablation_ids,
        include_anchor_full=False,
    )

    variant_policy_ids = _unique_in_order([target.policy_id for target in variant_targets])
    entries: list[Phase4ManifestEntry] = []

    if include_anchor_full:
        anchor_entries = [
            Phase4ManifestEntry(
                family="ablation",
                policy_id=entry.policy_id,
                canonical_seed=entry.canonical_seed,
                run_dir=entry.run_dir,
                checkpoint=entry.checkpoint,
                task=entry.task,
                output_stem=entry.output_stem,
                evidence_layer="C",
                official_hv_eligible=False,
                source_state=entry.source_state,
            )
            for entry in main_entries
            if entry.policy_id == anchor_policy_id and entry.canonical_seed in selected_seeds
        ]
        found_anchor_seeds = sorted(entry.canonical_seed for entry in anchor_entries)
        missing_anchor_seeds = sorted(set(selected_seeds) - set(found_anchor_seeds))
        if missing_anchor_seeds:
            raise ValueError(
                f"Missing anchor-full entries for {anchor_policy_id} seeds {missing_anchor_seeds} in main manifest."
            )
        entries.extend(sorted(anchor_entries, key=lambda entry: entry.canonical_seed))

    for target in variant_targets:
        entries.append(
            Phase4ManifestEntry(
                family="ablation",
                policy_id=target.policy_id,
                canonical_seed=target.seed,
                run_dir=target.run_dir_name,
                checkpoint=target.checkpoint,
                task=target.task,
                output_stem=target.output_stem,
                evidence_layer="C",
                official_hv_eligible=False,
                source_state="active",
            )
        )

    metadata = {
        "anchor_policy_id": anchor_policy_id,
        "variant_policy_ids": variant_policy_ids,
        "seed_list": selected_seeds,
    }
    return entries, metadata


def build_ablation_comparison_rows(
    policy_rows: list[dict],
    *,
    anchor_policy_id: str,
    variant_policy_ids: list[str],
) -> list[dict]:
    indexed = {(row["policy_id"], row["scenario_id"]): row for row in policy_rows}
    scenario_ids = sorted({row["scenario_id"] for row in policy_rows})
    rows: list[dict] = []

    for scenario_id in scenario_ids:
        anchor_row = indexed.get((anchor_policy_id, scenario_id))
        if anchor_row is None:
            continue
        for variant_policy_id in variant_policy_ids:
            variant_row = indexed.get((variant_policy_id, scenario_id))
            if variant_row is None:
                continue

            row = {
                "scenario_id": scenario_id,
                "scenario_name": anchor_row["scenario_name"],
                "terrain_mode": anchor_row["terrain_mode"],
                "disturbance_mode": anchor_row["disturbance_mode"],
                "analysis_group": anchor_row["analysis_group"],
                "anchor_policy_id": anchor_policy_id,
                "ablation_policy_id": variant_policy_id,
                "anchor_num_seeds": anchor_row["num_seeds"],
                "ablation_num_seeds": variant_row["num_seeds"],
                "anchor_seed_list": anchor_row["seed_list"],
                "ablation_seed_list": variant_row["seed_list"],
            }

            for key in COMPARISON_KEYS:
                anchor_value = anchor_row.get(key)
                variant_value = variant_row.get(key)
                row[f"anchor_{key}"] = anchor_value
                row[f"anchor_{key}_std"] = anchor_row.get(f"{key}_std")
                row[f"ablation_{key}"] = variant_value
                row[f"ablation_{key}_std"] = variant_row.get(f"{key}_std")
                row[f"delta_{key}"] = (
                    float(variant_value) - float(anchor_value)
                    if anchor_value not in (None, "") and variant_value not in (None, "")
                    else None
                )

            rows.append(row)

    return rows


def build_ablation_qc_payload(
    *,
    rows: list[dict],
    policy_rows: list[dict],
    comparison_rows: list[dict],
    anchor_policy_id: str,
    variant_policy_ids: list[str],
    expected_seed_count: int,
) -> dict:
    required_meta = ("scenario_id", "scenario_name", "terrain_mode", "disturbance_mode", "analysis_group")
    missing_meta_rows = [
        row["summary_path"]
        for row in rows
        if any(row.get(key) in (None, "") for key in required_meta)
    ]

    family_values = sorted({str(row["family"]) for row in rows})
    family_status = "PASS" if family_values == ["ablation"] else "FAIL"

    mean_cmd_vx_diffs = [float(row["mean_cmd_vx_abs_diff"]) for row in rows if row.get("mean_cmd_vx_abs_diff") is not None]
    max_cmd_vx_diff = max(mean_cmd_vx_diffs) if mean_cmd_vx_diffs else None
    mean_cmd_vx_status = "PASS" if max_cmd_vx_diff is not None and max_cmd_vx_diff <= MEAN_CMD_VX_PROXY_TOL else "FAIL"

    expected_policy_ids = [anchor_policy_id] + variant_policy_ids
    expected_comparison_rows = len({row["scenario_id"] for row in policy_rows}) * len(variant_policy_ids)
    pair_coverage_issues: list[str] = []
    grouped_policy_rows: dict[str, list[dict]] = defaultdict(list)
    for row in policy_rows:
        grouped_policy_rows[row["scenario_id"]].append(row)

    for scenario_id, scenario_rows in sorted(grouped_policy_rows.items()):
        present_policy_ids = {row["policy_id"] for row in scenario_rows}
        missing_policy_ids = [policy_id for policy_id in expected_policy_ids if policy_id not in present_policy_ids]
        if missing_policy_ids:
            pair_coverage_issues.append(f"{scenario_id}: missing policies {missing_policy_ids}")
        for row in scenario_rows:
            if int(row["num_seeds"]) != expected_seed_count:
                pair_coverage_issues.append(
                    f"{scenario_id}: {row['policy_id']} expected {expected_seed_count} seeds, got {row['num_seeds']}"
                )

    seed_outlier_rows = []
    grouped_checkpoint_rows: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped_checkpoint_rows[(row["policy_id"], row["scenario_id"])].append(row)

    for (policy_id, scenario_id), group_rows in grouped_checkpoint_rows.items():
        if len(group_rows) < 2:
            continue
        values = [float(row["mean_vx_meas"]) for row in group_rows if row.get("mean_vx_meas") is not None]
        if len(values) < 2:
            continue
        mean_value = _mean(values)
        if mean_value == 0.0:
            continue
        for row in group_rows:
            if row.get("mean_vx_meas") is None:
                continue
            rel_dev = abs(float(row["mean_vx_meas"]) - mean_value) / abs(mean_value)
            if rel_dev > SEED_OUTLIER_REL_DEV_THRESH:
                seed_outlier_rows.append(
                    {
                        "policy_id": policy_id,
                        "scenario_id": scenario_id,
                        "canonical_seed": row["canonical_seed"],
                        "mean_vx_meas": float(row["mean_vx_meas"]),
                        "group_mean_vx_meas": mean_value,
                        "rel_dev_from_mean": rel_dev,
                    }
                )
    seed_outlier_rows.sort(key=lambda item: (item["policy_id"], item["scenario_id"], item["canonical_seed"]))

    checks = {
        "summary_row_count": {
            "status": "PASS",
            "count": len(rows),
        },
        "policy_row_count": {
            "status": "PASS",
            "count": len(policy_rows),
        },
        "scenario_metadata_complete": {
            "status": "PASS" if not missing_meta_rows else "FAIL",
            "missing_rows": missing_meta_rows,
        },
        "family_tagging": {
            "status": family_status,
            "family_values": family_values,
        },
        "mean_cmd_vx_proxy": {
            "status": mean_cmd_vx_status,
            "max_abs_diff": max_cmd_vx_diff,
            "tolerance": MEAN_CMD_VX_PROXY_TOL,
        },
        "pair_coverage": {
            "status": "PASS" if not pair_coverage_issues and len(comparison_rows) == expected_comparison_rows else "FAIL",
            "expected_policy_ids": expected_policy_ids,
            "expected_seed_count": expected_seed_count,
            "expected_comparison_rows": expected_comparison_rows,
            "actual_comparison_rows": len(comparison_rows),
            "issues": pair_coverage_issues,
        },
        "seed_outlier": {
            "status": "WARN" if seed_outlier_rows else "PASS",
            "threshold": SEED_OUTLIER_REL_DEV_THRESH,
            "outlier_count": len(seed_outlier_rows),
            "rows": seed_outlier_rows,
        },
    }

    return {
        "anchor_policy_id": anchor_policy_id,
        "variant_policy_ids": variant_policy_ids,
        "checks": checks,
    }


def render_ablation_qc_report(payload: dict) -> str:
    checks = payload["checks"]
    lines = [
        "# Phase 4 Ablation QC Report",
        "",
        f"- anchor_policy_id: `{payload['anchor_policy_id']}`",
        f"- variant_policy_ids: `{','.join(payload['variant_policy_ids'])}`",
        "",
        "## Checks",
        "",
        f"- summary row count: **{checks['summary_row_count']['status']}** (`{checks['summary_row_count']['count']}` rows)",
        f"- policy row count: **{checks['policy_row_count']['status']}** (`{checks['policy_row_count']['count']}` rows)",
        f"- scenario metadata completeness: **{checks['scenario_metadata_complete']['status']}**",
        f"- family tagging: **{checks['family_tagging']['status']}** (`{','.join(checks['family_tagging']['family_values'])}`)",
        f"- mean_cmd_vx proxy check: **{checks['mean_cmd_vx_proxy']['status']}**",
        f"  - max_abs_diff: `{checks['mean_cmd_vx_proxy']['max_abs_diff']}`",
        f"  - tolerance: `{checks['mean_cmd_vx_proxy']['tolerance']}`",
        f"- pair coverage: **{checks['pair_coverage']['status']}**",
        f"  - expected_policy_ids: `{','.join(checks['pair_coverage']['expected_policy_ids'])}`",
        f"  - expected_seed_count: `{checks['pair_coverage']['expected_seed_count']}`",
        f"  - comparison_rows: `{checks['pair_coverage']['actual_comparison_rows']}/{checks['pair_coverage']['expected_comparison_rows']}`",
        f"- seed outlier: **{checks['seed_outlier']['status']}**",
        f"  - threshold: `{checks['seed_outlier']['threshold']}`",
        f"  - outlier_count: `{checks['seed_outlier']['outlier_count']}`",
        "",
    ]

    if checks["scenario_metadata_complete"]["missing_rows"]:
        lines.append("### Missing Metadata")
        for row in checks["scenario_metadata_complete"]["missing_rows"]:
            lines.append(f"- `{row}`")
        lines.append("")

    if checks["pair_coverage"]["issues"]:
        lines.append("### Pair Coverage Issues")
        for issue in checks["pair_coverage"]["issues"]:
            lines.append(f"- {issue}")
        lines.append("")

    if checks["seed_outlier"]["rows"]:
        lines.append("### Seed Outliers")
        for row in checks["seed_outlier"]["rows"]:
            lines.append(
                "- "
                f"{row['policy_id']} {row['scenario_id']} seed{row['canonical_seed']} "
                f"(mean_vx_meas={row['mean_vx_meas']:.6f}, group_mean={row['group_mean_vx_meas']:.6f}, "
                f"rel_dev={row['rel_dev_from_mean']:.3f})"
            )
        lines.append("")

    return "\n".join(lines)


def generate_phase4_ablation_postprocess_outputs(
    *,
    summary_dir: Path,
    main_manifest_path: Path,
    ablation_manifest_path: Path,
    run_root: Path,
    output_dir: Path,
    seeds: set[int] | None = None,
    ablation_ids: set[str] | None = None,
    include_anchor_full: bool = True,
    dry_run: bool = False,
) -> dict[str, Path]:
    outputs = _output_paths(output_dir)
    if dry_run:
        return outputs

    entries, metadata = build_resolved_ablation_entries(
        main_manifest_path=main_manifest_path,
        ablation_manifest_path=ablation_manifest_path,
        run_root=run_root,
        seeds=seeds,
        ablation_ids=ablation_ids,
        include_anchor_full=include_anchor_full,
    )
    _write_resolved_manifest(outputs["resolved_manifest_json"], entries=entries, metadata=metadata)

    rows = load_phase4_summary_rows(summary_dir=summary_dir, manifest_path=outputs["resolved_manifest_json"])
    checkpoint_rows = build_checkpoint_level_rows(rows)
    policy_rows = build_policy_level_rows(rows)
    comparison_rows = build_ablation_comparison_rows(
        policy_rows,
        anchor_policy_id=metadata["anchor_policy_id"],
        variant_policy_ids=metadata["variant_policy_ids"],
    )
    qc_payload = build_ablation_qc_payload(
        rows=rows,
        policy_rows=policy_rows,
        comparison_rows=comparison_rows,
        anchor_policy_id=metadata["anchor_policy_id"],
        variant_policy_ids=metadata["variant_policy_ids"],
        expected_seed_count=len(metadata["seed_list"]),
    )

    _write_csv(outputs["checkpoint_level_csv"], checkpoint_rows)
    _write_csv(outputs["policy_level_ablation_csv"], policy_rows)
    _write_csv(outputs["ablation_comparison_csv"], comparison_rows)
    outputs["qc_report_md"].write_text(render_ablation_qc_report(qc_payload), encoding="utf-8")
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 4 ablation postprocess outputs.")
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--main-manifest", type=Path, default=DEFAULT_PHASE4_MAIN_MANIFEST)
    parser.add_argument("--ablation-manifest", type=Path, default=DEFAULT_ABLATION_MANIFEST)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--ablation-ids", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    seeds = {int(item.strip()) for item in args.seeds.split(",") if item.strip()} if args.seeds else None
    ablation_ids = {item.strip() for item in args.ablation_ids.split(",") if item.strip()} if args.ablation_ids else None

    outputs = generate_phase4_ablation_postprocess_outputs(
        summary_dir=args.summary_dir,
        main_manifest_path=args.main_manifest,
        ablation_manifest_path=args.ablation_manifest,
        run_root=args.run_root,
        output_dir=args.output_dir,
        seeds=seeds,
        ablation_ids=ablation_ids,
        include_anchor_full=True,
        dry_run=args.dry_run,
    )
    for key, path in outputs.items():
        print(f"[ABLATION_POST] {key}: {path}")


if __name__ == "__main__":
    main()
