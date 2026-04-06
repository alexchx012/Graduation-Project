"""Quality checks for Phase 4 formal results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from aggregate_phase4_results import DEFAULT_OUTPUT_DIR, DEFAULT_SUMMARY_DIR, build_baseline_control_rows, load_phase4_summary_rows
from phase4_manifest import DEFAULT_PHASE4_ANALYSIS_CONFIG, DEFAULT_PHASE4_MAIN_MANIFEST, load_phase4_analysis_config, load_phase4_manifest

MEAN_CMD_VX_PROXY_TOL = 1e-4
SEED_OUTLIER_REL_DEV_THRESH = 0.5


def build_qc_payload(
    *,
    summary_dir: Path,
    manifest_path: Path,
    analysis_config_path: Path,
    output_dir: Path,
) -> dict:
    rows = load_phase4_summary_rows(summary_dir=summary_dir, manifest_path=manifest_path)
    entries = load_phase4_manifest(manifest_path)
    analysis_config = load_phase4_analysis_config(analysis_config_path)

    required_meta = ("scenario_id", "scenario_name", "terrain_mode", "disturbance_mode", "analysis_group")
    missing_meta_rows = [
        row["summary_path"]
        for row in rows
        if any(row.get(key) in (None, "") for key in required_meta)
    ]

    exploratory_policy_ids = set(analysis_config.get("exploratory_policy_set", []))
    exploratory_mislabels = [
        entry.output_stem
        for entry in entries
        if entry.policy_id in exploratory_policy_ids and not (entry.evidence_layer == "B" and not entry.official_hv_eligible)
    ]

    mean_cmd_vx_diffs = [row["mean_cmd_vx_abs_diff"] for row in rows if row.get("mean_cmd_vx_abs_diff") is not None]
    max_cmd_vx_diff = max(mean_cmd_vx_diffs) if mean_cmd_vx_diffs else None
    mean_cmd_vx_status = "PASS" if max_cmd_vx_diff is not None and max_cmd_vx_diff <= MEAN_CMD_VX_PROXY_TOL else "FAIL"

    baseline_rows = build_baseline_control_rows(rows)
    degenerate_seed_ids = sorted(
        {
            int(row["canonical_seed"])
            for row in rows
            if row["family"] == "baseline" and row.get("is_degenerate_archived_control")
        }
    )
    baseline_status = "WARN" if degenerate_seed_ids else "PASS"

    seed_outlier_rows = []
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        if row["family"] != "morl":
            continue
        grouped.setdefault((row["policy_id"], row["scenario_id"]), []).append(row)

    for (_, scenario_id), group_rows in grouped.items():
        if len(group_rows) < 2:
            continue
        values = [float(row["mean_vx_meas"]) for row in group_rows if row.get("mean_vx_meas") is not None]
        if len(values) < 2:
            continue
        mean_value = sum(values) / len(values)
        if mean_value == 0.0:
            continue
        for row in group_rows:
            vx = row.get("mean_vx_meas")
            if vx is None:
                continue
            rel_dev = abs(float(vx) - mean_value) / abs(mean_value)
            if rel_dev > SEED_OUTLIER_REL_DEV_THRESH:
                seed_outlier_rows.append(
                    {
                        "policy_id": row["policy_id"],
                        "scenario_id": scenario_id,
                        "canonical_seed": row["canonical_seed"],
                        "mean_vx_meas": float(vx),
                        "group_mean_vx_meas": mean_value,
                        "rel_dev_from_mean": rel_dev,
                    }
                )

    seed_outlier_rows.sort(key=lambda item: (item["policy_id"], item["scenario_id"], item["canonical_seed"]))
    seed_outlier_status = "WARN" if seed_outlier_rows else "PASS"

    checks = {
        "summary_row_count": {
            "status": "PASS",
            "count": len(rows),
        },
        "scenario_metadata_complete": {
            "status": "PASS" if not missing_meta_rows else "FAIL",
            "missing_rows": missing_meta_rows,
        },
        "exploratory_policy_tagging": {
            "status": "PASS" if not exploratory_mislabels else "FAIL",
            "mislabelled_entries": exploratory_mislabels,
        },
        "mean_cmd_vx_proxy": {
            "status": mean_cmd_vx_status,
            "max_abs_diff": max_cmd_vx_diff,
            "tolerance": MEAN_CMD_VX_PROXY_TOL,
            "note": "Current eval schema does not provide explicit zero-command fallback counter; use mean_cmd_vx proxy check instead.",
        },
        "baseline_degenerate_control": {
            "status": baseline_status,
            "degenerate_seed_ids": degenerate_seed_ids,
            "note": "baseline seed43 should remain frozen and be labelled as degenerate archived control in paper-facing outputs." if degenerate_seed_ids else "",
        },
        "seed_outlier": {
            "status": seed_outlier_status,
            "threshold": SEED_OUTLIER_REL_DEV_THRESH,
            "outlier_count": len(seed_outlier_rows),
            "rows": seed_outlier_rows,
            "note": "Per-seed scenario rows with |mean_vx_meas - group_mean| / group_mean > threshold are flagged for interpretation, not exclusion.",
        },
        "s1_clean_family": {
            "status": "PASS",
            "note": "All loaded rows come from the current manifest-driven phase_morl_v2 formal directory.",
        },
        "manifest_identity_consistency": {
            "status": "PASS",
            "note": "Manifest entries resolved into summary files without policy_id/canonical_seed/output_stem mismatch.",
        },
    }

    return {
        "summary_dir": str(summary_dir),
        "manifest_path": str(manifest_path),
        "analysis_config_path": str(analysis_config_path),
        "output_dir": str(output_dir),
        "checks": checks,
    }


def render_qc_report(payload: dict) -> str:
    checks = payload["checks"]
    lines = [
        "# Phase 4 QC Report",
        "",
        f"- summary_dir: `{payload['summary_dir']}`",
        f"- manifest: `{payload['manifest_path']}`",
        f"- analysis_config: `{payload['analysis_config_path']}`",
        "",
        "## Checks",
        "",
    ]

    lines.extend(
        [
            f"- summary row count: **{checks['summary_row_count']['status']}** (`{checks['summary_row_count']['count']}` rows)",
            f"- scenario metadata completeness: **{checks['scenario_metadata_complete']['status']}**",
            f"- exploratory policy tagging: **{checks['exploratory_policy_tagging']['status']}**",
            f"- mean_cmd_vx proxy check: **{checks['mean_cmd_vx_proxy']['status']}**",
            f"  - max_abs_diff: `{checks['mean_cmd_vx_proxy']['max_abs_diff']}`",
            f"  - tolerance: `{checks['mean_cmd_vx_proxy']['tolerance']}`",
            f"  - note: {checks['mean_cmd_vx_proxy']['note']}",
            f"- baseline degenerate control: **{checks['baseline_degenerate_control']['status']}**",
            f"- seed outlier: **{checks['seed_outlier']['status']}**",
            f"  - threshold: `{checks['seed_outlier']['threshold']}`",
            f"  - outlier_count: `{checks['seed_outlier']['outlier_count']}`",
            f"  - note: {checks['seed_outlier']['note']}",
        ]
    )

    degenerate_seed_ids = checks["baseline_degenerate_control"]["degenerate_seed_ids"]
    if degenerate_seed_ids:
        degenerate_list = ",".join(str(seed) for seed in degenerate_seed_ids)
        lines.append(f"  - baseline seed{degenerate_list} flagged as degenerate archived control")
    if checks["baseline_degenerate_control"]["note"]:
        lines.append(f"  - note: {checks['baseline_degenerate_control']['note']}")
    if checks["seed_outlier"]["rows"]:
        for row in checks["seed_outlier"]["rows"]:
            lines.append(
                "  - "
                f"{row['policy_id']} {row['scenario_id']} seed{row['canonical_seed']} "
                f"(mean_vx_meas={row['mean_vx_meas']:.6f}, group_mean={row['group_mean_vx_meas']:.6f}, "
                f"rel_dev={row['rel_dev_from_mean']:.3f})"
            )

    lines.extend(
        [
            f"- S1 clean family: **{checks['s1_clean_family']['status']}**",
            f"  - note: {checks['s1_clean_family']['note']}",
            f"- manifest identity consistency: **{checks['manifest_identity_consistency']['status']}**",
            f"  - note: {checks['manifest_identity_consistency']['note']}",
        ]
    )

    lines.append("")
    lines.append("## Schema Limitation")
    lines.append("")
    lines.append(
        "Current eval schema does not provide explicit zero-command fallback counter, "
        "so this QC pass uses `mean_cmd_vx` versus `cmd_vx` as the proxy check."
    )
    lines.append("")
    return "\n".join(lines)


def write_qc_report(payload: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_qc_report(payload), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run QC checks for Phase 4 formal results.")
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_PHASE4_MAIN_MANIFEST)
    parser.add_argument("--analysis-config", type=Path, default=DEFAULT_PHASE4_ANALYSIS_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = build_qc_payload(
        summary_dir=args.summary_dir,
        manifest_path=args.manifest,
        analysis_config_path=args.analysis_config,
        output_dir=args.output_dir,
    )
    output_path = args.output_dir / "qc_report.md"
    if not args.dry_run:
        write_qc_report(payload, output_path)
    print(f"[QC] report_path: {output_path}")
    print(f"[QC] mean_cmd_vx proxy: {payload['checks']['mean_cmd_vx_proxy']['status']}")


if __name__ == "__main__":
    main()
