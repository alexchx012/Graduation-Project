"""Aggregate Phase 4 formal eval JSONs into CSV artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from phase4_manifest import DEFAULT_PHASE4_ANALYSIS_CONFIG, DEFAULT_PHASE4_MAIN_MANIFEST, load_phase4_analysis_config, load_phase4_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY_DIR = PROJECT_ROOT / "logs" / "eval" / "phase_morl_v2"
DEFAULT_OUTPUT_DIR = DEFAULT_SUMMARY_DIR / "aggregated"

OBJECTIVE_KEYS = ("J_speed", "J_energy", "J_smooth", "J_stable")
SUPPLEMENTAL_KEYS = ("success_rate", "mean_base_contact_rate", "mean_timeout_rate", "recovery_time")
SCENARIO_META_KEYS = ("scenario_id", "scenario_name", "terrain_mode", "disturbance_mode", "analysis_group", "task")
CHECKPOINT_BASE_FIELDS = (
    "family",
    "policy_id",
    "canonical_seed",
    "output_stem",
    "source_state",
    "evidence_layer",
    "official_hv_eligible",
    "run_dir",
    "checkpoint",
    "summary_path",
)
COMMAND_KEYS = ("cmd_vx", "mean_cmd_vx", "mean_cmd_vx_abs_diff")
TRACKING_KEYS = ("mean_vx_meas", "mean_vx_abs_err")
RUNTIME_KEYS = ("eval_steps", "warmup_steps", "effective_steps", "effective_env_steps", "step_dt", "elapsed_seconds")


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values))


def _population_std(values: list[float]) -> float:
    mean_value = _mean(values)
    return float(math.sqrt(sum((value - mean_value) ** 2 for value in values) / len(values)))


def _format_bool(value: bool) -> str:
    return "true" if value else "false"


def _output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "checkpoint_level_csv": output_dir / "checkpoint_level.csv",
        "policy_level_confirmatory_csv": output_dir / "policy_level_confirmatory.csv",
        "policy_level_exploratory_csv": output_dir / "policy_level_exploratory.csv",
        "baseline_control_csv": output_dir / "baseline_control.csv",
        "qc_report_md": output_dir / "qc_report.md",
    }


def _summary_paths_for_entry(summary_dir: Path, output_stem: str) -> list[Path]:
    return sorted(summary_dir.glob(f"{output_stem}_S*.json"))


def _load_summary_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def is_degenerate_baseline_checkpoint(row: dict) -> bool:
    if row["family"] != "baseline":
        return False
    mean_vx_meas = row.get("mean_vx_meas")
    j_smooth = row.get("J_smooth")
    if mean_vx_meas is None or j_smooth is None:
        return False
    # Degenerate archived controls in this project present as "almost not moving"
    # together with abnormally tiny smoothness cost, i.e. near-constant standing.
    return bool(mean_vx_meas < 0.05 and j_smooth < 0.1)


def load_phase4_summary_rows(summary_dir: Path, manifest_path: Path) -> list[dict]:
    if not summary_dir.exists():
        raise FileNotFoundError(f"Summary directory not found: {summary_dir}")

    rows: list[dict] = []
    for entry in load_phase4_manifest(manifest_path):
        summary_paths = _summary_paths_for_entry(summary_dir, entry.output_stem)
        if not summary_paths:
            raise FileNotFoundError(f"Missing summary JSONs for manifest entry: {entry.output_stem}")

        for summary_path in summary_paths:
            data = _load_summary_json(summary_path)
            row = {
                "family": entry.family,
                "policy_id": entry.policy_id,
                "canonical_seed": entry.canonical_seed,
                "output_stem": entry.output_stem,
                "source_state": entry.source_state,
                "evidence_layer": entry.evidence_layer,
                "official_hv_eligible": entry.official_hv_eligible,
                "run_dir": entry.run_dir,
                "checkpoint": entry.checkpoint,
                "summary_path": str(summary_path),
            }
            for key in SCENARIO_META_KEYS + COMMAND_KEYS[:-1] + TRACKING_KEYS + OBJECTIVE_KEYS + SUPPLEMENTAL_KEYS + RUNTIME_KEYS:
                row[key] = data.get(key)
            if row.get("cmd_vx") is not None and row.get("mean_cmd_vx") is not None:
                row["mean_cmd_vx_abs_diff"] = abs(float(row["mean_cmd_vx"]) - float(row["cmd_vx"]))
            else:
                row["mean_cmd_vx_abs_diff"] = None
            row["is_degenerate_archived_control"] = is_degenerate_baseline_checkpoint(row)
            rows.append(row)

    return sorted(rows, key=lambda item: (item["family"], item["policy_id"], item["canonical_seed"], item.get("scenario_id") or "", item["output_stem"]))


def build_checkpoint_level_rows(rows: list[dict]) -> list[dict]:
    checkpoint_rows: list[dict] = []
    for row in rows:
        checkpoint_row = {}
        for key in CHECKPOINT_BASE_FIELDS + SCENARIO_META_KEYS + COMMAND_KEYS + TRACKING_KEYS + OBJECTIVE_KEYS + SUPPLEMENTAL_KEYS + RUNTIME_KEYS:
            value = row.get(key)
            if isinstance(value, bool):
                value = _format_bool(value)
            checkpoint_row[key] = value
        checkpoint_row["is_degenerate_archived_control"] = _format_bool(bool(row.get("is_degenerate_archived_control")))
        checkpoint_rows.append(checkpoint_row)
    return checkpoint_rows


def _aggregate_group(rows: list[dict]) -> dict:
    first = rows[0]
    aggregated = {
        "family": first["family"],
        "policy_id": first["policy_id"],
        "scenario_id": first["scenario_id"],
        "scenario_name": first["scenario_name"],
        "terrain_mode": first["terrain_mode"],
        "disturbance_mode": first["disturbance_mode"],
        "analysis_group": first["analysis_group"],
        "task": first["task"],
        "num_seeds": len(rows),
        "seed_list": ",".join(str(seed) for seed in sorted(row["canonical_seed"] for row in rows)),
        "output_stems": ",".join(row["output_stem"] for row in sorted(rows, key=lambda item: item["canonical_seed"])),
        "source_states": ",".join(row["source_state"] for row in sorted(rows, key=lambda item: item["canonical_seed"])),
        "evidence_layer": first["evidence_layer"],
        "official_hv_eligible": _format_bool(bool(first["official_hv_eligible"])),
    }

    for key in COMMAND_KEYS[:-1] + ("mean_cmd_vx_abs_diff",) + TRACKING_KEYS + OBJECTIVE_KEYS + SUPPLEMENTAL_KEYS:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        if not values:
            aggregated[key] = None
            aggregated[f"{key}_std"] = None
            continue
        aggregated[key] = _mean(values)
        aggregated[f"{key}_std"] = _population_std(values)

    return aggregated


def build_policy_level_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        key = (row["family"], row["policy_id"], row["scenario_id"])
        grouped.setdefault(key, []).append(row)

    aggregated_rows = [_aggregate_group(group_rows) for _, group_rows in sorted(grouped.items())]
    return sorted(aggregated_rows, key=lambda item: (item["family"], item["policy_id"], item["scenario_id"]))


def build_baseline_control_rows(rows: list[dict]) -> list[dict]:
    baseline_rows = [row for row in build_policy_level_rows(rows) if row["family"] == "baseline"]
    checkpoint_rows = [row for row in rows if row["family"] == "baseline"]

    for baseline_row in baseline_rows:
        scenario_id = baseline_row["scenario_id"]
        scenario_checkpoints = [row for row in checkpoint_rows if row["scenario_id"] == scenario_id]
        degenerate_ids = sorted(row["canonical_seed"] for row in scenario_checkpoints if row.get("is_degenerate_archived_control"))
        effective_ids = sorted(row["canonical_seed"] for row in scenario_checkpoints if row["canonical_seed"] not in degenerate_ids)
        baseline_row["has_degenerate_seed"] = _format_bool(bool(degenerate_ids))
        baseline_row["degenerate_seed_ids"] = ",".join(str(seed) for seed in degenerate_ids)
        baseline_row["narrative_effective_seed_ids"] = ",".join(str(seed) for seed in effective_ids)
        if degenerate_ids:
            baseline_row["paper_note"] = (
                f"baseline includes degenerate archived control seed(s) {baseline_row['degenerate_seed_ids']}; "
                f"effective narrative should use seed(s) {baseline_row['narrative_effective_seed_ids']}"
            )
        else:
            baseline_row["paper_note"] = ""

    return baseline_rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def generate_phase4_outputs(
    *,
    summary_dir: Path,
    manifest_path: Path,
    analysis_config_path: Path,
    output_dir: Path,
    dry_run: bool = False,
) -> dict[str, Path]:
    outputs = _output_paths(output_dir)
    if dry_run:
        return outputs

    rows = load_phase4_summary_rows(summary_dir=summary_dir, manifest_path=manifest_path)
    analysis_config = load_phase4_analysis_config(analysis_config_path)
    aggregated_rows = build_policy_level_rows(rows)

    confirmatory_policy_ids = set(analysis_config.get("official_policy_set", []))
    exploratory_policy_ids = set(analysis_config.get("exploratory_policy_set", []))

    checkpoint_rows = build_checkpoint_level_rows(rows)
    confirmatory_rows = [
        row for row in aggregated_rows if row["family"] == "morl" and row["policy_id"] in confirmatory_policy_ids
    ]
    exploratory_rows = [
        row for row in aggregated_rows if row["family"] == "morl" and row["policy_id"] in exploratory_policy_ids
    ]
    baseline_rows = build_baseline_control_rows(rows)

    _write_csv(outputs["checkpoint_level_csv"], checkpoint_rows)
    _write_csv(outputs["policy_level_confirmatory_csv"], confirmatory_rows)
    _write_csv(outputs["policy_level_exploratory_csv"], exploratory_rows)
    _write_csv(outputs["baseline_control_csv"], baseline_rows)

    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate Phase 4 formal results into CSV outputs.")
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_PHASE4_MAIN_MANIFEST)
    parser.add_argument("--analysis-config", type=Path, default=DEFAULT_PHASE4_ANALYSIS_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outputs = generate_phase4_outputs(
        summary_dir=args.summary_dir,
        manifest_path=args.manifest,
        analysis_config_path=args.analysis_config,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
    )
    for key, path in outputs.items():
        print(f"[AGGREGATE] {key}: {path}")


if __name__ == "__main__":
    main()
