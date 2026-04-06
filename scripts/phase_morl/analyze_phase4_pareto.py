"""Scenario-level Phase 4 Pareto/HV analysis for the confirmatory set."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_pareto import (
    OBJECTIVE_KEYS,
    compute_hypervolume,
    compute_pareto_front_mask,
    load_phase4_analysis_settings,
    normalize_objective_rows,
    save_pairwise_figure,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AGGREGATED_DIR = PROJECT_ROOT / "logs" / "eval" / "phase_morl_v2" / "aggregated"
DEFAULT_CHECKPOINT_LEVEL_CSV = DEFAULT_AGGREGATED_DIR / "checkpoint_level.csv"
DEFAULT_POLICY_LEVEL_CONFIRMATORY_CSV = DEFAULT_AGGREGATED_DIR / "policy_level_confirmatory.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "logs" / "eval" / "phase_morl_v2" / "pareto"
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "docs" / "figures"
DEFAULT_ANALYSIS_CONFIG = PROJECT_ROOT / "scripts" / "phase_morl" / "manifests" / "phase4_analysis_config.json"
DEFAULT_NUM_BOOTSTRAP = 1000
DEFAULT_RANDOM_SEED = 42
SEED_OUTLIER_REL_DEV_THRESH = 0.5


def _policy_sort_key(policy_name: str) -> tuple[int, str]:
    match = re.match(r"P(\d+)$", policy_name)
    if match:
        return (int(match.group(1)), policy_name)
    return (10**9, policy_name)


def _read_csv_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _to_float(value: str | float | int | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _to_int(value: str | int | None) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def load_policy_level_confirmatory_rows(path: Path) -> list[dict]:
    rows = []
    for raw in _read_csv_rows(path):
        row = dict(raw)
        row["policy"] = row["policy_id"]
        row["num_seeds"] = _to_int(raw.get("num_seeds"))
        for key in OBJECTIVE_KEYS + ("mean_vx_meas", "mean_vx_abs_err"):
            row[key] = _to_float(raw.get(key))
        rows.append(row)
    return rows


def load_checkpoint_level_rows(path: Path) -> list[dict]:
    rows = []
    for raw in _read_csv_rows(path):
        if raw.get("family") != "morl":
            continue
        if str(raw.get("official_hv_eligible", "")).lower() != "true":
            continue
        row = dict(raw)
        row["canonical_seed"] = _to_int(raw.get("canonical_seed"))
        for key in OBJECTIVE_KEYS + ("mean_vx_meas", "mean_vx_abs_err"):
            row[key] = _to_float(raw.get(key))
        rows.append(row)
    return rows


def build_scenario_payload(policy_rows: list[dict], bounds: dict[str, tuple[float, float]], ref_point: tuple[float, ...]) -> dict:
    normalized_rows = normalize_objective_rows(policy_rows, bounds=bounds)
    raw_objectives = [[row[key] for key in OBJECTIVE_KEYS] for row in policy_rows]
    pareto_mask = compute_pareto_front_mask(raw_objectives)
    pareto_rows = [row for row, is_pareto in zip(normalized_rows, pareto_mask) if is_pareto]
    hv = compute_hypervolume([row["normalized_objectives"] for row in pareto_rows], ref_point)

    scenario_id = policy_rows[0]["scenario_id"]
    return {
        "scenario_id": scenario_id,
        "scenario_name": policy_rows[0]["scenario_name"],
        "policy_rows": [dict(row) for row in policy_rows],
        "normalized_policy_rows": normalized_rows,
        "pareto_policies": [row["policy"] for row in pareto_rows],
        "hypervolume": float(hv),
        "num_policies": len(policy_rows),
    }


def build_confirmatory_scenario_hv_payload(scenario_payloads: dict[str, dict]) -> dict:
    payload = {}
    for scenario_id, scenario_payload in sorted(scenario_payloads.items()):
        payload[scenario_id] = {
            "scenario_name": scenario_payload["scenario_name"],
            "num_policies": scenario_payload["num_policies"],
            "pareto_policies": scenario_payload["pareto_policies"],
            "hypervolume": scenario_payload["hypervolume"],
        }
    return payload


def build_front_membership_frequency_rows(scenario_payloads: dict[str, dict]) -> list[dict]:
    policy_to_scenarios: dict[str, list[str]] = defaultdict(list)
    total_scenarios = len(scenario_payloads)
    for scenario_id, scenario_payload in scenario_payloads.items():
        for policy in scenario_payload["pareto_policies"]:
            policy_to_scenarios[policy].append(scenario_id)

    rows = []
    for policy in sorted(policy_to_scenarios, key=_policy_sort_key):
        scenarios_on_front = sorted(policy_to_scenarios[policy])
        rows.append(
            {
                "policy_id": policy,
                "front_count": len(scenarios_on_front),
                "front_rate": len(scenarios_on_front) / total_scenarios,
                "scenarios_on_front": ",".join(scenarios_on_front),
            }
        )
    return rows


def _aggregate_bootstrap_policy_rows(sampled_rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in sampled_rows:
        grouped[row["policy_id"]].append(row)

    aggregated = []
    for policy_id in sorted(grouped, key=_policy_sort_key):
        rows = grouped[policy_id]
        aggregated.append(
            {
                "policy": policy_id,
                "scenario_id": rows[0]["scenario_id"],
                "scenario_name": rows[0]["scenario_name"],
                "J_speed": float(np.mean([row["J_speed"] for row in rows])),
                "J_energy": float(np.mean([row["J_energy"] for row in rows])),
                "J_smooth": float(np.mean([row["J_smooth"] for row in rows])),
                "J_stable": float(np.mean([row["J_stable"] for row in rows])),
            }
        )
    return aggregated


def bootstrap_scenario(
    checkpoint_rows: list[dict],
    *,
    bounds: dict[str, tuple[float, float]],
    ref_point: tuple[float, ...],
    num_bootstrap: int,
    random_seed: int,
) -> tuple[list[dict], dict]:
    rng = np.random.default_rng(random_seed)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in checkpoint_rows:
        grouped[row["policy_id"]].append(row)

    front_counts = {policy_id: 0 for policy_id in grouped}
    hv_values: list[float] = []
    for _ in range(num_bootstrap):
        sampled_rows = []
        for policy_id, rows in grouped.items():
            indices = rng.integers(0, len(rows), size=len(rows))
            sampled_rows.extend(rows[idx] for idx in indices)
        aggregated = _aggregate_bootstrap_policy_rows(sampled_rows)
        scenario_payload = build_scenario_payload(aggregated, bounds=bounds, ref_point=ref_point)
        for policy_id in scenario_payload["pareto_policies"]:
            front_counts[policy_id] += 1
        hv_values.append(scenario_payload["hypervolume"])

    scenario_id = checkpoint_rows[0]["scenario_id"]
    front_rows = [
        {
            "scenario_id": scenario_id,
            "policy_id": policy_id,
            "p_on_front": front_counts[policy_id] / num_bootstrap,
            "front_count": front_counts[policy_id],
        }
        for policy_id in sorted(front_counts, key=_policy_sort_key)
    ]
    hv_ci = {
        "num_bootstrap": num_bootstrap,
        "mean_hv": float(np.mean(hv_values)),
        "ci_lower": float(np.percentile(hv_values, 2.5)),
        "ci_upper": float(np.percentile(hv_values, 97.5)),
    }
    return front_rows, hv_ci


def detect_seed_outliers(checkpoint_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in checkpoint_rows:
        grouped[(row["policy_id"], row["scenario_id"])].append(row)

    outliers = []
    for (policy_id, scenario_id), rows in grouped.items():
        if len(rows) < 2:
            continue
        mean_value = float(np.mean([row["mean_vx_meas"] for row in rows]))
        if mean_value == 0.0:
            continue
        for row in rows:
            rel_dev = abs(row["mean_vx_meas"] - mean_value) / abs(mean_value)
            if rel_dev > SEED_OUTLIER_REL_DEV_THRESH:
                outliers.append(
                    {
                        "policy_id": policy_id,
                        "scenario_id": scenario_id,
                        "canonical_seed": row["canonical_seed"],
                        "mean_vx_meas": row["mean_vx_meas"],
                        "group_mean_vx_meas": mean_value,
                        "rel_dev_from_mean": rel_dev,
                    }
                )
    outliers.sort(key=lambda row: (row["policy_id"], row["scenario_id"], row["canonical_seed"]))
    return outliers


def build_robustness_summary_rows(
    front_membership_rows: list[dict],
    bootstrap_front_rows: list[dict],
    seed_outliers: list[dict],
) -> list[dict]:
    point_rows = {row["policy_id"]: row for row in front_membership_rows}
    bootstrap_grouped: dict[str, list[float]] = defaultdict(list)
    for row in bootstrap_front_rows:
        bootstrap_grouped[row["policy_id"]].append(float(row["p_on_front"]))

    unstable_grouped: dict[str, list[str]] = defaultdict(list)
    for row in seed_outliers:
        unstable_grouped[row["policy_id"]].append(row["scenario_id"])

    rows = []
    for policy_id in sorted(point_rows, key=_policy_sort_key):
        unstable_scenarios = sorted(set(unstable_grouped.get(policy_id, [])))
        rows.append(
            {
                "policy_id": policy_id,
                "front_count": point_rows[policy_id]["front_count"],
                "front_rate": point_rows[policy_id]["front_rate"],
                "bootstrap_mean_p_on_front": float(np.mean(bootstrap_grouped.get(policy_id, [0.0]))),
                "unstable_scenario_count": len(unstable_scenarios),
                "unstable_scenarios": ",".join(unstable_scenarios),
            }
        )
    return rows


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


def render_phase4_figures(scenario_payloads: dict[str, dict], *, figure_dir: Path, bounds: dict[str, tuple[float, float]]) -> dict[str, object]:
    figure_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hv_bar_path = figure_dir / "phase4_hv_bar.png"
    scenarios = sorted(scenario_payloads)
    hv_values = [scenario_payloads[s]["hypervolume"] for s in scenarios]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(scenarios, hv_values, color="#1f77b4")
    ax.set_ylabel("Hypervolume")
    ax.set_title("Phase 4 Confirmatory Scenario HV")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(hv_bar_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    pareto_paths: dict[str, Path] = {}
    for scenario_id, scenario_payload in scenario_payloads.items():
        path = figure_dir / f"phase4_pareto_{scenario_id.lower()}.png"
        save_pairwise_figure(
            scenario_payload["policy_rows"],
            set(scenario_payload["pareto_policies"]),
            path,
        )
        pareto_paths[scenario_id] = path

    return {
        "phase4_hv_bar_png": hv_bar_path,
        "phase4_pareto_pngs": pareto_paths,
    }


def _output_paths(output_dir: Path, figure_dir: Path, scenario_ids: list[str]) -> dict[str, object]:
    return {
        "confirmatory_scenario_hv_json": output_dir / "confirmatory_scenario_hv.json",
        "front_membership_frequency_csv": output_dir / "front_membership_frequency.csv",
        "robustness_summary_csv": output_dir / "robustness_summary.csv",
        "bootstrap_front_membership_csv": output_dir / "bootstrap_front_membership.csv",
        "bootstrap_hv_ci_json": output_dir / "bootstrap_hv_ci.json",
        "phase4_hv_bar_png": figure_dir / "phase4_hv_bar.png",
        "phase4_pareto_pngs": {scenario_id: figure_dir / f"phase4_pareto_{scenario_id.lower()}.png" for scenario_id in scenario_ids},
    }


def generate_phase4_pareto_outputs(
    *,
    checkpoint_level_csv: Path,
    policy_level_confirmatory_csv: Path,
    analysis_config_path: Path,
    output_dir: Path,
    figure_dir: Path,
    num_bootstrap: int = DEFAULT_NUM_BOOTSTRAP,
    random_seed: int = DEFAULT_RANDOM_SEED,
    dry_run: bool = False,
) -> dict[str, object]:
    policy_rows = load_policy_level_confirmatory_rows(policy_level_confirmatory_csv)
    scenario_ids = sorted({row["scenario_id"] for row in policy_rows}) if policy_rows else []
    outputs = _output_paths(output_dir, figure_dir, scenario_ids)
    if dry_run:
        return outputs

    checkpoint_rows = load_checkpoint_level_rows(checkpoint_level_csv)
    bounds, ref_point = load_phase4_analysis_settings(analysis_config_path)

    scenario_payloads = {}
    for scenario_id in scenario_ids:
        scenario_policy_rows = [row for row in policy_rows if row["scenario_id"] == scenario_id]
        scenario_payloads[scenario_id] = build_scenario_payload(scenario_policy_rows, bounds=bounds, ref_point=ref_point)

    front_rows = build_front_membership_frequency_rows(scenario_payloads)

    bootstrap_front_rows: list[dict] = []
    bootstrap_hv_ci = {}
    for index, scenario_id in enumerate(scenario_ids):
        scenario_checkpoint_rows = [row for row in checkpoint_rows if row["scenario_id"] == scenario_id]
        front_membership_rows, hv_ci = bootstrap_scenario(
            scenario_checkpoint_rows,
            bounds=bounds,
            ref_point=ref_point,
            num_bootstrap=num_bootstrap,
            random_seed=random_seed + index,
        )
        bootstrap_front_rows.extend(front_membership_rows)
        bootstrap_hv_ci[scenario_id] = hv_ci

    seed_outliers = detect_seed_outliers(checkpoint_rows)
    robustness_rows = build_robustness_summary_rows(front_rows, bootstrap_front_rows, seed_outliers)

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs["confirmatory_scenario_hv_json"].write_text(
        json.dumps(build_confirmatory_scenario_hv_payload(scenario_payloads), ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    _write_csv(outputs["front_membership_frequency_csv"], front_rows)
    _write_csv(outputs["robustness_summary_csv"], robustness_rows)
    _write_csv(outputs["bootstrap_front_membership_csv"], bootstrap_front_rows)
    outputs["bootstrap_hv_ci_json"].write_text(json.dumps(bootstrap_hv_ci, ensure_ascii=True, indent=2), encoding="utf-8")

    figure_outputs = render_phase4_figures(scenario_payloads, figure_dir=figure_dir, bounds=bounds)
    outputs.update(figure_outputs)
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 4 confirmatory scenario Pareto/HV analysis.")
    parser.add_argument("--checkpoint-level-csv", type=Path, default=DEFAULT_CHECKPOINT_LEVEL_CSV)
    parser.add_argument("--policy-level-confirmatory-csv", type=Path, default=DEFAULT_POLICY_LEVEL_CONFIRMATORY_CSV)
    parser.add_argument("--analysis-config", type=Path, default=DEFAULT_ANALYSIS_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--num-bootstrap", type=int, default=DEFAULT_NUM_BOOTSTRAP)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outputs = generate_phase4_pareto_outputs(
        checkpoint_level_csv=args.checkpoint_level_csv,
        policy_level_confirmatory_csv=args.policy_level_confirmatory_csv,
        analysis_config_path=args.analysis_config,
        output_dir=args.output_dir,
        figure_dir=args.figure_dir,
        num_bootstrap=args.num_bootstrap,
        random_seed=args.random_seed,
        dry_run=args.dry_run,
    )
    print(f"[PHASE4-PARETO] confirmatory_scenario_hv_json: {outputs['confirmatory_scenario_hv_json']}")
    print(f"[PHASE4-PARETO] front_membership_frequency_csv: {outputs['front_membership_frequency_csv']}")
    print(f"[PHASE4-PARETO] robustness_summary_csv: {outputs['robustness_summary_csv']}")
    print(f"[PHASE4-PARETO] bootstrap_front_membership_csv: {outputs['bootstrap_front_membership_csv']}")
    print(f"[PHASE4-PARETO] bootstrap_hv_ci_json: {outputs['bootstrap_hv_ci_json']}")
    print(f"[PHASE4-PARETO] phase4_hv_bar_png: {outputs['phase4_hv_bar_png']}")


if __name__ == "__main__":
    main()
