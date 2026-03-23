"""Phase MORL M11: Pareto analysis and visualization."""

from __future__ import annotations

import argparse
import json
import math
import re
from itertools import combinations
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY_DIR = PROJECT_ROOT / "logs" / "eval" / "phase_morl"
DEFAULT_RUN_ROOT = PROJECT_ROOT / "logs" / "rsl_rl" / "unitree_go1_rough"
DEFAULT_OUTPUT_JSON = DEFAULT_SUMMARY_DIR / "pareto_analysis.json"
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "docs" / "figures"

OBJECTIVE_KEYS = ("J_speed", "J_energy", "J_smooth", "J_stable")
SUPPLEMENTAL_KEYS = ("success_rate", "mean_base_contact_rate", "mean_timeout_rate", "recovery_time")

# Frozen reporting bounds for the four minimization objectives.
FROZEN_NORMALIZATION_BOUNDS = {
    "J_speed": (0.0, 1.2),
    "J_energy": (0.0, 2500.0),
    "J_smooth": (0.0, 2.0),
    "J_stable": (0.0, 0.5),
}

# Must be strictly worse than any normalized point for minimization-space HV.
DEFAULT_REF_POINT = tuple(1.1 for _ in OBJECTIVE_KEYS)

POLICY_WEIGHT_MAP = {
    "P1": [0.7, 0.1, 0.1, 0.1],
    "P2": [0.1, 0.7, 0.1, 0.1],
    "P3": [0.1, 0.1, 0.7, 0.1],
    "P4": [0.1, 0.1, 0.1, 0.7],
    "P5": [0.4, 0.3, 0.2, 0.1],
    "P6": [0.5, 0.3, 0.1, 0.1],
    "P7": [0.3, 0.3, 0.2, 0.2],
    "P8": [0.2, 0.4, 0.2, 0.2],
    "P9": [0.3, 0.2, 0.3, 0.2],
    "P10": [0.2, 0.2, 0.2, 0.4],
}

_RUN_NAME_PATTERN = re.compile(r"morl_p(\d+)_seed(\d+)$")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze MORL eval results and produce Pareto/HV artifacts.")
    parser.add_argument("--summary_dir", type=Path, default=DEFAULT_SUMMARY_DIR, help="Directory containing eval JSONs.")
    parser.add_argument("--run_root", type=Path, default=DEFAULT_RUN_ROOT, help="Directory containing active run folders.")
    parser.add_argument("--output_json", type=Path, default=DEFAULT_OUTPUT_JSON, help="Output JSON artifact path.")
    parser.add_argument("--figure_dir", type=Path, default=DEFAULT_FIGURE_DIR, help="Directory for generated figures.")
    return parser


def _policy_sort_key(policy_name: str) -> tuple[int, str]:
    match = re.match(r"P(\d+)$", policy_name)
    if match:
        return (int(match.group(1)), policy_name)
    return (10**9, policy_name)


def _extract_policy_and_seed(name: str) -> tuple[str, int]:
    match = _RUN_NAME_PATTERN.search(name)
    if not match:
        raise ValueError(f"Unable to parse policy/seed from name: {name}")
    return f"P{match.group(1)}", int(match.group(2))


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values))


def _population_std(values: list[float]) -> float:
    mean_value = _mean(values)
    return float(math.sqrt(sum((value - mean_value) ** 2 for value in values) / len(values)))


def discover_active_run_names(run_root: Path) -> list[str]:
    if not run_root.exists():
        raise FileNotFoundError(f"Run root not found: {run_root}")
    return sorted(
        path.name
        for path in run_root.iterdir()
        if path.is_dir() and _RUN_NAME_PATTERN.search(path.name)
    )


def load_run_rows(summary_dir: Path, run_root: Path) -> list[dict]:
    if not summary_dir.exists():
        raise FileNotFoundError(f"Summary directory not found: {summary_dir}")

    run_rows: list[dict] = []
    for run_name in discover_active_run_names(run_root):
        summary_path = summary_dir / f"{run_name}.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing eval summary for active run: {summary_path}")

        data = json.loads(summary_path.read_text(encoding="utf-8"))
        policy_name, seed = _extract_policy_and_seed(run_name)
        row = {
            "run": run_name,
            "policy": policy_name,
            "seed": seed,
            "policy_id": data.get("policy_id"),
            "summary_path": str(summary_path),
            "weights": POLICY_WEIGHT_MAP.get(policy_name),
        }
        for key in OBJECTIVE_KEYS + SUPPLEMENTAL_KEYS + ("elapsed_seconds",):
            row[key] = data.get(key)
        run_rows.append(row)
    return run_rows


def aggregate_policy_rows(run_rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in run_rows:
        grouped.setdefault(row["policy"], []).append(row)

    policy_rows: list[dict] = []
    for policy_name in sorted(grouped, key=_policy_sort_key):
        rows = sorted(grouped[policy_name], key=lambda row: row["seed"])
        aggregated = {
            "policy": policy_name,
            "policy_id": policy_name.lower(),
            "num_seeds": len(rows),
            "seeds": [row["seed"] for row in rows],
            "weights": POLICY_WEIGHT_MAP.get(policy_name),
            "source_runs": [row["run"] for row in rows if "run" in row],
        }
        for key in OBJECTIVE_KEYS + SUPPLEMENTAL_KEYS:
            values = [float(row[key]) for row in rows if row.get(key) is not None]
            if not values:
                aggregated[key] = None
                aggregated[f"{key}_std"] = None
                continue
            aggregated[key] = _mean(values)
            aggregated[f"{key}_std"] = _population_std(values)
        policy_rows.append(aggregated)
    return policy_rows


def normalize_objective_rows(rows: list[dict], bounds: dict[str, tuple[float, float]]) -> list[dict]:
    normalized_rows: list[dict] = []
    for row in rows:
        normalized_metrics = {}
        normalized_objectives = []
        for key in OBJECTIVE_KEYS:
            lower, upper = bounds[key]
            if upper <= lower:
                raise ValueError(f"Invalid normalization bounds for {key}: {(lower, upper)}")
            raw_value = float(row[key])
            normalized_value = (raw_value - lower) / (upper - lower)
            normalized_value = min(1.0, max(0.0, normalized_value))
            normalized_metrics[key] = float(normalized_value)
            normalized_objectives.append(float(normalized_value))

        normalized_row = dict(row)
        normalized_row["normalized_metrics"] = normalized_metrics
        normalized_row["normalized_objectives"] = normalized_objectives
        normalized_rows.append(normalized_row)
    return normalized_rows


def compute_pareto_front_mask(objectives) -> list[bool]:
    values = np.asarray(objectives, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"Expected 2D objective matrix, got shape {values.shape}")

    num_points = values.shape[0]
    mask = [True] * num_points
    for i in range(num_points):
        for j in range(num_points):
            if i == j:
                continue
            if np.all(values[j] <= values[i]) and np.any(values[j] < values[i]):
                mask[i] = False
                break
    return mask


def compute_hypervolume(points, ref_point) -> float:
    point_array = np.asarray(points, dtype=float)
    ref = np.asarray(ref_point, dtype=float)
    if point_array.ndim != 2:
        raise ValueError(f"Expected 2D point matrix, got shape {point_array.shape}")
    if point_array.shape[1] != ref.shape[0]:
        raise ValueError(f"Point dimension {point_array.shape[1]} does not match ref point {ref.shape[0]}")
    if point_array.shape[0] == 0:
        return 0.0

    unique_points = np.unique(point_array, axis=0)
    hypervolume = 0.0
    for subset_size in range(1, len(unique_points) + 1):
        sign = 1.0 if subset_size % 2 == 1 else -1.0
        for subset_indices in combinations(range(len(unique_points)), subset_size):
            lower = np.max(unique_points[list(subset_indices)], axis=0)
            edge_lengths = ref - lower
            if np.any(edge_lengths <= 0.0):
                continue
            hypervolume += sign * float(np.prod(edge_lengths))
    return float(hypervolume)


def build_analysis_payload(
    run_rows: list[dict],
    bounds: dict[str, tuple[float, float]] = FROZEN_NORMALIZATION_BOUNDS,
    ref_point: tuple[float, ...] = DEFAULT_REF_POINT,
) -> dict:
    policy_rows = aggregate_policy_rows(run_rows)

    normalized_run_rows = normalize_objective_rows(run_rows, bounds=bounds)
    normalized_policy_rows = normalize_objective_rows(policy_rows, bounds=bounds)

    raw_policy_objectives = [[row[key] for key in OBJECTIVE_KEYS] for row in policy_rows]
    pareto_mask = compute_pareto_front_mask(raw_policy_objectives)

    pareto_rows = [row for row, is_pareto in zip(normalized_policy_rows, pareto_mask) if is_pareto]
    pareto_points = [row["normalized_objectives"] for row in pareto_rows]
    hypervolume = compute_hypervolume(points=pareto_points, ref_point=ref_point)

    return {
        "objective_keys": list(OBJECTIVE_KEYS),
        "normalization_bounds": {key: list(bounds[key]) for key in OBJECTIVE_KEYS},
        "ref_point": list(ref_point),
        "generated_from_runs": len(run_rows),
        "generated_from_policies": len(policy_rows),
        "policy_weights": POLICY_WEIGHT_MAP,
        "run_level": {
            "rows": normalized_run_rows,
        },
        "policy_level": {
            "rows": normalized_policy_rows,
        },
        "pareto_front": {
            "policy_names": [row["policy"] for row in pareto_rows],
            "rows": pareto_rows,
        },
        "hypervolume": float(hypervolume),
    }


def save_pairwise_figure(policy_rows: list[dict], pareto_policy_names: set[str], output_path: Path) -> None:
    metric_pairs = list(combinations(OBJECTIVE_KEYS, 2))
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for axis, (x_key, y_key) in zip(axes, metric_pairs):
        for row in policy_rows:
            is_pareto = row["policy"] in pareto_policy_names
            axis.scatter(
                row[x_key],
                row[y_key],
                color="#c0392b" if is_pareto else "#1f77b4",
                s=80 if is_pareto else 50,
                alpha=0.9,
            )
            axis.annotate(row["policy"], (row[x_key], row[y_key]), textcoords="offset points", xytext=(4, 4), fontsize=8)
        axis.set_xlabel(x_key)
        axis.set_ylabel(y_key)
        axis.grid(True, alpha=0.25)

    fig.suptitle("Phase MORL Pareto Pairwise Scatter (lower is better)", fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_policy_summary_figure(policy_rows: list[dict], pareto_policy_names: set[str], output_path: Path) -> None:
    normalized_rows = normalize_objective_rows(policy_rows, bounds=FROZEN_NORMALIZATION_BOUNDS)

    policies = [row["policy"] for row in normalized_rows]
    positions = np.arange(len(policies))
    width = 0.18

    fig, axis = plt.subplots(figsize=(15, 6))
    for idx, key in enumerate(OBJECTIVE_KEYS):
        offsets = positions + (idx - 1.5) * width
        values = [row["normalized_metrics"][key] for row in normalized_rows]
        axis.bar(offsets, values, width=width, label=key)

    for index, policy_name in enumerate(policies):
        if policy_name in pareto_policy_names:
            axis.text(positions[index], 1.04, "Pareto", ha="center", va="bottom", fontsize=8, color="#c0392b")

    axis.set_xticks(positions)
    axis.set_xticklabels(policies)
    axis.set_ylim(0.0, 1.1)
    axis.set_ylabel("Normalized objective value")
    axis.set_title("Phase MORL Policy-Level Objectives (lower is better)")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(ncols=4)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    run_rows = load_run_rows(summary_dir=args.summary_dir, run_root=args.run_root)
    payload = build_analysis_payload(run_rows)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    policy_rows = [dict(row) for row in payload["policy_level"]["rows"]]
    pareto_policy_names = set(payload["pareto_front"]["policy_names"])
    save_pairwise_figure(policy_rows, pareto_policy_names, args.figure_dir / "pareto_front_pairwise.png")
    save_policy_summary_figure(policy_rows, pareto_policy_names, args.figure_dir / "pareto_front_policy_summary.png")

    print(f"[PARETO] JSON written to: {args.output_json}")
    print(f"[PARETO] Policy count: {payload['generated_from_policies']}")
    print(f"[PARETO] Pareto policies: {', '.join(payload['pareto_front']['policy_names'])}")
    print(f"[PARETO] Hypervolume: {payload['hypervolume']:.6f}")


if __name__ == "__main__":
    main()
