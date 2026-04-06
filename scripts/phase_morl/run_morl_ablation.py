#!/usr/bin/env python3
"""
Phase 4 C-layer: MORL ablation experiments.

Trains two ablation variants of P10 (stability-balanced):
- P10-no-energy: Remove energy objective  (speed=0.25, energy=0.0, smooth=0.25, stable=0.5)
- P10-no-smooth: Remove smoothness objective (speed=0.25, energy=0.25, smooth=0.0, stable=0.5)

These are compared against P10-full from the A-layer confirm sweep to
isolate the contribution of each MORL objective.

All experiments use v2 architecture with curriculum (same as confirm sweep).

Usage:
    conda activate env_isaaclab

    # Run both ablations x 3 seeds (6 runs)
    python scripts/phase_morl/run_morl_ablation.py

    # Run specific ablation
    python scripts/phase_morl/run_morl_ablation.py --ablation-ids no-energy

    # Dry run
    python scripts/phase_morl/run_morl_ablation.py --dry-run

    # Resume
    python scripts/phase_morl/run_morl_ablation.py --resume
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_morl_train_sweep import (  # noqa: E402
    DEFAULT_CLIP_PARAM,
    DualLogger,
    NUM_ENVS,
    RunResult,
    SessionState,
    _parse_seeds,
    print_summary,
    run_single_training,
)

# ── v2 training defaults (same as confirm sweep v2) ─────────────────────
MORL_V2_TASK = "Isaac-Velocity-MORL-Unitree-Go1-ROS2Cmd-v2"
DEFAULT_MAX_ITERATIONS = 900
DEFAULT_CURRICULUM_WARMUP = 300
DEFAULT_CURRICULUM_RAMP = 300
DEFAULT_COMMAND_PROFILE = "repair_forward_v2"
DEFAULT_ABLATION_MANIFEST = SCRIPT_DIR / "manifests" / "phase4_ablation_manifest.json"
BASELINE_INIT_CHECKPOINT = (
    Path("logs")
    / "rsl_rl"
    / "unitree_go1_rough"
    / "2026-03-08_16-46-27_baseline_rough_ros2cmd"
    / "model_1499.pt"
)

# ── Ablation experiment definitions ──────────────────────────────────────
# P10 original: (0.2, 0.2, 0.2, 0.4)
# Ablation: zero out one objective, redistribute to others proportionally
ABLATION_SPECS = {
    "no-energy": {
        "name": "morl_p10_ablation_no_energy",
        "policy_id": "P10-no-energy",
        "morl_weights": "0.25,0.0,0.25,0.5",
        "note": "P10 ablation: energy objective removed",
    },
    "no-smooth": {
        "name": "morl_p10_ablation_no_smooth",
        "policy_id": "P10-no-smooth",
        "morl_weights": "0.25,0.25,0.0,0.5",
        "note": "P10 ablation: smoothness objective removed",
    },
}

DEFAULT_ABLATION_IDS = ("no-energy", "no-smooth")
DEFAULT_SEEDS = [42, 43, 44]


def _weights_to_cli_string(weights: list[float] | tuple[float, ...] | str) -> str:
    if isinstance(weights, str):
        return weights
    return ",".join(str(float(weight)) for weight in weights)


def _normalize_entry_id(entry_id: str) -> str:
    alias_map = {
        "no-energy": "anchor-no-energy",
        "no-smooth": "anchor-no-smooth",
        "full": "anchor-full",
    }
    normalized = entry_id.strip().lower()
    return alias_map.get(normalized, normalized)


def load_phase4_ablation_manifest(path: Path | str = DEFAULT_ABLATION_MANIFEST, *, project_root: Path) -> dict:
    manifest_path = Path(path)
    if not manifest_path.is_absolute():
        manifest_path = (project_root / manifest_path).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Phase 4 ablation manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def select_ablation_experiments(
    manifest: dict,
    *,
    project_root: Path,
    entry_ids: set[str] | None,
    include_anchor_full: bool,
) -> tuple[list[dict], dict]:
    protocol = dict(manifest.get("training_protocol", {}))
    entries_raw = manifest.get("entries", [])
    if not isinstance(entries_raw, list):
        raise ValueError("Invalid ablation manifest: missing 'entries' list")

    normalized_entry_ids = {_normalize_entry_id(entry_id) for entry_id in entry_ids} if entry_ids else None
    known_ids = {str(entry["ablation_id"]) for entry in entries_raw}
    if normalized_entry_ids:
        unknown = sorted(normalized_entry_ids - known_ids)
        if unknown:
            raise ValueError(f"Unknown ablation entry ids: {unknown}. Known: {sorted(known_ids)}")

    experiments: list[dict] = []
    for raw in entries_raw:
        entry_id = str(raw["ablation_id"])
        role = str(raw.get("role", "ablation_variant"))
        if normalized_entry_ids is not None:
            if entry_id not in normalized_entry_ids:
                continue
        else:
            if role != "ablation_variant" and not (include_anchor_full and role == "anchor_full"):
                continue

        spec = {
            "ablation_id": entry_id,
            "name": str(raw["name"]),
            "policy_id": str(raw["policy_id"]),
            "role": role,
            "morl_weights": _weights_to_cli_string(raw["morl_weights"]),
            "note": str(raw.get("note", "")),
            "task": str(protocol.get("task", MORL_V2_TASK)),
            "command_profile": str(protocol.get("command_profile", DEFAULT_COMMAND_PROFILE)),
            "morl_curriculum_warmup": int(protocol.get("curriculum_warmup", DEFAULT_CURRICULUM_WARMUP)),
            "morl_curriculum_ramp": int(protocol.get("curriculum_ramp", DEFAULT_CURRICULUM_RAMP)),
            "init_with_optimizer": bool(protocol.get("init_with_optimizer", True)),
        }

        init_checkpoint = protocol.get("init_checkpoint")
        if init_checkpoint:
            checkpoint_path = Path(str(init_checkpoint))
            if not checkpoint_path.is_absolute():
                checkpoint_path = (project_root / checkpoint_path).resolve()
            spec["init_checkpoint"] = str(checkpoint_path)

        experiments.append(spec)

    protocol["training_seeds"] = [int(seed) for seed in protocol.get("training_seeds", DEFAULT_SEEDS)]
    protocol["num_envs"] = int(protocol.get("num_envs", NUM_ENVS))
    protocol["max_iterations"] = int(protocol.get("max_iterations", DEFAULT_MAX_ITERATIONS))
    protocol["clip_param"] = float(protocol.get("clip_param", DEFAULT_CLIP_PARAM))
    return experiments, protocol


def main():
    parser = argparse.ArgumentParser(
        description="Phase 4 C-layer: MORL ablation experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_ABLATION_MANIFEST,
        help="Path to phase4_ablation_manifest.json",
    )
    parser.add_argument(
        "--ablation-ids",
        type=str,
        default=None,
        help="Comma-separated manifest ablation ids (legacy aliases no-energy/no-smooth are supported).",
    )
    parser.add_argument(
        "--include-anchor-full",
        action="store_true",
        help="Also include the anchor-full entry from the ablation manifest.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Optional comma-separated seeds override (default: use ablation manifest).",
    )
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--clip-param", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--project-root", type=Path, default=None)
    args = parser.parse_args()

    project_root = args.project_root or Path(__file__).resolve().parents[2]

    try:
        manifest = load_phase4_ablation_manifest(args.manifest, project_root=project_root)
        selected_ids = {item.strip() for item in args.ablation_ids.split(",") if item.strip()} if args.ablation_ids else None
        experiments, protocol = select_ablation_experiments(
            manifest,
            project_root=project_root,
            entry_ids=selected_ids,
            include_anchor_full=args.include_anchor_full,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    if not experiments:
        print("[ERROR] No ablation entries selected from manifest.")
        sys.exit(1)

    seeds = _parse_seeds(args.seeds) if args.seeds else list(protocol["training_seeds"])
    num_envs = args.num_envs if args.num_envs is not None else protocol["num_envs"]
    max_iterations = args.max_iterations if args.max_iterations is not None else protocol["max_iterations"]
    clip_param = args.clip_param if args.clip_param is not None else protocol["clip_param"]
    total_runs = len(experiments) * len(seeds)

    log_dir = project_root / "logs" / "sweep" / "phase_morl_ablation"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"ablation_{timestamp}.log"
    log = DualLogger(log_path)
    log.info(f"Project root: {project_root}")
    log.info(f"Manifest: {Path(args.manifest)}")
    log.info(f"Selected entries: {[exp['ablation_id'] for exp in experiments]}")
    log.info(f"Seeds: {seeds}, Total runs: {total_runs}")
    log.info(f"num_envs={num_envs}, max_iterations={max_iterations}, clip_param={clip_param}")

    session_path = log_dir / "session_ablation.json"
    session = SessionState.load(session_path) if args.resume else SessionState(
        start_time=datetime.now().isoformat()
    )

    results: list[RunResult] = []
    interrupted = False

    try:
        run_idx = 0
        for exp in experiments:
            for seed in seeds:
                run_idx += 1
                run_name = f"{exp['name']}_seed{seed}"
                if args.resume and session.is_done(run_name):
                    log.info(f"\n[SKIP] {run_name} ({run_idx}/{total_runs})")
                    continue

                log.info(f"\n[PROGRESS] {run_idx}/{total_runs} {exp['policy_id']} seed={seed}")
                log.info(f"  weights: {exp['morl_weights']}")
                result = run_single_training(
                    exp,
                    seed,
                    project_root,
                    log,
                    ros2_mgr=None,
                    dry_run=args.dry_run,
                    num_envs=num_envs,
                    max_iterations=max_iterations,
                    clip_param=clip_param,
                    sweep_log_subdir="phase_morl_ablation",
                )
                results.append(result)

                if result.passed:
                    session.completed.append(run_name)
                else:
                    session.failed.append(run_name)
                session.save(session_path)
    except KeyboardInterrupt:
        log.warn("\nUser interrupted (Ctrl+C)")
        interrupted = True

    if results:
        print_summary(results, log, summary_filename="phase_morl_ablation_summary.json")

    log.close()
    print(f"\n[INFO] Full log: {log_path}")

    if interrupted:
        sys.exit(130)
    all_passed = all(r.passed for r in results) if results else False
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
