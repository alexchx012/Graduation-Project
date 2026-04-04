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


def _build_ablation_experiment(ablation_id: str, project_root: Path) -> dict:
    spec = dict(ABLATION_SPECS[ablation_id])
    spec["task"] = MORL_V2_TASK
    spec["command_profile"] = DEFAULT_COMMAND_PROFILE
    spec["morl_curriculum_warmup"] = DEFAULT_CURRICULUM_WARMUP
    spec["morl_curriculum_ramp"] = DEFAULT_CURRICULUM_RAMP
    spec["init_with_optimizer"] = True
    checkpoint_path = (project_root / BASELINE_INIT_CHECKPOINT).resolve()
    if checkpoint_path.exists():
        spec["init_checkpoint"] = str(checkpoint_path)
    else:
        print(f"[WARNING] Baseline checkpoint not found: {checkpoint_path}")
    return spec


def main():
    parser = argparse.ArgumentParser(
        description="Phase 4 C-layer: MORL ablation experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--ablation-ids",
        type=str,
        default=None,
        help="Comma-separated ablation ids (default: no-energy,no-smooth).",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,43,44",
        help="Comma-separated seeds (default: 42,43,44).",
    )
    parser.add_argument("--num-envs", type=int, default=NUM_ENVS)
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    parser.add_argument("--clip-param", type=float, default=DEFAULT_CLIP_PARAM)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--project-root", type=Path, default=None)
    args = parser.parse_args()

    project_root = args.project_root or Path(__file__).resolve().parents[2]

    # Select ablations
    if args.ablation_ids:
        ablation_ids = [s.strip().lower() for s in args.ablation_ids.split(",") if s.strip()]
        unknown = [a for a in ablation_ids if a not in ABLATION_SPECS]
        if unknown:
            print(f"[ERROR] Unknown ablation ids: {unknown}. Known: {sorted(ABLATION_SPECS)}")
            sys.exit(1)
    else:
        ablation_ids = list(DEFAULT_ABLATION_IDS)

    seeds = _parse_seeds(args.seeds)
    experiments = [_build_ablation_experiment(aid, project_root) for aid in ablation_ids]
    total_runs = len(experiments) * len(seeds)

    log_dir = project_root / "logs" / "sweep" / "phase_morl_ablation"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"ablation_{timestamp}.log"
    log = DualLogger(log_path)
    log.info(f"Project root: {project_root}")
    log.info(f"Ablations: {ablation_ids}")
    log.info(f"Seeds: {seeds}, Total runs: {total_runs}")

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
                    num_envs=args.num_envs,
                    max_iterations=args.max_iterations,
                    clip_param=args.clip_param,
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
