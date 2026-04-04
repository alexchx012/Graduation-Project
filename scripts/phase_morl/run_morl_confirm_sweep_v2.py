#!/usr/bin/env python3
"""
Phase MORL: v2 confirm sweep with curriculum.

Retrains all 10 MORL weight policies using the v2 architecture:
- Fixed locomotion scaffold + secondary MORL objectives
- Curriculum: warmup (pure scaffold) -> ramp (linear 0->1) -> full MORL
- Warm start from baseline rough checkpoint
- clip_param = 0.2

Evidence layers:
  A layer (confirmation): P1, P2, P3, P4, P10 x seeds 42,43,44  (15 runs)
  B layer (exploration):  P5, P6, P7, P8, P9  x seed 42          (5 runs)

Usage:
    conda activate env_isaaclab

    # Batch 1: A layer x seed42 only (5 runs, quick validation)
    python scripts/phase_morl/run_morl_confirm_sweep_v2.py --layer A --seeds 42

    # Batch 2: remaining A layer + B layer
    python scripts/phase_morl/run_morl_confirm_sweep_v2.py --layer A --seeds 43,44
    python scripts/phase_morl/run_morl_confirm_sweep_v2.py --layer B

    # Full sweep (all 20 runs)
    python scripts/phase_morl/run_morl_confirm_sweep_v2.py --layer AB

    # Specific policies
    python scripts/phase_morl/run_morl_confirm_sweep_v2.py --policy-ids P1,P10 --seeds 42

    # Dry run / Resume
    python scripts/phase_morl/run_morl_confirm_sweep_v2.py --dry-run
    python scripts/phase_morl/run_morl_confirm_sweep_v2.py --resume
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
    MORL_EXPERIMENTS,
    NUM_ENVS,
    RunResult,
    SessionState,
    _parse_seeds,
    print_summary,
    run_single_training,
)

# ── v2 defaults ──────────────────────────────────────────────────────────
MORL_V2_TASK = "Isaac-Velocity-MORL-Unitree-Go1-ROS2Cmd-v2"
DEFAULT_MAX_ITERATIONS = 900  # 300 warmup + 300 ramp + 300 full
DEFAULT_CURRICULUM_WARMUP = 300  # iters
DEFAULT_CURRICULUM_RAMP = 300  # iters
BASELINE_INIT_CHECKPOINT = (
    Path("logs")
    / "rsl_rl"
    / "unitree_go1_rough"
    / "2026-03-08_16-46-27_baseline_rough_ros2cmd"
    / "model_1499.pt"
)
DEFAULT_COMMAND_PROFILE = "repair_forward_v2"

# Evidence layers
A_LAYER_POLICY_IDS = ("P1", "P2", "P3", "P4", "P10")
A_LAYER_SEEDS = [42, 43, 44]
B_LAYER_POLICY_IDS = ("P5", "P6", "P7", "P8", "P9")
B_LAYER_SEEDS = [42]


def _build_v2_experiment(base_exp: dict, project_root: Path) -> dict:
    """Augment a base MORL experiment dict with v2-specific fields."""
    exp = dict(base_exp)
    exp["task"] = MORL_V2_TASK
    exp["command_profile"] = DEFAULT_COMMAND_PROFILE
    exp["morl_curriculum_warmup"] = DEFAULT_CURRICULUM_WARMUP
    exp["morl_curriculum_ramp"] = DEFAULT_CURRICULUM_RAMP
    exp["init_with_optimizer"] = True
    checkpoint_path = (project_root / BASELINE_INIT_CHECKPOINT).resolve()
    if checkpoint_path.exists():
        exp["init_checkpoint"] = str(checkpoint_path)
    else:
        print(f"[WARNING] Baseline checkpoint not found: {checkpoint_path}")
        print("[WARNING] Training will start from scratch (no warm start).")
    return exp


def _select_experiments_and_seeds(
    layer: str,
    policy_ids_raw: str | None,
    seeds_raw: str | None,
) -> tuple[list[dict], list[int]]:
    """Return (experiments, seeds) based on layer/policy/seed selection."""
    layer = layer.upper()

    # Build policy id set
    if policy_ids_raw:
        wanted_ids = {item.strip().upper() for item in policy_ids_raw.split(",") if item.strip()}
    elif layer == "A":
        wanted_ids = set(A_LAYER_POLICY_IDS)
    elif layer == "B":
        wanted_ids = set(B_LAYER_POLICY_IDS)
    elif layer == "AB":
        wanted_ids = set(A_LAYER_POLICY_IDS) | set(B_LAYER_POLICY_IDS)
    else:
        raise ValueError(f"Unknown layer '{layer}'. Use A, B, or AB.")

    known_ids = {exp["policy_id"] for exp in MORL_EXPERIMENTS}
    unknown = sorted(wanted_ids - known_ids)
    if unknown:
        raise ValueError(f"Unknown policy ids: {unknown}. Known: {sorted(known_ids)}")

    experiments = [exp for exp in MORL_EXPERIMENTS if exp["policy_id"] in wanted_ids]

    # Build seed list
    if seeds_raw:
        seeds = _parse_seeds(seeds_raw)
    elif layer == "B":
        seeds = list(B_LAYER_SEEDS)
    else:
        seeds = list(A_LAYER_SEEDS)

    return experiments, seeds


def main():
    parser = argparse.ArgumentParser(
        description="Phase MORL v2 confirm sweep with curriculum",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--layer",
        type=str,
        default="A",
        help="Evidence layer: A (5 policies x 3 seeds), B (5 policies x seed42), AB (all 20 runs). Default: A.",
    )
    parser.add_argument(
        "--policy-ids",
        type=str,
        default=None,
        help="Override: specific policy ids, e.g. P1,P10.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Override: specific seeds, e.g. 42 or 42,43,44.",
    )
    parser.add_argument("--num-envs", type=int, default=NUM_ENVS, help="Training env count.")
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help="Training iterations (default 900 = 300 warmup + 300 ramp + 300 full).",
    )
    parser.add_argument("--clip-param", type=float, default=DEFAULT_CLIP_PARAM, help="PPO clip_param.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only.")
    parser.add_argument("--resume", action="store_true", help="Skip completed runs.")
    parser.add_argument("--project-root", type=Path, default=None, help="Project root.")
    args = parser.parse_args()

    project_root = args.project_root or Path(__file__).resolve().parents[2]
    if not (project_root / "AGENTS.md").exists() and not (project_root / "CLAUDE.md").exists():
        print(f"[ERROR] Invalid project root: {project_root}")
        sys.exit(1)

    try:
        base_experiments, seeds = _select_experiments_and_seeds(
            args.layer, args.policy_ids, args.seeds
        )
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    experiments = [_build_v2_experiment(exp, project_root) for exp in base_experiments]
    total_runs = len(experiments) * len(seeds)

    log_dir = project_root / "logs" / "sweep" / "phase_morl_confirm_v2"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    layer_tag = args.layer.upper() if args.policy_ids is None else args.policy_ids.replace(",", "_")
    log_path = log_dir / f"confirm_v2_{layer_tag}_{timestamp}.log"
    log = DualLogger(log_path)
    log.info(f"Project root: {project_root}")
    log.info(f"Layer: {args.layer}, Policies: {[e['policy_id'] for e in experiments]}")
    log.info(f"Seeds: {seeds}, Total runs: {total_runs}")
    log.info(f"Task: {MORL_V2_TASK}")
    log.info(f"Curriculum: warmup={DEFAULT_CURRICULUM_WARMUP}, ramp={DEFAULT_CURRICULUM_RAMP}")
    log.info(f"num_envs={args.num_envs}, max_iterations={args.max_iterations}, clip_param={args.clip_param}")

    session_path = log_dir / f"session_{layer_tag}.json"
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
                    log.info(f"\n[SKIP] {run_name} already completed ({run_idx}/{total_runs})")
                    continue

                log.info(f"\n[PROGRESS] {run_idx}/{total_runs} policy={exp['policy_id']} seed={seed}")
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
                    sweep_log_subdir="phase_morl_confirm_v2",
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
        print_summary(results, log, summary_filename="phase_morl_confirm_v2_summary.json")

    log.close()
    print(f"\n[INFO] Full log: {log_path}")
    if args.resume:
        print(f"[INFO] Session state: {session_path}")

    if interrupted:
        sys.exit(130)
    all_passed = all(result.passed for result in results) if results else False
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
