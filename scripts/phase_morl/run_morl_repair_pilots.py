#!/usr/bin/env python3
"""
Phase MORL repair pilots.

Pilot A: P1, command-profile fix only
Pilot B: P10, command-profile fix + baseline warm-start
Pilot C: P2, command-profile fix + baseline warm-start
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
    RunResult,
    SessionState,
    _build_train_cmd,
    _parse_seeds,
    print_summary,
    run_single_training,
    NUM_ENVS,
)


DEFAULT_MAX_ITERATIONS = 600
DEFAULT_PILOT_IDS = ("A", "B", "C")
DEFAULT_COMMAND_PROFILE = "repair_forward_v1"
BASELINE_INIT_CHECKPOINT = (
    Path("logs")
    / "rsl_rl"
    / "unitree_go1_rough"
    / "2026-03-08_16-46-27_baseline_rough_ros2cmd"
    / "model_1499.pt"
)

PILOT_SPECS = {
    "A": {
        "pilot_id": "A",
        "name": "pilot_a_p1_cmdfix",
        "policy_id": "P1",
        "morl_weights": "0.7,0.1,0.1,0.1",
        "command_profile": DEFAULT_COMMAND_PROFILE,
        "use_warm_start": False,
    },
    "B": {
        "pilot_id": "B",
        "name": "pilot_b_p10_cmdfix_warm",
        "policy_id": "P10",
        "morl_weights": "0.2,0.2,0.2,0.4",
        "command_profile": DEFAULT_COMMAND_PROFILE,
        "use_warm_start": True,
    },
    "C": {
        "pilot_id": "C",
        "name": "pilot_c_p2_cmdfix_warm",
        "policy_id": "P2",
        "morl_weights": "0.1,0.7,0.1,0.1",
        "command_profile": DEFAULT_COMMAND_PROFILE,
        "use_warm_start": True,
    },
}


def _is_project_root(project_root: Path) -> bool:
    return (project_root / "AGENTS.md").exists() or (project_root / "CLAUDE.md").exists()


def _select_pilots(pilot_ids_raw: str | None) -> list[str]:
    if not pilot_ids_raw:
        return list(DEFAULT_PILOT_IDS)

    wanted = [item.strip().upper() for item in pilot_ids_raw.split(",") if item.strip()]
    unknown = [item for item in wanted if item not in PILOT_SPECS]
    if unknown:
        raise ValueError(f"Unknown pilot ids: {unknown}. Known: {sorted(PILOT_SPECS)}")
    return wanted


def _build_repair_experiment(pilot_id: str, project_root: Path | None = None) -> dict:
    spec = dict(PILOT_SPECS[pilot_id])
    exp = {
        "name": spec["name"],
        "policy_id": spec["policy_id"],
        "morl_weights": spec["morl_weights"],
        "command_profile": spec["command_profile"],
    }
    if spec["use_warm_start"]:
        root = project_root or Path(__file__).resolve().parents[2]
        exp["init_checkpoint"] = str((root / BASELINE_INIT_CHECKPOINT).resolve())
    return exp


def _build_repair_train_cmd(
    project_root: Path,
    exp: dict,
    *,
    seed: int,
    num_envs: int = NUM_ENVS,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    clip_param: float = DEFAULT_CLIP_PARAM,
) -> list[str]:
    run_name = f"{exp['name']}_seed{seed}"
    return _build_train_cmd(
        project_root,
        exp,
        seed,
        run_name,
        num_envs=num_envs,
        max_iterations=max_iterations,
        clip_param=clip_param,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Phase MORL repair pilots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--pilot-ids",
        type=str,
        default=None,
        help="Comma-separated pilot ids, default A,B,C.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42",
        help="Comma-separated seeds, default 42.",
    )
    parser.add_argument("--num-envs", type=int, default=NUM_ENVS, help="Training env count.")
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help="Training iterations for repair pilots.",
    )
    parser.add_argument("--clip-param", type=float, default=DEFAULT_CLIP_PARAM, help="PPO clip_param.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only.")
    parser.add_argument("--resume", action="store_true", help="Skip pilots already recorded in the session file.")
    parser.add_argument("--project-root", type=Path, default=None, help="Project root.")
    args = parser.parse_args()

    project_root = args.project_root or Path(__file__).resolve().parents[2]
    if not _is_project_root(project_root):
        print(f"[ERROR] Invalid project root: {project_root}")
        sys.exit(1)

    try:
        pilot_ids = _select_pilots(args.pilot_ids)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    seeds = _parse_seeds(args.seeds)
    experiments = [_build_repair_experiment(pilot_id, project_root) for pilot_id in pilot_ids]
    total_runs = len(experiments) * len(seeds)

    log_dir = project_root / "logs" / "sweep" / "phase_morl_repair"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pilot_tag = "all" if args.pilot_ids is None else args.pilot_ids.replace(",", "_").replace(" ", "")
    log_path = log_dir / f"phase_morl_repair_{pilot_tag}_{timestamp}.log"
    log = DualLogger(log_path)
    log.info(f"Project root: {project_root}")
    log.info(f"Pilots: {pilot_ids}")
    log.info(f"Seeds: {seeds}")
    log.info(f"num_envs={args.num_envs}, max_iterations={args.max_iterations}, clip_param={args.clip_param}")

    session_path = log_dir / f"session_{pilot_tag}.json"
    session = SessionState.load(session_path) if args.resume else SessionState(
        start_time=datetime.now().isoformat()
    )

    results: list[RunResult] = []
    interrupted = False

    try:
        run_idx = 0
        for pilot_id, exp in zip(pilot_ids, experiments, strict=True):
            for seed in seeds:
                run_idx += 1
                run_name = f"{exp['name']}_seed{seed}"
                if args.resume and session.is_done(run_name):
                    log.info(f"\n[SKIP] {run_name} already completed ({run_idx}/{total_runs})")
                    continue

                log.info(f"\n[PROGRESS] {run_idx}/{total_runs} pilot={pilot_id}")
                log.info(f"Pilot spec: {exp}")
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
                    sweep_log_subdir="phase_morl_repair",
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
        print_summary(results, log, summary_filename="phase_morl_repair_summary.json")

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
