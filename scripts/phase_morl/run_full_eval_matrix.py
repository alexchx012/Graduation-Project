#!/usr/bin/env python3
"""
Phase 4: Full evaluation matrix.

Evaluates all (policy x scenario x seed) combinations by invoking
run_morl_eval.py as subprocesses.

Usage:
    conda activate env_isaaclab

    # Evaluate A-layer policies on S1 (quick validation)
    python scripts/phase_morl/run_full_eval_matrix.py --scenarios S1

    # Full matrix: all policies x all scenarios
    python scripts/phase_morl/run_full_eval_matrix.py --scenarios S1,S2,S3,S4,S5,S6

    # Specific policies
    python scripts/phase_morl/run_full_eval_matrix.py --policy-ids P1,P10 --scenarios S1,S2

    # Custom run root and output
    python scripts/phase_morl/run_full_eval_matrix.py --run-root logs/rsl_rl/unitree_go1_rough

    # Resume interrupted matrix
    python scripts/phase_morl/run_full_eval_matrix.py --resume

    # Dry run
    python scripts/phase_morl/run_full_eval_matrix.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

# Default evaluation parameters
DEFAULT_TASK = "Isaac-Velocity-MORL-Unitree-Go1-ROS2Cmd-Play-v2"
DEFAULT_SCENARIOS = ["S1"]
DEFAULT_EVAL_STEPS = 3000
DEFAULT_WARMUP_STEPS = 300
DEFAULT_NUM_ENVS = 64
DEFAULT_CHECKPOINT = "model_899.pt"

# Run name pattern for discovering trained policies
_RUN_NAME_PATTERN = re.compile(r"morl_p(\d+)_seed(\d+)$")


@dataclass
class EvalResult:
    policy_id: str
    seed: int
    scenario: str
    run_name: str
    passed: bool
    return_code: int = -1
    output_json: str | None = None
    duration_s: float = 0.0
    error: str | None = None


@dataclass
class EvalSession:
    completed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    start_time: str = ""

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "EvalSession":
        if not path.exists():
            return cls(start_time=datetime.now().isoformat())
        with open(path, "r", encoding="utf-8") as f:
            return cls(**json.load(f))

    def is_done(self, key: str) -> bool:
        return key in self.completed


def discover_trained_runs(
    run_root: Path,
    policy_ids: set[str] | None = None,
    seeds: set[int] | None = None,
) -> list[tuple[str, str, int]]:
    """Discover trained policy directories matching morl_pX_seedY pattern.

    Returns list of (run_dir_name, policy_id, seed).
    """
    if not run_root.exists():
        return []

    runs = []
    for path in sorted(run_root.iterdir()):
        if not path.is_dir():
            continue
        match = _RUN_NAME_PATTERN.search(path.name)
        if not match:
            continue
        policy_id = f"P{match.group(1)}"
        seed = int(match.group(2))
        if policy_ids and policy_id not in policy_ids:
            continue
        if seeds and seed not in seeds:
            continue
        runs.append((path.name, policy_id, seed))
    return runs


def build_eval_cmd(
    run_dir_name: str,
    scenario: str,
    output_json: str,
    *,
    task: str = DEFAULT_TASK,
    num_envs: int = DEFAULT_NUM_ENVS,
    eval_steps: int = DEFAULT_EVAL_STEPS,
    warmup_steps: int = DEFAULT_WARMUP_STEPS,
    checkpoint: str = DEFAULT_CHECKPOINT,
) -> list[str]:
    """Build the run_morl_eval.py command."""
    eval_script = str(SCRIPT_DIR / "run_morl_eval.py")
    cmd = [
        sys.executable,
        eval_script,
        "--task", task,
        "--load_run", run_dir_name,
        "--checkpoint", checkpoint,
        "--scenario", scenario,
        "--skip_ros2",
        "--num_envs", str(num_envs),
        "--eval_steps", str(eval_steps),
        "--warmup_steps", str(warmup_steps),
        "--summary_json", output_json,
        "--headless",
    ]
    return cmd


def main():
    parser = argparse.ArgumentParser(
        description="Phase 4: Full evaluation matrix",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--policy-ids",
        type=str,
        default=None,
        help="Comma-separated policy ids to evaluate (default: all discovered).",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds to evaluate (default: all discovered).",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default=",".join(DEFAULT_SCENARIOS),
        help="Comma-separated scenario ids (default: S1).",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=DEFAULT_TASK,
        help=f"Task id for evaluation (default: {DEFAULT_TASK}).",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=PROJECT_ROOT / "logs" / "rsl_rl" / "unitree_go1_rough",
        help="Directory containing trained policy run folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "logs" / "eval" / "phase_morl_v2",
        help="Directory for output eval JSONs.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help=f"Checkpoint filename (default: {DEFAULT_CHECKPOINT}).",
    )
    parser.add_argument("--num-envs", type=int, default=DEFAULT_NUM_ENVS)
    parser.add_argument("--eval-steps", type=int, default=DEFAULT_EVAL_STEPS)
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument("--dry-run", action="store_true", help="Print commands only.")
    parser.add_argument("--resume", action="store_true", help="Skip completed evals.")
    args = parser.parse_args()

    scenarios = [s.strip().upper() for s in args.scenarios.split(",") if s.strip()]
    policy_ids = (
        {s.strip().upper() for s in args.policy_ids.split(",") if s.strip()}
        if args.policy_ids
        else None
    )
    seeds = (
        {int(s.strip()) for s in args.seeds.split(",") if s.strip()}
        if args.seeds
        else None
    )

    # Discover trained runs
    runs = discover_trained_runs(args.run_root, policy_ids=policy_ids, seeds=seeds)
    if not runs:
        print(f"[ERROR] No trained runs found in {args.run_root}")
        print("[HINT] Make sure the confirm sweep v2 training has completed.")
        sys.exit(1)

    total_evals = len(runs) * len(scenarios)
    print(f"[INFO] Discovered {len(runs)} trained runs")
    print(f"[INFO] Scenarios: {scenarios}")
    print(f"[INFO] Total evaluations: {total_evals}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Session management
    session_path = args.output_dir / "eval_matrix_session.json"
    session = EvalSession.load(session_path) if args.resume else EvalSession(
        start_time=datetime.now().isoformat()
    )

    results: list[EvalResult] = []
    interrupted = False
    eval_idx = 0

    try:
        for run_dir_name, policy_id, seed in runs:
            for scenario in scenarios:
                eval_idx += 1
                eval_key = f"{run_dir_name}_{scenario}"
                output_json = str(
                    args.output_dir / f"morl_{policy_id.lower()}_seed{seed}_{scenario}.json"
                )

                if args.resume and session.is_done(eval_key):
                    print(f"[SKIP] {eval_key} ({eval_idx}/{total_evals})")
                    continue

                print(f"\n[EVAL {eval_idx}/{total_evals}] {policy_id} seed={seed} {scenario}")

                cmd = build_eval_cmd(
                    run_dir_name,
                    scenario,
                    output_json,
                    task=args.task,
                    num_envs=args.num_envs,
                    eval_steps=args.eval_steps,
                    warmup_steps=args.warmup_steps,
                    checkpoint=args.checkpoint,
                )

                if args.dry_run:
                    print(f"  CMD: {' '.join(cmd)}")
                    results.append(EvalResult(
                        policy_id=policy_id, seed=seed, scenario=scenario,
                        run_name=run_dir_name, passed=True, error="dry-run",
                    ))
                    continue

                start_t = time.time()
                stderr_path = args.output_dir / f"morl_{policy_id.lower()}_seed{seed}_{scenario}_stderr.log"
                try:
                    with open(stderr_path, "w", encoding="utf-8") as f_err:
                        proc = subprocess.run(
                            cmd,
                            cwd=str(PROJECT_ROOT),
                            stdout=subprocess.PIPE,
                            stderr=f_err,
                            timeout=600,
                        )
                    duration = time.time() - start_t
                    passed = proc.returncode == 0 and Path(output_json).exists()

                    result = EvalResult(
                        policy_id=policy_id, seed=seed, scenario=scenario,
                        run_name=run_dir_name, passed=passed,
                        return_code=proc.returncode,
                        output_json=output_json if passed else None,
                        duration_s=duration,
                    )

                    if passed:
                        session.completed.append(eval_key)
                        # Clean up stderr log on success
                        stderr_path.unlink(missing_ok=True)
                        print(f"  [PASS] {duration:.0f}s -> {output_json}")
                    else:
                        session.failed.append(eval_key)
                        print(f"  [FAIL] rc={proc.returncode}, stderr: {stderr_path}")

                except subprocess.TimeoutExpired:
                    duration = time.time() - start_t
                    result = EvalResult(
                        policy_id=policy_id, seed=seed, scenario=scenario,
                        run_name=run_dir_name, passed=False,
                        duration_s=duration, error="timeout",
                    )
                    session.failed.append(eval_key)
                    print(f"  [TIMEOUT] after {duration:.0f}s")
                except Exception as exc:
                    result = EvalResult(
                        policy_id=policy_id, seed=seed, scenario=scenario,
                        run_name=run_dir_name, passed=False,
                        error=str(exc),
                    )
                    session.failed.append(eval_key)
                    print(f"  [ERROR] {exc}")

                results.append(result)
                session.save(session_path)

    except KeyboardInterrupt:
        print("\n[INFO] User interrupted (Ctrl+C)")
        interrupted = True

    # Summary
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    total_time = sum(r.duration_s for r in results)
    print(f"\n{'=' * 50}")
    print(f"EVAL MATRIX SUMMARY")
    print(f"Passed: {passed}/{len(results)}, Failed: {failed}")
    print(f"Total time: {total_time / 3600:.1f}h")
    print(f"Output dir: {args.output_dir}")
    print(f"{'=' * 50}")

    if interrupted:
        sys.exit(130)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
