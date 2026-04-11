#!/usr/bin/env python3
"""
Phase 4 ablation evaluation runner.

Expands the frozen `phase4_ablation_manifest.json` into the formal
six-scenario evaluation matrix used by S4-8.

Usage:
    conda activate env_isaaclab

    # Default: 2 ablations x 3 seeds x 6 scenarios = 36 evals
    python scripts/phase_morl/run_phase4_ablation_eval.py

    # Include anchor-full control as well
    python scripts/phase_morl/run_phase4_ablation_eval.py --include-anchor-full

    # Only evaluate one ablation entry
    python scripts/phase_morl/run_phase4_ablation_eval.py --ablation-ids anchor-no-energy

    # Validate run directories and checkpoints before execution
    python scripts/phase_morl/run_phase4_ablation_eval.py --validate
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_ABLATION_MANIFEST = SCRIPT_DIR / "manifests" / "phase4_ablation_manifest.json"

DEFAULT_EVAL_TASK = "Isaac-Velocity-MORL-Unitree-Go1-ROS2Cmd-Play-v2"
DEFAULT_SCENARIOS = ["S1", "S2", "S3", "S4", "S5", "S6"]
DEFAULT_EVAL_STEPS = 3000
DEFAULT_WARMUP_STEPS = 300
DEFAULT_NUM_ENVS = 64
DEFAULT_CHECKPOINT = "model_899.pt"


@dataclass(frozen=True)
class AblationEvalTarget:
    run_dir_name: str
    output_stem: str
    policy_id: str
    seed: int
    task: str
    checkpoint: str


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
        with path.open("w", encoding="utf-8") as handle:
            json.dump(asdict(self), handle, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "EvalSession":
        if not path.exists():
            return cls(start_time=datetime.now().isoformat())
        with path.open("r", encoding="utf-8") as handle:
            return cls(**json.load(handle))

    def is_done(self, key: str) -> bool:
        return key in self.completed


def _normalize_entry_id(entry_id: str) -> str:
    alias_map = {
        "no-energy": "anchor-no-energy",
        "no-smooth": "anchor-no-smooth",
        "full": "anchor-full",
    }
    normalized = entry_id.strip().lower()
    return alias_map.get(normalized, normalized)


def _resolve_manifest_path(path: Path | str, *, project_root: Path) -> Path:
    manifest_path = Path(path)
    if not manifest_path.is_absolute():
        manifest_path = (project_root / manifest_path).resolve()
    return manifest_path


def load_phase4_ablation_manifest(path: Path | str = DEFAULT_ABLATION_MANIFEST, *, project_root: Path) -> dict:
    manifest_path = _resolve_manifest_path(path, project_root=project_root)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Phase 4 ablation manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def select_ablation_entries(
    manifest: dict,
    *,
    entry_ids: set[str] | None,
    include_anchor_full: bool,
) -> list[dict]:
    entries = manifest.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError("Invalid ablation manifest: missing 'entries' list")

    normalized_entry_ids = {_normalize_entry_id(entry_id) for entry_id in entry_ids} if entry_ids else None
    known_ids = {str(entry["ablation_id"]) for entry in entries}
    if normalized_entry_ids:
        unknown = sorted(normalized_entry_ids - known_ids)
        if unknown:
            raise ValueError(f"Unknown ablation entry ids: {unknown}. Known: {sorted(known_ids)}")

    selected: list[dict] = []
    for entry in entries:
        entry_id = str(entry["ablation_id"])
        role = str(entry.get("role", "ablation_variant"))
        if normalized_entry_ids is not None:
            if entry_id not in normalized_entry_ids:
                continue
        else:
            if role != "ablation_variant" and not (include_anchor_full and role == "anchor_full"):
                continue
        selected.append(entry)
    return selected


def _discover_run_dir(run_root: Path, run_suffix: str) -> Path:
    exact_path = run_root / run_suffix
    if exact_path.exists():
        return exact_path

    matches = sorted(path for path in run_root.iterdir() if path.is_dir() and path.name.endswith(run_suffix)) if run_root.exists() else []
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous run directory for {run_suffix}: {[str(path) for path in matches]}")
    return exact_path


def load_eval_targets_from_ablation_manifest(
    manifest_path: Path | str = DEFAULT_ABLATION_MANIFEST,
    *,
    project_root: Path,
    run_root: Path,
    seeds: set[int] | None = None,
    entry_ids: set[str] | None = None,
    include_anchor_full: bool = False,
    task: str = DEFAULT_EVAL_TASK,
    checkpoint: str = DEFAULT_CHECKPOINT,
) -> list[AblationEvalTarget]:
    manifest = load_phase4_ablation_manifest(manifest_path, project_root=project_root)
    selected_entries = select_ablation_entries(
        manifest,
        entry_ids=entry_ids,
        include_anchor_full=include_anchor_full,
    )

    protocol = manifest.get("training_protocol", {})
    manifest_seeds = [int(seed) for seed in protocol.get("training_seeds", [42, 43, 44])]
    selected_seeds = sorted(seeds) if seeds is not None else manifest_seeds

    targets: list[AblationEvalTarget] = []
    for entry in selected_entries:
        name = str(entry["name"])
        policy_id = str(entry["policy_id"])
        for seed in selected_seeds:
            output_stem = f"{name}_seed{seed}"
            run_dir = _discover_run_dir(run_root, output_stem)
            targets.append(
                AblationEvalTarget(
                    run_dir_name=str(run_dir),
                    output_stem=output_stem,
                    policy_id=policy_id,
                    seed=seed,
                    task=task,
                    checkpoint=checkpoint,
                )
            )
    return targets


def validate_eval_targets(targets: list[AblationEvalTarget]) -> list[str]:
    errors: list[str] = []
    for target in targets:
        run_dir = Path(target.run_dir_name)
        if not run_dir.exists():
            errors.append(f"Missing run directory: {run_dir}")
            continue

        checkpoint_path = run_dir / target.checkpoint
        if not checkpoint_path.exists():
            errors.append(f"Missing checkpoint: {checkpoint_path}")
    return errors


def build_eval_cmd(
    run_dir_name: str,
    scenario: str,
    output_json: str,
    *,
    task: str,
    num_envs: int,
    eval_steps: int,
    warmup_steps: int,
    checkpoint: str,
) -> list[str]:
    eval_script = str(SCRIPT_DIR / "run_morl_eval.py")
    return [
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 4 ablation six-scenario evaluation runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_ABLATION_MANIFEST)
    parser.add_argument(
        "--ablation-ids",
        type=str,
        default=None,
        help="Comma-separated manifest entry ids. Aliases no-energy/no-smooth/full are supported.",
    )
    parser.add_argument("--include-anchor-full", action="store_true")
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--scenarios", type=str, default=",".join(DEFAULT_SCENARIOS))
    parser.add_argument("--task", type=str, default=DEFAULT_EVAL_TASK)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=PROJECT_ROOT / "logs" / "rsl_rl" / "unitree_go1_rough",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "logs" / "eval" / "phase_morl_v2",
    )
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--num-envs", type=int, default=DEFAULT_NUM_ENVS)
    parser.add_argument("--eval-steps", type=int, default=DEFAULT_EVAL_STEPS)
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate", action="store_true")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    scenarios = [item.strip().upper() for item in args.scenarios.split(",") if item.strip()]
    seeds = {int(item.strip()) for item in args.seeds.split(",") if item.strip()} if args.seeds else None
    entry_ids = {item.strip() for item in args.ablation_ids.split(",") if item.strip()} if args.ablation_ids else None

    try:
        targets = load_eval_targets_from_ablation_manifest(
            args.manifest,
            project_root=PROJECT_ROOT,
            run_root=args.run_root,
            seeds=seeds,
            entry_ids=entry_ids,
            include_anchor_full=args.include_anchor_full,
            task=args.task,
            checkpoint=args.checkpoint,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    if not targets:
        print("[ERROR] No ablation evaluation targets selected.")
        sys.exit(1)

    if args.validate:
        errors = validate_eval_targets(targets)
        if errors:
            print(f"[VALIDATE] Failed: {len(errors)} issue(s)")
            for err in errors:
                print(f"[VALIDATE] {err}")
            sys.exit(1)
        print(f"[VALIDATE] OK: {len(targets)} target(s) validated")
        sys.exit(0)

    total_evals = len(targets) * len(scenarios)
    print(f"[INFO] Selected {len(targets)} ablation run(s)")
    print(f"[INFO] Scenarios: {scenarios}")
    print(f"[INFO] Total evaluations: {total_evals}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    session_path = args.output_dir / "eval_matrix_session_ablation.json"
    session = EvalSession.load(session_path) if args.resume else EvalSession(start_time=datetime.now().isoformat())

    results: list[EvalResult] = []
    interrupted = False
    eval_idx = 0

    try:
        for target in targets:
            for scenario in scenarios:
                eval_idx += 1
                eval_key = f"{target.output_stem}_{scenario}"
                output_json = str(args.output_dir / f"{target.output_stem}_{scenario}.json")

                if args.resume and session.is_done(eval_key):
                    print(f"[SKIP] {eval_key} ({eval_idx}/{total_evals})")
                    continue

                print(f"\n[EVAL {eval_idx}/{total_evals}] {target.policy_id} seed={target.seed} {scenario}")
                cmd = build_eval_cmd(
                    target.run_dir_name,
                    scenario,
                    output_json,
                    task=target.task,
                    num_envs=args.num_envs,
                    eval_steps=args.eval_steps,
                    warmup_steps=args.warmup_steps,
                    checkpoint=target.checkpoint,
                )

                if args.dry_run:
                    print(f"  CMD: {' '.join(cmd)}")
                    results.append(
                        EvalResult(
                            policy_id=target.policy_id,
                            seed=target.seed,
                            scenario=scenario,
                            run_name=target.run_dir_name,
                            passed=True,
                            error="dry-run",
                        )
                    )
                    continue

                start_t = time.time()
                stderr_path = args.output_dir / f"{target.output_stem}_{scenario}_stderr.log"
                try:
                    with stderr_path.open("w", encoding="utf-8") as handle:
                        proc = subprocess.run(
                            cmd,
                            cwd=str(PROJECT_ROOT),
                            stdout=subprocess.PIPE,
                            stderr=handle,
                            timeout=600,
                        )
                    duration = time.time() - start_t
                    passed = proc.returncode == 0 and Path(output_json).exists()

                    result = EvalResult(
                        policy_id=target.policy_id,
                        seed=target.seed,
                        scenario=scenario,
                        run_name=target.run_dir_name,
                        passed=passed,
                        return_code=proc.returncode,
                        output_json=output_json if passed else None,
                        duration_s=duration,
                    )

                    if passed:
                        session.completed.append(eval_key)
                        stderr_path.unlink(missing_ok=True)
                        print(f"  [PASS] {duration:.0f}s -> {output_json}")
                    else:
                        session.failed.append(eval_key)
                        print(f"  [FAIL] rc={proc.returncode}, stderr: {stderr_path}")
                except subprocess.TimeoutExpired:
                    duration = time.time() - start_t
                    result = EvalResult(
                        policy_id=target.policy_id,
                        seed=target.seed,
                        scenario=scenario,
                        run_name=target.run_dir_name,
                        passed=False,
                        duration_s=duration,
                        error="timeout",
                    )
                    session.failed.append(eval_key)
                    print(f"  [TIMEOUT] after {duration:.0f}s")
                except Exception as exc:
                    result = EvalResult(
                        policy_id=target.policy_id,
                        seed=target.seed,
                        scenario=scenario,
                        run_name=target.run_dir_name,
                        passed=False,
                        error=str(exc),
                    )
                    session.failed.append(eval_key)
                    print(f"  [ERROR] {exc}")

                results.append(result)
                session.save(session_path)
    except KeyboardInterrupt:
        print("\n[INFO] User interrupted (Ctrl+C)")
        interrupted = True

    passed = sum(1 for result in results if result.passed)
    failed = sum(1 for result in results if not result.passed)
    total_time = sum(result.duration_s for result in results)

    print(f"\n{'=' * 50}")
    print("ABLATION EVAL SUMMARY")
    print(f"Passed: {passed}/{len(results)}, Failed: {failed}")
    print(f"Total time: {total_time / 3600:.1f}h")
    print(f"Output dir: {args.output_dir}")
    print(f"{'=' * 50}")

    if interrupted:
        sys.exit(130)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
