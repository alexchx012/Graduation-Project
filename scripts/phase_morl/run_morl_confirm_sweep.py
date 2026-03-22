#!/usr/bin/env python3
"""
Phase MORL M8: 代表性策略补训脚本

默认补训从 M7 探索层中选出的 5 个代表性策略：
- P1, P2, P3, P4: 四个单目标偏好点
- P10: 从 P5-P10 中选出的偏稳定折中点

默认训练配置：
- seeds = 42,43,44  (包含 seed42 重训)
- num_envs = 4096
- max_iterations = 1500
- clip_param = 0.3

工程特性复用 run_morl_train_sweep.py：
- ROS2 Publisher 已禁用（使用 Isaac Lab 随机命令）
- Isaac Sim 瞬态错误自动重试
- Checkpoint / env.yaml / agent.yaml 核验
- 断点续训
- 每次训练独立 stdout/stderr 日志

使用方法:
    conda activate env_isaaclab

    # 运行默认 5 策略 × 3 seeds（ROS2 已禁用）
    python scripts/phase_morl/run_morl_confirm_sweep.py

    # 只运行部分策略或自定义 seeds
    python scripts/phase_morl/run_morl_confirm_sweep.py --policy-ids P1,P10
    python scripts/phase_morl/run_morl_confirm_sweep.py --seeds 43

    # Dry run / Resume
    python scripts/phase_morl/run_morl_confirm_sweep.py --dry-run
    python scripts/phase_morl/run_morl_confirm_sweep.py --resume
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_morl_train_sweep import (
    DEFAULT_CLIP_PARAM,
    DualLogger,
    Ros2PublisherManager,
    RunResult,
    SessionState,
    _parse_seeds,
    check_prerequisites,
    print_summary,
    run_single_training,
    MORL_EXPERIMENTS,
    NUM_ENVS,
    MAX_ITERATIONS,
)


DEFAULT_CONFIRM_POLICY_IDS = ("P1", "P2", "P3", "P4", "P10")
DEFAULT_CONFIRM_SEEDS = [42, 43, 44]  # [2026-03-22] 包含 seed42 重训（ROS2 命令源修复后）


def _select_confirmation_experiments(policy_ids_raw: str | None) -> list[dict]:
    if not policy_ids_raw:
        wanted = set(DEFAULT_CONFIRM_POLICY_IDS)
    else:
        wanted = {item.strip().upper() for item in policy_ids_raw.split(",") if item.strip()}

    selected = [exp for exp in MORL_EXPERIMENTS if exp["policy_id"] in wanted]
    if len(selected) != len(wanted):
        known = {exp["policy_id"] for exp in MORL_EXPERIMENTS}
        unknown = sorted(wanted - known)
        raise ValueError(f"Unknown policy ids: {unknown}. Known: {sorted(known)}")

    selected.sort(key=lambda exp: list(DEFAULT_CONFIRM_POLICY_IDS).index(exp["policy_id"]) if exp["policy_id"] in DEFAULT_CONFIRM_POLICY_IDS else exp["policy_id"])
    return selected


def main():
    parser = argparse.ArgumentParser(
        description="Phase MORL M8: 代表性策略补训",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--policy-ids",
        type=str,
        default=None,
        help="仅运行指定策略，如 P1,P10（默认 P1,P2,P3,P4,P10）",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,43,44",
        help="逗号分隔种子列表（默认 43,44）",
    )
    parser.add_argument("--num-envs", type=int, default=NUM_ENVS, help="训练 env 数量")
    parser.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS, help="训练迭代数")
    parser.add_argument("--clip-param", type=float, default=DEFAULT_CLIP_PARAM, help="PPO clip_param")
    parser.add_argument("--skip-ros2", action="store_true", help="跳过 ROS2 publisher 管理")
    parser.add_argument("--dry-run", action="store_true", help="只打印命令，不实际训练")
    parser.add_argument("--resume", action="store_true", help="断点续训：跳过已完成实验")
    parser.add_argument("--project-root", type=Path, default=None, help="项目根目录（默认自动检测）")
    parser.add_argument(
        "--ros2-script",
        type=Path,
        default=None,
        help="可选：覆盖 WSL ROS2 publisher 脚本路径",
    )
    args = parser.parse_args()

    # [2026-03-22] 强制禁用 ROS2 publisher 管理
    # 原因：rough_env_cfg.py 已禁用 ROS2 命令源，使用 Isaac Lab 标准随机命令生成器
    # 详见：docs/daily_logs/2026-3/2026-03-22/2026-3-22.md
    args.skip_ros2 = True

    project_root = args.project_root or Path(__file__).resolve().parents[2]
    if not (project_root / "CLAUDE.md").exists():
        print(f"[ERROR] 项目根目录无效: {project_root}")
        sys.exit(1)

    try:
        experiments = _select_confirmation_experiments(args.policy_ids)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    seeds = _parse_seeds(args.seeds) if args.seeds else DEFAULT_CONFIRM_SEEDS
    total_runs = len(experiments) * len(seeds)

    # [2026-03-22] 简化前置检查：只检查训练脚本，跳过 ROS2 检查
    train_script = project_root / "scripts" / "go1-ros2-test" / "train.py"
    if not train_script.exists():
        print(f"[ERROR] 训练脚本不存在: {train_script}")
        sys.exit(1)

    print(f"\n[INFO] 默认确认策略: {DEFAULT_CONFIRM_POLICY_IDS}")
    print(f"[INFO] 本次策略: {[exp['policy_id'] for exp in experiments]}, seeds: {seeds}, 总计: {total_runs} 次训练")

    log_dir = project_root / "logs" / "sweep" / "phase_morl_confirm"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    policy_tag = "default" if args.policy_ids is None else args.policy_ids.replace(",", "_").replace(" ", "")
    log_path = log_dir / f"phase_morl_confirm_{policy_tag}_{timestamp}.log"
    log = DualLogger(log_path)
    log.info(f"Project root: {project_root}")
    log.info(f"Default selected policies for M8: {DEFAULT_CONFIRM_POLICY_IDS}")
    log.info(f"Policies this run: {[exp['policy_id'] for exp in experiments]}")
    log.info(f"Seeds: {seeds}")
    log.info(f"num_envs={args.num_envs}, max_iterations={args.max_iterations}, clip_param={args.clip_param}")

    session_path = log_dir / f"session_{policy_tag}.json"
    session = SessionState.load(session_path) if args.resume else SessionState(
        start_time=datetime.now().isoformat()
    )

    ros2_mgr: Ros2PublisherManager | None = None
    if not args.skip_ros2 and not args.dry_run:
        ros2_mgr = Ros2PublisherManager(project_root, log, args.ros2_script)
        if not ros2_mgr.start():
            log.error("ROS2 publisher 启动失败，退出")
            log.close()
            sys.exit(1)

    results: list[RunResult] = []
    interrupted = False

    try:
        run_idx = 0
        for exp in experiments:
            for seed in seeds:
                run_idx += 1
                run_name = f"{exp['name']}_seed{seed}"
                if args.resume and session.is_done(run_name):
                    log.info(f"\n[SKIP] {run_name} 已完成，跳过 ({run_idx}/{total_runs})")
                    continue

                log.info(f"\n[PROGRESS] {run_idx}/{total_runs}")
                result = run_single_training(
                    exp,
                    seed,
                    project_root,
                    log,
                    ros2_mgr,
                    dry_run=args.dry_run,
                    num_envs=args.num_envs,
                    max_iterations=args.max_iterations,
                    clip_param=args.clip_param,
                )
                results.append(result)

                if result.passed:
                    session.completed.append(run_name)
                else:
                    session.failed.append(run_name)
                session.save(session_path)

    except KeyboardInterrupt:
        log.warn("\n用户中断 (Ctrl+C)")
        interrupted = True
    finally:
        if ros2_mgr is not None:
            ros2_mgr.stop()

    if results:
        print_summary(results, log)

    log.close()
    print(f"\n[INFO] 完整日志: {log_path}")
    if args.resume:
        print(f"[INFO] 会话状态: {session_path}")

    if interrupted:
        sys.exit(130)
    all_passed = all(result.passed for result in results) if results else False
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
