#!/usr/bin/env python3
"""
Phase 4 Step 4.0: Baseline 3-seed 补训脚本

补训 Rough ROS2Cmd baseline 的 seed=43, 44，与现有 seed=42 组成 3 种子统计基础。
训练配置与 Phase 1 baseline 完全一致：vx=1.0, 50Hz, 4096 envs, 1500 iters。

使用方法:
    # 在 PowerShell 中运行（需要先激活 conda 环境）
    conda activate env_isaaclab
    python docs/daily_logs/2026-3/2026-03-10/run_baseline_seed_补训.py

    # 只训练单个种子
    python docs/daily_logs/2026-3/2026-03-10/run_baseline_seed_补训.py --seeds 43

    # 跳过 ROS2 publisher（如果已在其他终端启动）
    python docs/daily_logs/2026-3/2026-03-10/run_baseline_seed_补训.py --skip-ros2

    # Dry run（只打印命令，不执行）
    python docs/daily_logs/2026-3/2026-03-10/run_baseline_seed_补训.py --dry-run
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class TrainConfig:
    """训练配置，与 Phase 1 baseline 完全一致"""
    task: str = "Isaac-Velocity-Rough-Unitree-Go1-ROS2Cmd-v0"
    num_envs: int = 4096
    max_iterations: int = 1500
    wandb_project: str = "go1-rough-locomotion"
    # Phase 1 baseline 使用 vx=1.0, 50Hz
    ros2_vx: float = 1.0
    ros2_rate: int = 50


@dataclass
class RunResult:
    """单次训练结果"""
    seed: int
    success: bool
    checkpoint_path: Optional[str] = None
    run_dir: Optional[str] = None
    duration_s: float = 0.0
    error_msg: Optional[str] = None


@dataclass
class SessionState:
    """会话状态，用于断点续训"""
    completed_seeds: list = field(default_factory=list)
    failed_seeds: list = field(default_factory=list)
    start_time: str = ""

    def save(self, path: Path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'completed_seeds': self.completed_seeds,
                'failed_seeds': self.failed_seeds,
                'start_time': self.start_time,
            }, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'SessionState':
        if not path.exists():
            return cls(start_time=datetime.now().isoformat())
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return cls(**data)


class Ros2PublisherManager:
    """WSL ROS2 Publisher 生命周期管理"""

    def __init__(self, project_root: Path, config: TrainConfig):
        self.project_root = project_root
        self.config = config
        self.wsl_process: Optional[subprocess.Popen] = None
        self._ros2_script = project_root / "scripts/baseline-repro/Phase1-Baseline/run_ros2_cmd.sh"

    def start(self) -> bool:
        """启动 WSL ROS2 publisher，返回是否成功"""
        if not self._ros2_script.exists():
            print(f"[ERROR] ROS2 脚本不存在: {self._ros2_script}")
            return False

        # 先清理可能残留的 publisher 进程
        self._cleanup_stale_publishers()

        # 转换路径为 WSL 格式
        wsl_project_root = self._get_wsl_path(self.project_root)
        wsl_script = f"{wsl_project_root}/scripts/baseline-repro/Phase1-Baseline/run_ros2_cmd.sh"

        cmd = [
            "wsl", "-d", "Ubuntu-22.04", "bash", "-lc",
            f"cd '{wsl_project_root}' && bash '{wsl_script}'"
        ]

        print(f"[ROS2] 启动 WSL publisher (vx={self.config.ros2_vx}, {self.config.ros2_rate}Hz)...")

        try:
            # 使用 DEVNULL 避免 Windows 上的 GBK 编码问题
            # WSL ROS2 输出包含 UTF-8 字符，PIPE + GBK 解码会报错
            self.wsl_process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == 'win32' else 0
            )

            # 等待 publisher 稳定
            time.sleep(5)

            # 验证 publisher 是否在运行
            if not self._is_publisher_running():
                print("[ERROR] ROS2 publisher 启动失败或已退出")
                return False

            # 验证 ROS2 topic 是否在发布
            if not self._verify_topic_publishing():
                print("[WARN] ROS2 topic 验证失败，但 publisher 进程存在，继续...")

            print(f"[ROS2] Publisher 已启动 (WSL PID: {self._get_publisher_pid()})")
            return True

        except Exception as e:
            print(f"[ERROR] 启动 ROS2 publisher 失败: {e}")
            return False

    def stop(self):
        """停止 WSL ROS2 publisher"""
        print("[ROS2] 停止 publisher...")

        try:
            # 先尝试优雅终止 WSL 内的进程
            subprocess.run(
                ["wsl", "-d", "Ubuntu-22.04", "bash", "-c",
                 "pkill -f go1_cmd_script_node.py 2>/dev/null || true"],
                timeout=5, capture_output=True
            )
        except Exception:
            pass

        # 终止 WSL 进程
        if self.wsl_process and self.wsl_process.poll() is None:
            try:
                self.wsl_process.terminate()
                self.wsl_process.wait(timeout=3)
            except Exception:
                self.wsl_process.kill()

        self.wsl_process = None
        print("[ROS2] Publisher 已停止")

    def health_check(self) -> bool:
        """检查 publisher 是否健康运行"""
        if self.wsl_process is None:
            return False
        if self.wsl_process.poll() is not None:
            return False
        return self._is_publisher_running()

    def _is_publisher_running(self) -> bool:
        """检查 WSL 内 publisher 进程是否存在"""
        try:
            result = subprocess.run(
                ["wsl", "-d", "Ubuntu-22.04", "bash", "-c",
                 "pgrep -f go1_cmd_script_node.py"],
                capture_output=True, text=True, timeout=5
            )
            return bool(result.stdout.strip())
        except Exception:
            return False

    def _get_publisher_pid(self) -> str:
        """获取 WSL 内 publisher 的 PID"""
        try:
            result = subprocess.run(
                ["wsl", "-d", "Ubuntu-22.04", "bash", "-c",
                 "pgrep -f go1_cmd_script_node.py | head -n 1"],
                capture_output=True, text=True, timeout=5
            )
            return result.stdout.strip() or "unknown"
        except Exception:
            return "unknown"

    def _cleanup_stale_publishers(self):
        """清理残留的 publisher 进程"""
        try:
            subprocess.run(
                ["wsl", "-d", "Ubuntu-22.04", "bash", "-c",
                 "pkill -f go1_cmd_script_node.py 2>/dev/null || true"],
                timeout=5, capture_output=True
            )
            time.sleep(1)
        except Exception:
            pass

    def _verify_topic_publishing(self) -> bool:
        """验证 ROS2 topic 是否在发布消息"""
        try:
            # 在 WSL 中检查 topic 是否存在
            result = subprocess.run(
                ["wsl", "-d", "Ubuntu-22.04", "bash", "-lc",
                 "source /opt/ros/humble/setup.bash && ros2 topic list 2>/dev/null | grep -q /go1/cmd_vel && echo OK"],
                capture_output=True, text=True, timeout=10
            )
            if "OK" in result.stdout:
                print("[ROS2] Topic /go1/cmd_vel 已注册 ✓")
                return True
            else:
                print("[WARN] Topic /go1/cmd_vel 未找到")
                return False
        except Exception as e:
            print(f"[WARN] Topic 验证异常: {e}")
            return False

    def _get_wsl_path(self, windows_path: Path) -> str:
        """将 Windows 路径转换为 WSL 路径"""
        path_str = str(windows_path).replace('\\', '/')
        try:
            result = subprocess.run(
                ["wsl", "-d", "Ubuntu-22.04", "bash", "-c", f"wslpath -u '{path_str}'"],
                capture_output=True, text=True, timeout=5
            )
            return result.stdout.strip()
        except Exception:
            # Fallback: 手动转换
            if path_str[1] == ':':
                drive = path_str[0].lower()
                return f"/mnt/{drive}{path_str[2:]}"
            return path_str


class BaselineTrainer:
    """Baseline 补训执行器"""

    def __init__(self, project_root: Path, config: TrainConfig, dry_run: bool = False):
        self.project_root = project_root
        self.config = config
        self.dry_run = dry_run
        self.train_script = project_root / "scripts/go1-ros2-test/train.py"
        self.logs_dir = project_root / "logs/rsl_rl/unitree_go1_rough"

    def train_seed(self, seed: int) -> RunResult:
        """训练单个种子"""
        run_name = f"baseline_rough_seed{seed}"

        cmd = [
            sys.executable, str(self.train_script),
            "--task", self.config.task,
            "--num_envs", str(self.config.num_envs),
            "--max_iterations", str(self.config.max_iterations),
            "--seed", str(seed),
            "--logger", "wandb",
            "--log_project_name", self.config.wandb_project,
            "--run_name", run_name,
            "--headless",
            "--disable_ros2_tracking_tune"
        ]

        print(f"\n{'='*60}")
        print(f"[TRAIN] Seed={seed}, RunName={run_name}")
        print(f"{'='*60}")
        print(f"[CMD] {' '.join(cmd)}")

        if self.dry_run:
            print("[DRY-RUN] 跳过实际训练")
            return RunResult(seed=seed, success=True, error_msg="dry-run")

        start_time = time.time()

        try:
            result = subprocess.run(cmd, check=True)
            duration = time.time() - start_time

            # 查找生成的 checkpoint
            checkpoint_path, run_dir = self._find_checkpoint(run_name)

            if checkpoint_path:
                print(f"[SUCCESS] Seed={seed} 训练完成")
                print(f"  Checkpoint: {checkpoint_path}")
                print(f"  Duration: {duration/3600:.2f}h")
                return RunResult(
                    seed=seed, success=True,
                    checkpoint_path=str(checkpoint_path),
                    run_dir=str(run_dir),
                    duration_s=duration
                )
            else:
                return RunResult(
                    seed=seed, success=False,
                    duration_s=duration,
                    error_msg="Checkpoint not found after training"
                )

        except subprocess.CalledProcessError as e:
            duration = time.time() - start_time
            print(f"[FAILED] Seed={seed} 训练失败: {e}")
            return RunResult(
                seed=seed, success=False,
                duration_s=duration,
                error_msg=str(e)
            )
        except KeyboardInterrupt:
            print(f"\n[INTERRUPTED] Seed={seed} 训练被中断")
            raise

    def _find_checkpoint(self, run_name: str) -> tuple[Optional[Path], Optional[Path]]:
        """查找训练生成的 checkpoint"""
        if not self.logs_dir.exists():
            return None, None

        # 查找包含 run_name 的最新目录
        matching_dirs = sorted(
            [d for d in self.logs_dir.iterdir() if d.is_dir() and run_name in d.name],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )

        if not matching_dirs:
            return None, None

        run_dir = matching_dirs[0]
        checkpoint = run_dir / "model_1499.pt"

        if checkpoint.exists():
            return checkpoint, run_dir

        # 尝试查找最新的 model_*.pt
        model_files = sorted(run_dir.glob("model_*.pt"), reverse=True)
        if model_files:
            return model_files[0], run_dir

        return None, run_dir

    def verify_checkpoint(self, run_dir: Path, seed: int) -> bool:
        """验证 checkpoint 的 agent.yaml 中 seed 值正确"""
        agent_yaml = run_dir / "params/agent.yaml"
        if not agent_yaml.exists():
            print(f"[WARN] agent.yaml 不存在: {agent_yaml}")
            return False

        with open(agent_yaml, 'r', encoding='utf-8') as f:
            content = f.read()

        # 简单检查 seed 值
        if f"seed: {seed}" in content:
            print(f"[VERIFY] agent.yaml seed={seed} ✓")
            return True
        else:
            print(f"[WARN] agent.yaml seed 值不匹配，预期 {seed}")
            return False


def check_prerequisites(project_root: Path) -> bool:
    """检查前置条件"""
    print("[CHECK] 检查前置条件...")

    # 检查训练脚本
    train_script = project_root / "scripts/go1-ros2-test/train.py"
    if not train_script.exists():
        print(f"[ERROR] 训练脚本不存在: {train_script}")
        return False

    # 检查 ROS2 脚本
    ros2_script = project_root / "scripts/baseline-repro/Phase1-Baseline/run_ros2_cmd.sh"
    if not ros2_script.exists():
        print(f"[ERROR] ROS2 脚本不存在: {ros2_script}")
        return False

    # 检查 conda 环境
    conda_prefix = os.environ.get('CONDA_PREFIX', '')
    if 'env_isaaclab' not in conda_prefix:
        print(f"[WARN] 当前 conda 环境可能不正确: {conda_prefix}")
        print("       建议先运行: conda activate env_isaaclab")

    # 检查磁盘空间（每个 checkpoint 约 100MB，加上日志约 200MB）
    logs_dir = project_root / "logs"
    if logs_dir.exists():
        free_space = shutil.disk_usage(logs_dir).free / (1024**3)  # GB
        required_space = 0.5  # 2 seeds × ~200MB
        if free_space < required_space:
            print(f"[ERROR] 磁盘空间不足: {free_space:.1f}GB < {required_space}GB")
            return False
        print(f"[CHECK] 磁盘空间: {free_space:.1f}GB ✓")

    # 检查现有 baseline (seed=42)
    existing_baseline = list((project_root / "logs/rsl_rl/unitree_go1_rough").glob("*baseline_rough_ros2cmd*"))
    if existing_baseline:
        print(f"[CHECK] 现有 baseline: {existing_baseline[0].name} ✓")
    else:
        print("[WARN] 未找到现有 baseline (seed=42)，请确认 Phase 1 已完成")

    print("[CHECK] 前置条件检查完成 ✓")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Phase 4 Step 4.0: Baseline 3-seed 补训",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[43, 44],
        help="要训练的种子列表 (default: 43 44)"
    )
    parser.add_argument(
        "--skip-ros2", action="store_true",
        help="跳过 ROS2 publisher 启动（假设已在其他终端运行）"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只打印命令，不实际执行训练"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="从上次中断处继续（跳过已完成的种子）"
    )
    parser.add_argument(
        "--project-root", type=Path, default=None,
        help="项目根目录 (default: 自动检测)"
    )

    args = parser.parse_args()

    # 确定项目根目录
    if args.project_root:
        project_root = args.project_root
    else:
        # 从脚本位置推断
        script_path = Path(__file__).resolve()
        project_root = script_path.parents[4]  # docs/daily_logs/2026-3/2026-03-10/ -> project root

    if not (project_root / "CLAUDE.md").exists():
        print(f"[ERROR] 无法确定项目根目录，请使用 --project-root 指定")
        sys.exit(1)

    print(f"[INFO] 项目根目录: {project_root}")
    print(f"[INFO] 待训练种子: {args.seeds}")

    # 检查前置条件
    if not check_prerequisites(project_root):
        sys.exit(1)

    config = TrainConfig()

    # 会话状态管理
    state_file = project_root / "docs/daily_logs/2026-3/2026-03-10/.补训_session_state.json"
    state = SessionState.load(state_file) if args.resume else SessionState(start_time=datetime.now().isoformat())

    # 过滤已完成的种子
    seeds_to_train = [s for s in args.seeds if s not in state.completed_seeds]
    if not seeds_to_train:
        print("[INFO] 所有种子已完成训练")
        return

    if args.resume and state.completed_seeds:
        print(f"[RESUME] 已完成种子: {state.completed_seeds}")
        print(f"[RESUME] 待训练种子: {seeds_to_train}")

    # ROS2 Publisher 管理
    ros2_manager: Optional[Ros2PublisherManager] = None

    if not args.skip_ros2 and not args.dry_run:
        ros2_manager = Ros2PublisherManager(project_root, config)
        if not ros2_manager.start():
            print("[ERROR] ROS2 publisher 启动失败，退出")
            sys.exit(1)

    trainer = BaselineTrainer(project_root, config, dry_run=args.dry_run)
    results: list[RunResult] = []

    # 注册信号处理，确保清理
    def signal_handler(sig, frame):
        print("\n[SIGNAL] 收到中断信号，正在清理...")
        if ros2_manager:
            ros2_manager.stop()
        state.save(state_file)
        sys.exit(1)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        for i, seed in enumerate(seeds_to_train):
            print(f"\n[PROGRESS] 训练 {i+1}/{len(seeds_to_train)}: seed={seed}")

            # 健康检查 ROS2 publisher
            if ros2_manager and not ros2_manager.health_check():
                print("[WARN] ROS2 publisher 异常，尝试重启...")
                ros2_manager.stop()
                time.sleep(2)
                if not ros2_manager.start():
                    print("[ERROR] ROS2 publisher 重启失败")
                    break

            result = trainer.train_seed(seed)
            results.append(result)

            if result.success:
                state.completed_seeds.append(seed)
                # 验证 checkpoint
                if result.run_dir:
                    trainer.verify_checkpoint(Path(result.run_dir), seed)
            else:
                state.failed_seeds.append(seed)

            # 保存状态（支持断点续训）
            state.save(state_file)

    finally:
        if ros2_manager:
            ros2_manager.stop()

    # 打印汇总
    print(f"\n{'='*60}")
    print("[SUMMARY] 补训完成")
    print(f"{'='*60}")

    success_count = sum(1 for r in results if r.success)
    total_duration = sum(r.duration_s for r in results)

    print(f"  成功: {success_count}/{len(results)}")
    print(f"  总耗时: {total_duration/3600:.2f}h")

    for r in results:
        status = "✓" if r.success else "✗"
        print(f"  [{status}] seed={r.seed}: {r.checkpoint_path or r.error_msg}")

    # 清理状态文件（如果全部成功）
    if all(r.success for r in results):
        if state_file.exists():
            state_file.unlink()
        print("\n[DONE] 所有种子训练成功，可继续执行步骤 4.1")
    else:
        print(f"\n[WARN] 部分种子失败，状态已保存到 {state_file}")
        print("       使用 --resume 可从断点继续")


if __name__ == "__main__":
    main()
