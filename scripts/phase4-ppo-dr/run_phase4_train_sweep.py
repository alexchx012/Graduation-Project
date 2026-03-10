#!/usr/bin/env python3
"""
Phase 4 Step 4.4b: 批量训练 Sweep 脚本

自动化执行 PPO / DR / Combo 训练实验，支持：
- PPO OFAT 实验（6 组 × 3 种子 = 18 次）
- DR 变体实验（3 组 × 3 种子 = 9 次）
- 组合验证实验（1 组 × 3 种子 = 3 次，需手动指定 best PPO + best DR）

功能特性：
- ROS2 Publisher 生命周期管理 + 5 分钟 Watchdog
- Isaac Sim 瞬态错误自动重试（最多 3 次）
- Checkpoint 验证 + agent.yaml PPO 参数核验
- 发散检测（mean_reward < baseline 30% 标记为发散）
- 会话断点续训（SessionState JSON）
- 每次训练独立 stdout/stderr 日志

使用方法:
    conda activate env_isaaclab

    # PPO 全部 18 次
    python scripts/go1-ros2-test/run/phase4/run_phase4_train_sweep.py --mode ppo

    # DR 全部 9 次
    python scripts/go1-ros2-test/run/phase4/run_phase4_train_sweep.py --mode dr

    # 组合验证（需先完成 4.5+4.6 并手动填入 best 配置）
    python scripts/go1-ros2-test/run/phase4/run_phase4_train_sweep.py --mode combo

    # 全部（PPO + DR）
    python scripts/go1-ros2-test/run/phase4/run_phase4_train_sweep.py --mode all

    # 跳过 ROS2 / Dry run
    python scripts/go1-ros2-test/run/phase4/run_phase4_train_sweep.py --mode ppo --skip-ros2
    python scripts/go1-ros2-test/run/phase4/run_phase4_train_sweep.py --mode ppo --dry-run

    # 断点续训（自动跳过已完成的实验）
    python scripts/go1-ros2-test/run/phase4/run_phase4_train_sweep.py --mode ppo --resume
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional


# ── 常量 ─────────────────────────────────────────────────────────────────

SEEDS = [42, 43, 44]
BASELINE_TASK = "Isaac-Velocity-Rough-Unitree-Go1-ROS2Cmd-v0"
NUM_ENVS = 4096
MAX_ITERATIONS = 1500
MAX_RETRY = 3                     # Isaac Sim 瞬态错误最大重试次数
WATCHDOG_INTERVAL_S = 300         # ROS2 Publisher 健康检查间隔（5 分钟）
DIVERGENCE_THRESHOLD = 0.3       # 发散判定：最终 reward < baseline * 此比例
BASELINE_REWARD_ESTIMATE = 15.0   # Baseline 收敛后大致 reward（用于发散粗判）

# PPO OFAT 实验矩阵（6 组）— CLI 覆盖 baseline 任务
PPO_EXPERIMENTS = [
    {"name": "ppo_lr_low",   "learning_rate": 5e-4,  "clip_param": None, "entropy_coef": None},
    {"name": "ppo_lr_high",  "learning_rate": 3e-3,  "clip_param": None, "entropy_coef": None},
    {"name": "ppo_clip_low", "learning_rate": None,   "clip_param": 0.1,  "entropy_coef": None},
    {"name": "ppo_clip_high","learning_rate": None,   "clip_param": 0.3,  "entropy_coef": None},
    {"name": "ppo_ent_low",  "learning_rate": None,   "clip_param": None, "entropy_coef": 0.005},
    {"name": "ppo_ent_high", "learning_rate": None,   "clip_param": None, "entropy_coef": 0.02},
]

# DR 变体实验矩阵（3 组）— 各有独立任务 ID
DR_EXPERIMENTS = [
    {"name": "dr_friction", "task": "Isaac-Velocity-Rough-Unitree-Go1-ROS2Cmd-DRFriction-v0"},
    {"name": "dr_mass",     "task": "Isaac-Velocity-Rough-Unitree-Go1-ROS2Cmd-DRMass-v0"},
    {"name": "dr_push",     "task": "Isaac-Velocity-Rough-Unitree-Go1-ROS2Cmd-DRPush-v0"},
]

# 组合验证实验（步骤 4.6b，需手动指定 best PPO + best DR）
# 训练完成后根据结果修改此处
COMBO_EXPERIMENTS = [
    {
        "name": "combo_best",
        "task": "Isaac-Velocity-Rough-Unitree-Go1-ROS2Cmd-DRPush-v0",  # 替换为 best DR
        "learning_rate": 5e-4,   # 替换为 best PPO lr（如适用）
        "clip_param": None,      # 替换为 best PPO clip（如适用）
        "entropy_coef": None,    # 替换为 best PPO ent（如适用）
    },
]


# ── 数据类 ───────────────────────────────────────────────────────────────

@dataclass
class RunResult:
    """单次训练结果"""
    exp_name: str
    seed: int
    run_name: str
    passed: bool
    return_code: int = -1
    checkpoint_path: Optional[str] = None
    run_dir: Optional[str] = None
    duration_s: float = 0.0
    attempts: int = 0
    diverged: bool = False
    error: Optional[str] = None


@dataclass
class SessionState:
    """会话状态，支持断点续训"""
    completed: list = field(default_factory=list)   # ["ppo_lr_low_seed42", ...]
    failed: list = field(default_factory=list)
    start_time: str = ""

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "SessionState":
        if not path.exists():
            return cls(start_time=datetime.now().isoformat())
        with open(path, "r", encoding="utf-8") as f:
            return cls(**json.load(f))

    def is_done(self, run_name: str) -> bool:
        return run_name in self.completed


# ── DualLogger ───────────────────────────────────────────────────────────

class DualLogger:
    """同时写控制台和日志文件"""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(log_path, "a", encoding="utf-8")
        self._write(f"\n{'='*70}")
        self._write(f"Started: {datetime.now().isoformat()}")

    def info(self, msg: str):
        self._write(f"[INFO] {msg}")

    def warn(self, msg: str):
        self._write(f"[WARN] {msg}")

    def error(self, msg: str):
        self._write(f"[ERROR] {msg}")

    def _write(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        self._f.write(line + "\n")
        self._f.flush()

    def close(self):
        self._write(f"Finished: {datetime.now().isoformat()}")
        self._f.close()


# ── ROS2 Publisher 管理 ──────────────────────────────────────────────────

class Ros2PublisherManager:
    """WSL ROS2 Publisher 生命周期管理 + Watchdog"""

    def __init__(self, project_root: Path, log: DualLogger):
        self.project_root = project_root
        self.log = log
        self.wsl_process: Optional[subprocess.Popen] = None
        self._ros2_script = project_root / "scripts/baseline-repro/Phase1-Baseline/run_ros2_cmd.sh"
        self._restart_count = 0

    def start(self) -> bool:
        """启动 WSL ROS2 publisher"""
        if not self._ros2_script.exists():
            self.log.error(f"ROS2 脚本不存在: {self._ros2_script}")
            return False

        self._cleanup_stale()

        wsl_root = self._to_wsl_path(self.project_root)
        wsl_script = f"{wsl_root}/scripts/baseline-repro/Phase1-Baseline/run_ros2_cmd.sh"

        cmd = ["wsl", "-d", "Ubuntu-22.04", "bash", "-lc",
               f"cd '{wsl_root}' && bash '{wsl_script}'"]

        self.log.info("启动 WSL ROS2 publisher (vx=1.0, 50Hz)...")
        try:
            self.wsl_process = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
            )
            time.sleep(5)
            if not self._is_alive():
                self.log.error("ROS2 publisher 启动后立即退出")
                return False
            self.log.info(f"ROS2 publisher 已启动 (PID: {self._get_pid()})")
            return True
        except Exception as e:
            self.log.error(f"启动 ROS2 publisher 失败: {e}")
            return False

    def stop(self):
        """停止 publisher"""
        self.log.info("停止 ROS2 publisher...")
        try:
            subprocess.run(
                ["wsl", "-d", "Ubuntu-22.04", "bash", "-c",
                 "pkill -f go1_cmd_script_node.py 2>/dev/null || true"],
                timeout=5, capture_output=True,
            )
        except Exception:
            pass
        if self.wsl_process and self.wsl_process.poll() is None:
            try:
                self.wsl_process.terminate()
                self.wsl_process.wait(timeout=3)
            except Exception:
                self.wsl_process.kill()
        self.wsl_process = None
        self.log.info("ROS2 publisher 已停止")

    def health_check(self) -> bool:
        """Watchdog 健康检查"""
        if self.wsl_process is None or self.wsl_process.poll() is not None:
            return False
        return self._is_alive()

    def ensure_alive(self) -> bool:
        """确保 publisher 存活，必要时自动重启（最多 3 次）"""
        if self.health_check():
            return True
        self.log.warn("ROS2 publisher 已挂掉，尝试重启...")
        self._restart_count += 1
        if self._restart_count > 3:
            self.log.error("ROS2 publisher 重启次数超限（>3），放弃")
            return False
        self.stop()
        time.sleep(2)
        return self.start()

    def _is_alive(self) -> bool:
        try:
            r = subprocess.run(
                ["wsl", "-d", "Ubuntu-22.04", "bash", "-c",
                 "pgrep -f go1_cmd_script_node.py"],
                capture_output=True, text=True, timeout=5,
            )
            return bool(r.stdout.strip())
        except Exception:
            return False

    def _get_pid(self) -> str:
        try:
            r = subprocess.run(
                ["wsl", "-d", "Ubuntu-22.04", "bash", "-c",
                 "pgrep -f go1_cmd_script_node.py | head -n 1"],
                capture_output=True, text=True, timeout=5,
            )
            return r.stdout.strip() or "unknown"
        except Exception:
            return "unknown"

    def _cleanup_stale(self):
        try:
            subprocess.run(
                ["wsl", "-d", "Ubuntu-22.04", "bash", "-c",
                 "pkill -f go1_cmd_script_node.py 2>/dev/null || true"],
                timeout=5, capture_output=True,
            )
            time.sleep(1)
        except Exception:
            pass

    def _to_wsl_path(self, p: Path) -> str:
        s = str(p).replace("\\", "/")
        try:
            r = subprocess.run(
                ["wsl", "-d", "Ubuntu-22.04", "bash", "-c", f"wslpath -u '{s}'"],
                capture_output=True, text=True, timeout=5,
            )
            return r.stdout.strip()
        except Exception:
            if s[1] == ":":
                return f"/mnt/{s[0].lower()}{s[2:]}"
            return s




# ── 训练执行 ─────────────────────────────────────────────────────────────

TRANSIENT_ERROR_SIGNATURES = [
    "PytorchStreamReader failed reading zip archive",
    "failed finding central directory",
    "CUDA out of memory",
    "cuDNN error",
]


def _is_transient_error(stderr_path: Path) -> bool:
    """检查 stderr 日志中是否包含已知的 Isaac Sim 瞬态错误"""
    if not stderr_path.exists():
        return False
    try:
        content = stderr_path.read_text(encoding="utf-8", errors="replace")
        return any(sig in content for sig in TRANSIENT_ERROR_SIGNATURES)
    except Exception:
        return False


def _build_train_cmd(
    project_root: Path, exp: dict, seed: int, run_name: str
) -> list[str]:
    """构建训练命令"""
    train_script = project_root / "scripts/go1-ros2-test/train.py"
    task = exp.get("task", BASELINE_TASK)

    cmd = [
        sys.executable, str(train_script),
        "--task", task,
        "--num_envs", str(NUM_ENVS),
        "--max_iterations", str(MAX_ITERATIONS),
        "--seed", str(seed),
        "--headless",
        "--disable_ros2_tracking_tune",
        "--run_name", run_name,
    ]

    # PPO CLI 覆盖
    if exp.get("learning_rate") is not None:
        cmd += ["--learning_rate", str(exp["learning_rate"])]
    if exp.get("clip_param") is not None:
        cmd += ["--clip_param", str(exp["clip_param"])]
    if exp.get("entropy_coef") is not None:
        cmd += ["--entropy_coef", str(exp["entropy_coef"])]

    return cmd


def _find_run_dir(logs_dir: Path, run_name: str) -> Optional[Path]:
    """查找包含 run_name 的最新训练目录"""
    if not logs_dir.exists():
        return None
    dirs = sorted(
        [d for d in logs_dir.iterdir() if d.is_dir() and run_name in d.name],
        key=lambda x: x.stat().st_mtime, reverse=True,
    )
    return dirs[0] if dirs else None


def _verify_checkpoint(run_dir: Path, exp: dict, log: DualLogger) -> bool:
    """验证 checkpoint 存在 + agent.yaml PPO 参数正确"""
    ckpt = run_dir / "model_1499.pt"
    if not ckpt.exists():
        log.warn(f"  model_1499.pt 不存在: {run_dir}")
        return False

    agent_yaml = run_dir / "params" / "agent.yaml"
    if not agent_yaml.exists():
        log.warn(f"  agent.yaml 不存在")
        return True  # checkpoint 在就行

    try:
        content = agent_yaml.read_text(encoding="utf-8")
        ok = True
        if exp.get("learning_rate") is not None:
            if f"learning_rate: {exp['learning_rate']}" not in content:
                log.warn(f"  agent.yaml lr 不匹配，预期 {exp['learning_rate']}")
                ok = False
        if exp.get("clip_param") is not None:
            if f"clip_param: {exp['clip_param']}" not in content:
                log.warn(f"  agent.yaml clip 不匹配，预期 {exp['clip_param']}")
                ok = False
        if exp.get("entropy_coef") is not None:
            if f"entropy_coef: {exp['entropy_coef']}" not in content:
                log.warn(f"  agent.yaml ent 不匹配，预期 {exp['entropy_coef']}")
                ok = False
        if ok:
            log.info(f"  agent.yaml PPO 参数核验 ✓")
        return ok
    except Exception as e:
        log.warn(f"  agent.yaml 读取失败: {e}")
        return True


def _check_divergence(run_dir: Path, log: DualLogger) -> bool:
    """检查训练是否发散（简易判定：检查最后保存的 reward）"""
    # 尝试读取 rsl_rl 的 log.json（如果存在）
    for name in ["log.json", "progress.json"]:
        log_file = run_dir / name
        if log_file.exists():
            try:
                with open(log_file, "r") as f:
                    data = json.load(f)
                # 取最后几条的平均 reward
                if isinstance(data, list) and len(data) > 0:
                    last_rewards = [
                        d.get("mean_reward", d.get("episode_reward", 0))
                        for d in data[-10:]
                    ]
                    avg = sum(last_rewards) / max(len(last_rewards), 1)
                    threshold = BASELINE_REWARD_ESTIMATE * DIVERGENCE_THRESHOLD
                    if avg < threshold:
                        log.warn(f"  ⚠ 发散疑似: 最终 reward={avg:.2f} < {threshold:.2f}")
                        return True
                    else:
                        log.info(f"  最终 reward={avg:.2f} (> {threshold:.2f}) ✓")
            except Exception:
                pass
    return False


def run_single_training(
    exp: dict,
    seed: int,
    project_root: Path,
    log: DualLogger,
    ros2_mgr: Optional[Ros2PublisherManager],
    dry_run: bool = False,
) -> RunResult:
    """执行单次训练，含重试 + watchdog + checkpoint 验证"""
    run_name = f"{exp['name']}_seed{seed}"
    logs_dir = project_root / "logs/rsl_rl/unitree_go1_rough"
    log_out_dir = project_root / "logs/sweep/phase4_train"
    log_out_dir.mkdir(parents=True, exist_ok=True)

    cmd = _build_train_cmd(project_root, exp, seed, run_name)

    log.info(f"\n{'='*60}")
    log.info(f"实验: {run_name}")
    log.info(f"命令: {' '.join(cmd)}")
    log.info(f"{'='*60}")

    if dry_run:
        log.info("[DRY-RUN] 跳过实际训练")
        return RunResult(
            exp_name=exp["name"], seed=seed, run_name=run_name,
            passed=True, return_code=0, attempts=0, error="dry-run",
        )

    # 训练前确认 ROS2 publisher 存活
    if ros2_mgr and not ros2_mgr.ensure_alive():
        log.error("ROS2 publisher 不可用，跳过此训练")
        return RunResult(
            exp_name=exp["name"], seed=seed, run_name=run_name,
            passed=False, return_code=-1, error="ROS2 publisher down",
        )

    # 重试循环
    for attempt in range(1, MAX_RETRY + 1):
        stdout_log = log_out_dir / f"{run_name}_attempt{attempt}_stdout.log"
        stderr_log = log_out_dir / f"{run_name}_attempt{attempt}_stderr.log"

        log.info(f"  尝试 {attempt}/{MAX_RETRY}...")
        start_time = time.time()

        try:
            with open(stdout_log, "w", encoding="utf-8") as f_out, \
                 open(stderr_log, "w", encoding="utf-8") as f_err:

                proc = subprocess.Popen(
                    cmd, stdout=f_out, stderr=f_err,
                    cwd=str(project_root),
                )

                # Watchdog 循环：每 WATCHDOG_INTERVAL_S 秒检查 ROS2
                while proc.poll() is None:
                    try:
                        proc.wait(timeout=WATCHDOG_INTERVAL_S)
                    except subprocess.TimeoutExpired:
                        # 训练仍在运行 — 检查 ROS2 publisher
                        if ros2_mgr:
                            if not ros2_mgr.ensure_alive():
                                log.error("  ROS2 publisher 挂掉且重启失败，终止训练")
                                proc.kill()
                                proc.wait(timeout=10)
                                return RunResult(
                                    exp_name=exp["name"], seed=seed,
                                    run_name=run_name, passed=False,
                                    return_code=-1, attempts=attempt,
                                    error="ROS2 publisher died during training",
                                )
                            else:
                                elapsed = (time.time() - start_time) / 3600
                                log.info(f"  Watchdog: ROS2 ✓, 已运行 {elapsed:.1f}h")

                duration = time.time() - start_time
                rc = proc.returncode

            if rc == 0:
                # 成功 — 验证 checkpoint
                run_dir = _find_run_dir(logs_dir, run_name)
                if run_dir is None:
                    log.error(f"  训练返回 0 但未找到 run_dir: {run_name}")
                    return RunResult(
                        exp_name=exp["name"], seed=seed, run_name=run_name,
                        passed=False, return_code=0, attempts=attempt,
                        duration_s=duration, error="run_dir not found",
                    )

                _verify_checkpoint(run_dir, exp, log)
                diverged = _check_divergence(run_dir, log)
                ckpt = run_dir / "model_1499.pt"

                log.info(f"  ✅ 成功: {run_name} ({duration/3600:.2f}h)")
                return RunResult(
                    exp_name=exp["name"], seed=seed, run_name=run_name,
                    passed=True, return_code=0, attempts=attempt,
                    checkpoint_path=str(ckpt) if ckpt.exists() else None,
                    run_dir=str(run_dir), duration_s=duration,
                    diverged=diverged,
                )
            else:
                # 失败 — 检查是否为瞬态错误
                if _is_transient_error(stderr_log) and attempt < MAX_RETRY:
                    log.warn(f"  瞬态错误 (attempt {attempt}), stderr → {stderr_log}")
                    log.warn(f"  等待 10s 后重试...")
                    time.sleep(10)
                    continue
                else:
                    # 非瞬态错误或重试用尽
                    err_tail = ""
                    try:
                        lines = stderr_log.read_text(encoding="utf-8", errors="replace").splitlines()
                        err_tail = "\n".join(lines[-5:])
                    except Exception:
                        pass
                    log.error(f"  ❌ 失败: rc={rc}, stderr 尾部:")
                    log.error(f"  {err_tail}")
                    return RunResult(
                        exp_name=exp["name"], seed=seed, run_name=run_name,
                        passed=False, return_code=rc, attempts=attempt,
                        duration_s=duration, error=f"exit code {rc}",
                    )

        except KeyboardInterrupt:
            log.warn(f"  用户中断: {run_name}")
            try:
                proc.kill()
                proc.wait(timeout=5)
            except Exception:
                pass
            raise

        except Exception as e:
            log.error(f"  异常: {e}")
            return RunResult(
                exp_name=exp["name"], seed=seed, run_name=run_name,
                passed=False, return_code=-1, attempts=attempt,
                duration_s=time.time() - start_time, error=str(e),
            )

    # 所有重试用尽
    log.error(f"  ❌ {run_name} 在 {MAX_RETRY} 次重试后仍失败")
    return RunResult(
        exp_name=exp["name"], seed=seed, run_name=run_name,
        passed=False, return_code=-1, attempts=MAX_RETRY,
        error=f"all {MAX_RETRY} retries exhausted",
    )


# ── 汇总 ─────────────────────────────────────────────────────────────────

def print_summary(results: list[RunResult], log: DualLogger):
    """打印并保存汇总结果"""
    log.info(f"\n{'='*70}")
    log.info("SWEEP SUMMARY")
    log.info(f"{'='*70}")
    log.info(f"| {'Experiment':<25} | {'Seed':>4} | {'RC':>3} | {'Time':>7} | {'Div':>3} | {'Result':<8} |")
    log.info(f"|{'-'*27}|{'-'*6}|{'-'*5}|{'-'*9}|{'-'*5}|{'-'*10}|")

    for r in results:
        t = f"{r.duration_s/3600:.1f}h" if r.duration_s > 0 else "—"
        div = "⚠" if r.diverged else ""
        status = "✅ PASS" if r.passed else "❌ FAIL"
        log.info(f"| {r.run_name:<25} | {r.seed:>4} | {r.return_code:>3} | {t:>7} | {div:>3} | {status:<8} |")

    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    diverged = sum(1 for r in results if r.diverged)
    total_h = sum(r.duration_s for r in results) / 3600

    log.info(f"\n通过: {passed}/{len(results)}, 失败: {failed}, 发散疑似: {diverged}")
    log.info(f"总耗时: {total_h:.1f}h")

    if failed > 0:
        log.error("失败列表:")
        for r in results:
            if not r.passed:
                log.error(f"  {r.run_name}: {r.error}")

    # 保存 JSON
    summary = {
        "timestamp": datetime.now().isoformat(),
        "passed": passed, "failed": failed, "diverged": diverged,
        "total": len(results), "total_hours": round(total_h, 2),
        "results": [asdict(r) for r in results],
    }
    json_path = log.log_path.parent / "phase4_train_sweep_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log.info(f"JSON summary: {json_path}")


# ── 前置检查 ─────────────────────────────────────────────────────────────

def check_prerequisites(project_root: Path) -> bool:
    """检查前置条件"""
    ok = True
    train_script = project_root / "scripts/go1-ros2-test/train.py"
    if not train_script.exists():
        print(f"[ERROR] 训练脚本不存在: {train_script}")
        ok = False

    ros2_script = project_root / "scripts/baseline-repro/Phase1-Baseline/run_ros2_cmd.sh"
    if not ros2_script.exists():
        print(f"[ERROR] ROS2 脚本不存在: {ros2_script}")
        ok = False

    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if "env_isaaclab" not in conda_prefix:
        print(f"[WARN] 当前 conda 环境: {conda_prefix}，建议 conda activate env_isaaclab")

    # 磁盘空间（30 次训练 × ~200MB = ~6GB 最低需求）
    logs_dir = project_root / "logs"
    if logs_dir.exists():
        free_gb = shutil.disk_usage(logs_dir).free / (1024 ** 3)
        if free_gb < 10:
            print(f"[WARN] 磁盘剩余 {free_gb:.1f}GB，建议 >10GB")
        else:
            print(f"[CHECK] 磁盘空间: {free_gb:.1f}GB ✓")

    return ok


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 4 Step 4.4b: 批量训练 Sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode", choices=["ppo", "dr", "combo", "all"], required=True,
        help="训练模式: ppo(6×3=18), dr(3×3=9), combo(1×3=3), all(ppo+dr)",
    )
    parser.add_argument("--skip-ros2", action="store_true",
                        help="跳过 ROS2 publisher 管理")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印命令，不实际训练")
    parser.add_argument("--resume", action="store_true",
                        help="断点续训：跳过已完成的实验")
    parser.add_argument("--project-root", type=Path, default=None,
                        help="项目根目录 (默认自动检测)")
    args = parser.parse_args()

    # 确定项目根目录
    if args.project_root:
        project_root = args.project_root
    else:
        script_path = Path(__file__).resolve()
        project_root = script_path.parents[2]  # scripts/phase4-ppo-dr/ → root

    if not (project_root / "CLAUDE.md").exists():
        print(f"[ERROR] 项目根目录无效: {project_root}")
        sys.exit(1)

    # 前置检查
    if not check_prerequisites(project_root):
        print("[ERROR] 前置检查失败")
        sys.exit(1)

    # 选择实验集
    experiments: list[dict] = []
    if args.mode == "ppo":
        experiments = PPO_EXPERIMENTS
    elif args.mode == "dr":
        experiments = DR_EXPERIMENTS
    elif args.mode == "combo":
        experiments = COMBO_EXPERIMENTS
    elif args.mode == "all":
        experiments = PPO_EXPERIMENTS + DR_EXPERIMENTS

    total_runs = len(experiments) * len(SEEDS)
    print(f"\n[INFO] 模式: {args.mode}")
    print(f"[INFO] 实验组数: {len(experiments)}, 种子数: {len(SEEDS)}, 总计: {total_runs} 次训练")

    # 初始化日志
    log_dir = project_root / "logs" / "sweep" / "phase4_train"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"phase4_train_{args.mode}_{timestamp}.log"
    log = DualLogger(log_path)
    log.info(f"Project root: {project_root}")
    log.info(f"Mode: {args.mode}, Experiments: {[e['name'] for e in experiments]}")
    log.info(f"Seeds: {SEEDS}")

    # 会话状态（断点续训）
    session_path = log_dir / f"session_{args.mode}.json"
    session = SessionState.load(session_path) if args.resume else SessionState(
        start_time=datetime.now().isoformat()
    )

    # ROS2 Publisher
    ros2_mgr: Optional[Ros2PublisherManager] = None
    if not args.skip_ros2 and not args.dry_run:
        ros2_mgr = Ros2PublisherManager(project_root, log)
        if not ros2_mgr.start():
            log.error("ROS2 publisher 启动失败，退出")
            log.close()
            sys.exit(1)

    # 执行训练
    results: list[RunResult] = []
    interrupted = False

    try:
        run_idx = 0
        for exp in experiments:
            for seed in SEEDS:
                run_idx += 1
                run_name = f"{exp['name']}_seed{seed}"

                # 断点续训：跳过已完成
                if args.resume and session.is_done(run_name):
                    log.info(f"\n[SKIP] {run_name} 已完成，跳过 ({run_idx}/{total_runs})")
                    continue

                log.info(f"\n[PROGRESS] {run_idx}/{total_runs}")

                result = run_single_training(
                    exp, seed, project_root, log, ros2_mgr, dry_run=args.dry_run,
                )
                results.append(result)

                # 更新会话状态
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

    # 汇总
    if results:
        print_summary(results, log)

    log.close()
    print(f"\n[INFO] 完整日志: {log_path}")
    if args.resume:
        print(f"[INFO] 会话状态: {session_path}")

    # 退出码
    if interrupted:
        sys.exit(130)
    all_passed = all(r.passed for r in results) if results else False
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()