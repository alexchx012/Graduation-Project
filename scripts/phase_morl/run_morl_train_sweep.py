#!/usr/bin/env python3
"""
Phase MORL M7: 批量训练 Sweep 脚本

自动化执行 MORL 探索层训练实验，默认覆盖计划中的 10 组权重：
- 10 组 MORL 权重策略
- 默认统一使用 seed=42
- 默认使用 PPO-Clip-High（clip_param=0.3）

功能特性：
- ROS2 Publisher 生命周期管理 + 5 分钟 Watchdog
- Isaac Sim 瞬态错误自动重试（最多 3 次）
- Checkpoint / env.yaml / agent.yaml 核验
- 发散检测（mean_reward < baseline 30% 标记为发散）
- 会话断点续训（SessionState JSON）
- 每次训练独立 stdout/stderr 日志

使用方法:
    conda activate env_isaaclab

    # 运行全部 10 组探索策略
    python scripts/phase_morl/run_morl_train_sweep.py

    # 只运行部分策略
    python scripts/phase_morl/run_morl_train_sweep.py --policy-ids P1,P5,P7

    # 自定义 seed / num_envs / iterations
    python scripts/phase_morl/run_morl_train_sweep.py --seeds 42,43 --num-envs 2048 --max-iterations 300

    # 跳过 ROS2 / Dry run / Resume
    python scripts/phase_morl/run_morl_train_sweep.py --skip-ros2
    python scripts/phase_morl/run_morl_train_sweep.py --dry-run
    python scripts/phase_morl/run_morl_train_sweep.py --resume
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


SEEDS = [42]
MORL_TASK = "Isaac-Velocity-MORL-Unitree-Go1-ROS2Cmd-v0"
NUM_ENVS = 4096
MAX_ITERATIONS = 1500
DEFAULT_CLIP_PARAM = 0.3
MAX_RETRY = 3
WATCHDOG_INTERVAL_S = 300
DIVERGENCE_THRESHOLD = 0.3
BASELINE_REWARD_ESTIMATE = 15.0

PRIMARY_REWARD_NAMES = (
    "track_lin_vel_xy_exp",
    "morl_energy",
    "morl_smooth",
    "morl_stable",
)

MORL_EXPERIMENTS = [
    {"name": "morl_p1", "policy_id": "P1", "morl_weights": "0.7,0.1,0.1,0.1", "note": "速度优先"},
    {"name": "morl_p2", "policy_id": "P2", "morl_weights": "0.1,0.7,0.1,0.1", "note": "能效优先"},
    {"name": "morl_p3", "policy_id": "P3", "morl_weights": "0.1,0.1,0.7,0.1", "note": "平滑优先"},
    {"name": "morl_p4", "policy_id": "P4", "morl_weights": "0.1,0.1,0.1,0.7", "note": "稳定优先"},
    {"name": "morl_p5", "policy_id": "P5", "morl_weights": "0.4,0.3,0.2,0.1", "note": "综合均衡"},
    {"name": "morl_p6", "policy_id": "P6", "morl_weights": "0.5,0.3,0.1,0.1", "note": "速度+能效"},
    {"name": "morl_p7", "policy_id": "P7", "morl_weights": "0.3,0.3,0.2,0.2", "note": "四目标均衡"},
    {"name": "morl_p8", "policy_id": "P8", "morl_weights": "0.2,0.4,0.2,0.2", "note": "能效偏重"},
    {"name": "morl_p9", "policy_id": "P9", "morl_weights": "0.3,0.2,0.3,0.2", "note": "平滑偏重"},
    {"name": "morl_p10", "policy_id": "P10", "morl_weights": "0.2,0.2,0.2,0.4", "note": "稳定偏重"},
]


@dataclass
class RunResult:
    exp_name: str
    policy_id: str
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
    completed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
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


class DualLogger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(log_path, "a", encoding="utf-8")
        self._write(f"\n{'=' * 70}")
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


class Ros2PublisherManager:
    """WSL ROS2 Publisher lifecycle management + watchdog."""

    def __init__(self, project_root: Path, log: DualLogger, ros2_script: Optional[Path] = None):
        self.project_root = project_root
        self.log = log
        self.wsl_process: Optional[subprocess.Popen] = None
        self._ros2_script = ros2_script or (
            project_root / "scripts" / "baseline-repro" / "Phase1-Baseline" / "run_ros2_cmd.sh"
        )
        self._restart_count = 0

    def start(self) -> bool:
        if not self._ros2_script.exists():
            self.log.error(f"ROS2 脚本不存在: {self._ros2_script}")
            return False

        self._cleanup_stale()
        wsl_root = self._to_wsl_path(self.project_root)
        wsl_script = self._to_wsl_path(self._ros2_script)
        cmd = [
            "wsl",
            "-d",
            "Ubuntu-22.04",
            "bash",
            "-lc",
            f"cd '{wsl_root}' && bash '{wsl_script}'",
        ]

        self.log.info("启动 WSL ROS2 publisher (vx=1.0, 50Hz)...")
        try:
            self.wsl_process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            )
            time.sleep(5)
            if not self._is_alive():
                self.log.error("ROS2 publisher 启动后立即退出")
                return False
            self.log.info(f"ROS2 publisher 已启动 (PID: {self._get_pid()})")
            return True
        except Exception as exc:
            self.log.error(f"启动 ROS2 publisher 失败: {exc}")
            return False

    def stop(self):
        self.log.info("停止 ROS2 publisher...")
        try:
            subprocess.run(
                ["wsl", "-d", "Ubuntu-22.04", "bash", "-c", "pkill -f go1_cmd_script_node.py 2>/dev/null || true"],
                timeout=5,
                capture_output=True,
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
        if self.wsl_process is None or self.wsl_process.poll() is not None:
            return False
        return self._is_alive()

    def ensure_alive(self) -> bool:
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
            result = subprocess.run(
                ["wsl", "-d", "Ubuntu-22.04", "bash", "-c", "pgrep -f go1_cmd_script_node.py"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return bool(result.stdout.strip())
        except Exception:
            return False

    def _get_pid(self) -> str:
        try:
            result = subprocess.run(
                ["wsl", "-d", "Ubuntu-22.04", "bash", "-c", "pgrep -f go1_cmd_script_node.py | head -n 1"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.stdout.strip() or "unknown"
        except Exception:
            return "unknown"

    def _cleanup_stale(self):
        try:
            subprocess.run(
                ["wsl", "-d", "Ubuntu-22.04", "bash", "-c", "pkill -f go1_cmd_script_node.py 2>/dev/null || true"],
                timeout=5,
                capture_output=True,
            )
            time.sleep(1)
        except Exception:
            pass

    def _to_wsl_path(self, path: Path) -> str:
        path_str = str(path).replace("\\", "/")
        try:
            result = subprocess.run(
                ["wsl", "-d", "Ubuntu-22.04", "bash", "-c", f"wslpath -u '{path_str}'"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            resolved = result.stdout.strip()
            if resolved:
                return resolved
        except Exception:
            pass

        if len(path_str) > 1 and path_str[1] == ":":
            return f"/mnt/{path_str[0].lower()}{path_str[2:]}"
        return path_str


TRANSIENT_ERROR_SIGNATURES = [
    "PytorchStreamReader failed reading zip archive",
    "failed finding central directory",
    "CUDA out of memory",
    "cuDNN error",
]


def _is_transient_error(stderr_path: Path) -> bool:
    if not stderr_path.exists():
        return False
    try:
        content = stderr_path.read_text(encoding="utf-8", errors="replace")
        return any(signature in content for signature in TRANSIENT_ERROR_SIGNATURES)
    except Exception:
        return False


def _build_train_cmd(
    project_root: Path,
    exp: dict,
    seed: int,
    run_name: str,
    *,
    num_envs: int = NUM_ENVS,
    max_iterations: int = MAX_ITERATIONS,
    clip_param: float = DEFAULT_CLIP_PARAM,
) -> list[str]:
    train_script = project_root / "scripts" / "go1-ros2-test" / "train.py"
    return [
        *[
            sys.executable,
            str(train_script),
            "--task",
            exp.get("task", MORL_TASK),
            "--num_envs",
            str(num_envs),
            "--max_iterations",
            str(max_iterations),
            "--seed",
            str(seed),
            "--headless",
            "--disable_ros2_tracking_tune",
            "--clip_param",
            str(clip_param),
            "--morl_weights",
            exp["morl_weights"],
            "--run_name",
            run_name,
        ],
        *(
            ["--morl_command_profile", exp["command_profile"]]
            if exp.get("command_profile")
            else []
        ),
        *(
            ["--init_checkpoint", exp["init_checkpoint"]]
            if exp.get("init_checkpoint")
            else []
        ),
        *(
            ["--load_run", exp["load_run"]]
            if exp.get("load_run")
            else []
        ),
    ]


def _find_run_dir(logs_dir: Path, run_name: str) -> Optional[Path]:
    if not logs_dir.exists():
        return None
    dirs = sorted(
        [path for path in logs_dir.iterdir() if path.is_dir() and run_name in path.name],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return dirs[0] if dirs else None


def _extract_checkpoint_index(checkpoint_name: str) -> Optional[int]:
    match = re.fullmatch(r"model_(\d+)\.pt", checkpoint_name)
    if match is None:
        return None
    return int(match.group(1))


def _find_latest_checkpoint(run_dir: Path) -> Optional[Path]:
    candidates: list[tuple[int, Path]] = []
    for path in run_dir.glob("model_*.pt"):
        index = _extract_checkpoint_index(path.name)
        if index is not None:
            candidates.append((index, path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _resolve_expected_checkpoint(run_dir: Path, exp: dict, max_iterations: int) -> Path:
    init_checkpoint = exp.get("init_checkpoint")
    if init_checkpoint and exp.get("resume_like_init", False):
        init_name = Path(str(init_checkpoint)).name
        init_index = _extract_checkpoint_index(init_name)
        if init_index is not None:
            return run_dir / f"model_{init_index + max_iterations - 1}.pt"
    return run_dir / f"model_{max_iterations - 1}.pt"


def _verify_run_artifacts(run_dir: Path, exp: dict, max_iterations: int, log: DualLogger) -> bool:
    expected_checkpoint = _resolve_expected_checkpoint(run_dir, exp, max_iterations)
    if not expected_checkpoint.exists():
        latest_checkpoint = _find_latest_checkpoint(run_dir)
        latest_name = latest_checkpoint.name if latest_checkpoint is not None else "none"
        log.warn(
            f"  checkpoint 不存在: {expected_checkpoint.name} (latest found: {latest_name})"
        )
        return False

    params_dir = run_dir / "params"
    agent_yaml = params_dir / "agent.yaml"
    env_yaml = params_dir / "env.yaml"

    ok = True
    if not agent_yaml.exists():
        log.warn("  agent.yaml 不存在")
        ok = False
    else:
        content = agent_yaml.read_text(encoding="utf-8", errors="replace")
        expected_clip = exp.get("clip_param", DEFAULT_CLIP_PARAM)
        if f"clip_param: {expected_clip}" not in content:
            log.warn(f"  agent.yaml clip_param 不匹配，预期 {expected_clip}")
            ok = False

    if not env_yaml.exists():
        log.warn("  env.yaml 不存在")
        ok = False
    else:
        weights = [float(weight) for weight in exp["morl_weights"].split(",")]
        content = env_yaml.read_text(encoding="utf-8", errors="replace")
        for reward_name, weight in zip(PRIMARY_REWARD_NAMES, weights, strict=True):
            if not _yaml_text_has_weight(content, reward_name, weight):
                log.warn(f"  env.yaml {reward_name} 权重不匹配，预期 {weight}")
                ok = False

    if ok:
        log.info("  checkpoint / env.yaml / agent.yaml 核验 OK")
    return ok


def _yaml_text_has_weight(content: str, reward_name: str, expected_weight: float) -> bool:
    pattern = re.compile(
        rf"^\s*{re.escape(reward_name)}:\s*$.*?^\s*weight:\s*{re.escape(str(expected_weight))}\s*$",
        re.MULTILINE | re.DOTALL,
    )
    return bool(pattern.search(content))


def _check_divergence(run_dir: Path, log: DualLogger) -> bool:
    for name in ["log.json", "progress.json"]:
        log_file = run_dir / name
        if not log_file.exists():
            continue
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list) and data:
                rewards = [entry.get("mean_reward", entry.get("episode_reward", 0.0)) for entry in data[-10:]]
                avg_reward = sum(rewards) / max(len(rewards), 1)
                threshold = BASELINE_REWARD_ESTIMATE * DIVERGENCE_THRESHOLD
                if avg_reward < threshold:
                    log.warn(f"  [DIVERGED?] 最终 reward={avg_reward:.2f} < {threshold:.2f}")
                    return True
                log.info(f"  最终 reward={avg_reward:.2f} (> {threshold:.2f}) OK")
        except Exception:
            pass
    return False


def run_single_training(
    exp: dict,
    seed: int,
    project_root: Path,
    log: DualLogger,
    ros2_mgr: Optional[Ros2PublisherManager],
    *,
    dry_run: bool = False,
    num_envs: int = NUM_ENVS,
    max_iterations: int = MAX_ITERATIONS,
    clip_param: float = DEFAULT_CLIP_PARAM,
    sweep_log_subdir: str = "phase_morl_train",
) -> RunResult:
    run_name = f"{exp['name']}_seed{seed}"
    logs_dir = project_root / "logs" / "rsl_rl" / "unitree_go1_rough"
    sweep_log_dir = project_root / "logs" / "sweep" / sweep_log_subdir
    sweep_log_dir.mkdir(parents=True, exist_ok=True)

    cmd = _build_train_cmd(
        project_root,
        exp,
        seed,
        run_name,
        num_envs=num_envs,
        max_iterations=max_iterations,
        clip_param=clip_param,
    )

    log.info(f"\n{'=' * 60}")
    log.info(f"实验: {run_name} ({exp['policy_id']})")
    log.info(f"权重: {exp['morl_weights']}")
    log.info(f"命令: {' '.join(cmd)}")
    log.info(f"{'=' * 60}")

    if dry_run:
        log.info("[DRY-RUN] 跳过实际训练")
        return RunResult(
            exp_name=exp["name"],
            policy_id=exp["policy_id"],
            seed=seed,
            run_name=run_name,
            passed=True,
            return_code=0,
            attempts=0,
            error="dry-run",
        )

    if ros2_mgr and not ros2_mgr.ensure_alive():
        log.error("ROS2 publisher 不可用，跳过此训练")
        return RunResult(
            exp_name=exp["name"],
            policy_id=exp["policy_id"],
            seed=seed,
            run_name=run_name,
            passed=False,
            return_code=-1,
            error="ROS2 publisher down",
        )

    for attempt in range(1, MAX_RETRY + 1):
        stdout_log = sweep_log_dir / f"{run_name}_attempt{attempt}_stdout.log"
        stderr_log = sweep_log_dir / f"{run_name}_attempt{attempt}_stderr.log"

        log.info(f"  尝试 {attempt}/{MAX_RETRY}...")
        start_time = time.time()
        proc: Optional[subprocess.Popen] = None

        try:
            with open(stdout_log, "w", encoding="utf-8") as f_out, open(
                stderr_log, "w", encoding="utf-8"
            ) as f_err:
                proc = subprocess.Popen(
                    cmd,
                    stdout=f_out,
                    stderr=f_err,
                    cwd=str(project_root),
                )

                while proc.poll() is None:
                    try:
                        proc.wait(timeout=WATCHDOG_INTERVAL_S)
                    except subprocess.TimeoutExpired:
                        if ros2_mgr and not ros2_mgr.ensure_alive():
                            log.error("  ROS2 publisher 挂掉且重启失败，终止训练")
                            proc.kill()
                            proc.wait(timeout=10)
                            return RunResult(
                                exp_name=exp["name"],
                                policy_id=exp["policy_id"],
                                seed=seed,
                                run_name=run_name,
                                passed=False,
                                return_code=-1,
                                attempts=attempt,
                                error="ROS2 publisher died during training",
                            )
                        elapsed_h = (time.time() - start_time) / 3600
                        log.info(f"  Watchdog: ROS2 OK, 已运行 {elapsed_h:.1f}h")

            duration = time.time() - start_time
            rc = proc.returncode if proc is not None else -1

            if rc == 0:
                run_dir = _find_run_dir(logs_dir, run_name)
                if run_dir is None:
                    log.error(f"  训练返回 0 但未找到 run_dir: {run_name}")
                    return RunResult(
                        exp_name=exp["name"],
                        policy_id=exp["policy_id"],
                        seed=seed,
                        run_name=run_name,
                        passed=False,
                        return_code=0,
                        attempts=attempt,
                        duration_s=duration,
                        error="run_dir not found",
                    )

                verified = _verify_run_artifacts(run_dir, exp, max_iterations, log)
                diverged = _check_divergence(run_dir, log)
                checkpoint = _resolve_expected_checkpoint(run_dir, exp, max_iterations)

                if verified:
                    log.info(f"  [PASS] {run_name} ({duration / 3600:.2f}h)")
                else:
                    log.warn(f"  [WARN] {run_name} 训练返回 0，但产物核验不完整")

                return RunResult(
                    exp_name=exp["name"],
                    policy_id=exp["policy_id"],
                    seed=seed,
                    run_name=run_name,
                    passed=verified,
                    return_code=0,
                    checkpoint_path=str(checkpoint) if checkpoint.exists() else None,
                    run_dir=str(run_dir),
                    duration_s=duration,
                    attempts=attempt,
                    diverged=diverged,
                    error=None if verified else "artifact verification failed",
                )

            if _is_transient_error(stderr_log) and attempt < MAX_RETRY:
                log.warn(f"  瞬态错误 (attempt {attempt}), stderr → {stderr_log}")
                log.warn("  等待 10s 后重试...")
                time.sleep(10)
                continue

            err_tail = ""
            try:
                lines = stderr_log.read_text(encoding="utf-8", errors="replace").splitlines()
                err_tail = "\n".join(lines[-5:])
            except Exception:
                pass
            log.error(f"  [FAIL] rc={rc}, stderr 尾部:")
            log.error(f"  {err_tail}")
            return RunResult(
                exp_name=exp["name"],
                policy_id=exp["policy_id"],
                seed=seed,
                run_name=run_name,
                passed=False,
                return_code=rc,
                attempts=attempt,
                duration_s=duration,
                error=f"exit code {rc}",
            )

        except KeyboardInterrupt:
            log.warn(f"  用户中断: {run_name}")
            if proc is not None:
                try:
                    proc.kill()
                    proc.wait(timeout=5)
                except Exception:
                    pass
            raise
        except Exception as exc:
            log.error(f"  异常: {exc}")
            return RunResult(
                exp_name=exp["name"],
                policy_id=exp["policy_id"],
                seed=seed,
                run_name=run_name,
                passed=False,
                return_code=-1,
                attempts=attempt,
                duration_s=time.time() - start_time,
                error=str(exc),
            )

    log.error(f"  [FAIL] {run_name} 在 {MAX_RETRY} 次重试后仍失败")
    return RunResult(
        exp_name=exp["name"],
        policy_id=exp["policy_id"],
        seed=seed,
        run_name=run_name,
        passed=False,
        return_code=-1,
        attempts=MAX_RETRY,
        error=f"all {MAX_RETRY} retries exhausted",
    )


def print_summary(
    results: list[RunResult],
    log: DualLogger,
    summary_filename: str = "phase_morl_train_sweep_summary.json",
):
    log.info(f"\n{'=' * 70}")
    log.info("SWEEP SUMMARY")
    log.info(f"{'=' * 70}")
    log.info(f"| {'Experiment':<18} | {'Seed':>4} | {'RC':>3} | {'Time':>7} | {'Div':>3} | {'Result':<8} |")
    log.info(f"|{'-' * 20}|{'-' * 6}|{'-' * 5}|{'-' * 9}|{'-' * 5}|{'-' * 10}|")

    for result in results:
        duration = f"{result.duration_s / 3600:.1f}h" if result.duration_s > 0 else "—"
        div = "[D]" if result.diverged else ""
        status = "PASS" if result.passed else "FAIL"
        log.info(
            f"| {result.run_name:<18} | {result.seed:>4} | {result.return_code:>3} | {duration:>7} | {div:>3} | {status:<8} |"
        )

    passed = sum(1 for result in results if result.passed)
    failed = sum(1 for result in results if not result.passed)
    diverged = sum(1 for result in results if result.diverged)
    total_hours = sum(result.duration_s for result in results) / 3600

    log.info(f"\n通过: {passed}/{len(results)}, 失败: {failed}, 发散疑似: {diverged}")
    log.info(f"总耗时: {total_hours:.1f}h")

    if failed > 0:
        log.error("失败列表:")
        for result in results:
            if not result.passed:
                log.error(f"  {result.run_name}: {result.error}")

    summary = {
        "timestamp": datetime.now().isoformat(),
        "passed": passed,
        "failed": failed,
        "diverged": diverged,
        "total": len(results),
        "total_hours": round(total_hours, 2),
        "results": [asdict(result) for result in results],
    }
    json_path = log.log_path.parent / summary_filename
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log.info(f"JSON summary: {json_path}")


def check_prerequisites(project_root: Path, ros2_script: Optional[Path]) -> bool:
    ok = True
    train_script = project_root / "scripts" / "go1-ros2-test" / "train.py"
    if not train_script.exists():
        print(f"[ERROR] 训练脚本不存在: {train_script}")
        ok = False

    ros2_path = ros2_script or (
        project_root / "scripts" / "baseline-repro" / "Phase1-Baseline" / "run_ros2_cmd.sh"
    )
    if not ros2_path.exists():
        print(f"[ERROR] ROS2 脚本不存在: {ros2_path}")
        ok = False

    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if "env_isaaclab" not in conda_prefix:
        print(f"[WARN] 当前 conda 环境: {conda_prefix}，建议 conda activate env_isaaclab")

    logs_dir = project_root / "logs"
    if logs_dir.exists():
        free_gb = shutil.disk_usage(logs_dir).free / (1024**3)
        if free_gb < 10:
            print(f"[WARN] 磁盘剩余 {free_gb:.1f}GB，建议 >10GB")
        else:
            print(f"[CHECK] 磁盘空间: {free_gb:.1f}GB OK")

    return ok


def _select_experiments(policy_ids_raw: Optional[str]) -> list[dict]:
    if not policy_ids_raw:
        return MORL_EXPERIMENTS

    wanted = {item.strip().upper() for item in policy_ids_raw.split(",") if item.strip()}
    selected = [exp for exp in MORL_EXPERIMENTS if exp["policy_id"] in wanted]
    if len(selected) != len(wanted):
        known = {exp["policy_id"] for exp in MORL_EXPERIMENTS}
        unknown = sorted(wanted - known)
        raise ValueError(f"Unknown policy ids: {unknown}. Known: {sorted(known)}")
    return selected


def _parse_seeds(seeds_raw: Optional[str]) -> list[int]:
    if not seeds_raw:
        return SEEDS
    return [int(item.strip()) for item in seeds_raw.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Phase MORL M7: 批量训练 Sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--policy-ids",
        type=str,
        default=None,
        help="仅运行指定策略，如 P1,P5,P7（默认全部 10 组）",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="逗号分隔种子列表（默认 42）",
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

    if args.project_root:
        project_root = args.project_root
    else:
        project_root = Path(__file__).resolve().parents[2]

    if not (project_root / "CLAUDE.md").exists():
        print(f"[ERROR] 项目根目录无效: {project_root}")
        sys.exit(1)

    try:
        experiments = _select_experiments(args.policy_ids)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    seeds = _parse_seeds(args.seeds)
    total_runs = len(experiments) * len(seeds)

    if not check_prerequisites(project_root, args.ros2_script):
        print("[ERROR] 前置检查失败")
        sys.exit(1)

    print(f"\n[INFO] 策略数: {len(experiments)}, seeds: {seeds}, 总计: {total_runs} 次训练")

    log_dir = project_root / "logs" / "sweep" / "phase_morl_train"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    policy_tag = "all" if args.policy_ids is None else args.policy_ids.replace(",", "_").replace(" ", "")
    log_path = log_dir / f"phase_morl_train_{policy_tag}_{timestamp}.log"
    log = DualLogger(log_path)
    log.info(f"Project root: {project_root}")
    log.info(f"Policies: {[exp['policy_id'] for exp in experiments]}")
    log.info(f"Seeds: {seeds}")
    log.info(f"num_envs={args.num_envs}, max_iterations={args.max_iterations}, clip_param={args.clip_param}")

    session_path = log_dir / f"session_{policy_tag}.json"
    session = SessionState.load(session_path) if args.resume else SessionState(
        start_time=datetime.now().isoformat()
    )

    ros2_mgr: Optional[Ros2PublisherManager] = None
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
