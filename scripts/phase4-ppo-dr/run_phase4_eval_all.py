#!/usr/bin/env python3
"""
Phase 4 Step 4.4b: 批量评估脚本

自动化执行 Phase 4 全部模型评估，支持：
- 标准环境评估（Baseline + PPO + DR + Combo = 31-33 次）
- 抗扰动交叉评估（4 个模型 × DR-Push-Play 环境 = 4 次）

功能特性：
- ROS2 Publisher 生命周期管理
- 自动发现训练目录中的 checkpoint
- Isaac Sim 瞬态错误自动重试（最多 3 次）
- 结构化 JSON 输出到 logs/eval/phase4_formal/
- 会话断点续评

使用方法:
    conda activate env_isaaclab

    # 标准评估（全部模型）
    python scripts/phase4-ppo-dr/run_phase4_eval_all.py --mode standard

    # 抗扰动交叉评估
    python scripts/phase4-ppo-dr/run_phase4_eval_all.py --mode cross

    # 全部
    python scripts/phase4-ppo-dr/run_phase4_eval_all.py --mode all

    # 跳过 ROS2 / Dry run / 断点续评
    python scripts/phase4-ppo-dr/run_phase4_eval_all.py --mode all --skip-ros2
    python scripts/phase4-ppo-dr/run_phase4_eval_all.py --mode all --dry-run
    python scripts/phase4-ppo-dr/run_phase4_eval_all.py --mode all --resume
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional


# ── 常量 ─────────────────────────────────────────────────────────────────

SEEDS = [42, 43, 44]
BASELINE_SEEDS = [42, 43, 44]
MAX_RETRY = 3

# 评估参数（与 phase4_plan.md 步骤 4.7 一致）
EVAL_TASK_STANDARD = "Isaac-Velocity-Rough-Unitree-Go1-ROS2Cmd-Play-v0"
EVAL_TASK_CROSS = "Isaac-Velocity-Rough-Unitree-Go1-ROS2Cmd-DRPush-Play-v0"
NUM_ENVS_EVAL = 64
EVAL_STEPS = 3000
WARMUP_STEPS = 300
TARGET_VX = 1.0
PASS_ABS_ERR = 0.25
PASS_STABLE_RATIO = 0.7
STABLE_ERR_THRESH = 0.2

TRANSIENT_ERROR_SIGNATURES = [
    "PytorchStreamReader failed reading zip archive",
    "failed finding central directory",
    "CUDA out of memory",
    "cuDNN error",
]

# 实验名 → run_name 前缀映射（用于自动发现 checkpoint）
PPO_EXPERIMENTS = [
    "ppo_lr_low", "ppo_lr_high",
    "ppo_clip_low", "ppo_clip_high",
    "ppo_ent_low", "ppo_ent_high",
]
DR_EXPERIMENTS = ["dr_friction", "dr_mass", "dr_push"]
COMBO_EXPERIMENTS = ["combo_best"]

# Baseline checkpoint 路径模板（已知的固定目录名）
BASELINE_CHECKPOINTS = {
    42: "2026-03-08_16-46-27_baseline_rough_ros2cmd",
    43: "2026-03-10_10-59-57_baseline_rough_seed43",
    44: "2026-03-10_13-02-19_baseline_rough_seed44",
}


# ── 数据类 ───────────────────────────────────────────────────────────────

@dataclass
class EvalJob:
    """单次评估任务"""
    name: str              # e.g. "ppo_lr_low_seed42"
    category: str          # "baseline" | "ppo" | "dr" | "combo" | "cross"
    eval_task: str         # 评估环境 task ID
    load_run: str          # checkpoint 目录路径
    checkpoint: str        # "model_1499.pt"
    seed: int              # 评估用 seed
    output_json: str       # 输出 JSON 路径


@dataclass
class EvalResult:
    """单次评估结果"""
    name: str
    category: str
    passed: bool
    return_code: int = -1
    duration_s: float = 0.0
    attempts: int = 0
    output_json: Optional[str] = None
    error: Optional[str] = None


@dataclass
class SessionState:
    """会话状态，支持断点续评"""
    completed: list = field(default_factory=list)
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

    def is_done(self, name: str) -> bool:
        return name in self.completed


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


# ── ROS2 Publisher 管理（简化版，评估时间短） ──────────────────────────────

class Ros2PublisherManager:
    """WSL ROS2 Publisher 生命周期管理"""

    def __init__(self, project_root: Path, log: DualLogger):
        self.project_root = project_root
        self.log = log
        self.wsl_process: Optional[subprocess.Popen] = None
        self._ros2_script = project_root / "scripts/baseline-repro/Phase1-Baseline/run_ros2_cmd.sh"

    def start(self) -> bool:
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
            self.log.info("ROS2 publisher 已启动 OK")
            return True
        except Exception as e:
            self.log.error(f"启动失败: {e}")
            return False

    def stop(self):
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

    def ensure_alive(self) -> bool:
        if self._is_alive():
            return True
        self.log.warn("ROS2 publisher 已挂掉，尝试重启...")
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



# ── 评估任务发现 ─────────────────────────────────────────────────────────

def _find_run_dir(logs_dir: Path, run_name_pattern: str) -> Optional[Path]:
    """查找包含 pattern 的最新训练目录"""
    if not logs_dir.exists():
        return None
    dirs = sorted(
        [d for d in logs_dir.iterdir()
         if d.is_dir() and run_name_pattern in d.name
         and (d / "model_1499.pt").exists()],
        key=lambda x: x.stat().st_mtime, reverse=True,
    )
    return dirs[0] if dirs else None


def _extract_seed_from_stem(stem: str) -> Optional[int]:
    """从类似 dr_mass_seed43 / baseline_seed42 的文件名中提取 seed。"""
    if "seed" not in stem:
        return None
    suffix = stem.rsplit("seed", 1)[-1]
    return int(suffix) if suffix.isdigit() else None


def _load_json(path: Path) -> Optional[dict]:
    """安全读取 JSON 文件。"""
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def _pick_best_standard_seed(
    eval_out_base: Path,
    category: str,
    model_name: str,
) -> Optional[dict]:
    """从 standard 评估 JSON 中挑选 best seed。

    排序规则：
    1. pass=True 优先
    2. mean_vx_abs_err 越低越好
    3. stable_ratio 越高越好
    4. mean_base_contact_rate 越低越好
    5. seed 越小越优先（仅用于稳定 tie-break）
    """
    category_dir = eval_out_base / category
    if not category_dir.exists():
        return None

    pattern = "baseline_seed*.json" if model_name == "baseline" else f"{model_name}_seed*.json"
    candidates: list[dict] = []

    for json_path in sorted(category_dir.glob(pattern)):
        data = _load_json(json_path)
        if not data:
            continue

        seed = _extract_seed_from_stem(json_path.stem)
        if seed is None:
            continue

        mean_vx_abs_err = data.get("mean_vx_abs_err")
        stable_ratio = data.get("stable_ratio")
        if mean_vx_abs_err is None or stable_ratio is None:
            continue

        candidates.append({
            "seed": seed,
            "json_path": json_path,
            "pass": bool(data.get("pass", False)),
            "mean_vx_abs_err": float(mean_vx_abs_err),
            "stable_ratio": float(stable_ratio),
            "mean_base_contact_rate": float(data.get("mean_base_contact_rate", float("inf"))),
        })

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: (
            not x["pass"],
            x["mean_vx_abs_err"],
            -x["stable_ratio"],
            x["mean_base_contact_rate"],
            x["seed"],
        )
    )
    return candidates[0]


def discover_eval_jobs(
    project_root: Path, mode: str, log: DualLogger,
) -> list[EvalJob]:
    """自动发现 checkpoint 并构建评估任务列表"""
    logs_dir = project_root / "logs/rsl_rl/unitree_go1_rough"
    eval_out_base = project_root / "logs/eval/phase4_formal"
    jobs: list[EvalJob] = []

    # ── 标准评估 ──
    if mode in ("standard", "all"):
        # Baseline（3 种子，已知目录名）
        for seed, dir_name in BASELINE_CHECKPOINTS.items():
            run_dir = logs_dir / dir_name
            if run_dir.exists() and (run_dir / "model_1499.pt").exists():
                jobs.append(EvalJob(
                    name=f"baseline_seed{seed}", category="baseline",
                    eval_task=EVAL_TASK_STANDARD,
                    load_run=str(run_dir), checkpoint="model_1499.pt",
                    seed=42,  # 评估统一用 seed=42
                    output_json=str(eval_out_base / "baseline" / f"baseline_seed{seed}.json"),
                ))
            else:
                log.warn(f"Baseline seed={seed} checkpoint 未找到: {run_dir}")

        # PPO 实验（6 组 × 3 种子）
        for exp_name in PPO_EXPERIMENTS:
            for seed in SEEDS:
                pattern = f"{exp_name}_seed{seed}"
                run_dir = _find_run_dir(logs_dir, pattern)
                if run_dir:
                    jobs.append(EvalJob(
                        name=pattern, category="ppo",
                        eval_task=EVAL_TASK_STANDARD,
                        load_run=str(run_dir), checkpoint="model_1499.pt",
                        seed=42,
                        output_json=str(eval_out_base / "ppo" / f"{pattern}.json"),
                    ))
                else:
                    log.warn(f"PPO checkpoint 未找到: {pattern}")

        # DR 实验（3 组 × 3 种子）
        for exp_name in DR_EXPERIMENTS:
            for seed in SEEDS:
                pattern = f"{exp_name}_seed{seed}"
                run_dir = _find_run_dir(logs_dir, pattern)
                if run_dir:
                    jobs.append(EvalJob(
                        name=pattern, category="dr",
                        eval_task=EVAL_TASK_STANDARD,
                        load_run=str(run_dir), checkpoint="model_1499.pt",
                        seed=42,
                        output_json=str(eval_out_base / "dr" / f"{pattern}.json"),
                    ))
                else:
                    log.warn(f"DR checkpoint 未找到: {pattern}")

        # Combo 实验
        for exp_name in COMBO_EXPERIMENTS:
            for seed in SEEDS:
                pattern = f"{exp_name}_seed{seed}"
                run_dir = _find_run_dir(logs_dir, pattern)
                if run_dir:
                    jobs.append(EvalJob(
                        name=pattern, category="combo",
                        eval_task=EVAL_TASK_STANDARD,
                        load_run=str(run_dir), checkpoint="model_1499.pt",
                        seed=42,
                        output_json=str(eval_out_base / "combo" / f"{pattern}.json"),
                    ))

    # ── 抗扰动交叉评估 ──
    if mode in ("cross", "all"):
        # 选取 baseline + 3 DR 变体在 standard 评估中的 best seed。
        # 若 standard JSON 尚不存在，则保守回退到 seed42。
        cross_models: list[tuple[str, str]] = []

        for model_name in ["baseline"] + DR_EXPERIMENTS:
            category = "baseline" if model_name == "baseline" else "dr"
            best = _pick_best_standard_seed(eval_out_base, category, model_name)

            if best is not None:
                selected_seed = best["seed"]
                log.info(
                    f"Cross-eval best seed: {model_name}=seed{selected_seed} "
                    f"(pass={best['pass']}, abs_err={best['mean_vx_abs_err']:.4f}, "
                    f"stable_ratio={best['stable_ratio']:.4f})"
                )
            else:
                selected_seed = 42
                log.warn(f"Cross-eval 未找到 {model_name} 的 standard 评估 JSON，回退到 seed42")

            if model_name == "baseline":
                dir_name = BASELINE_CHECKPOINTS.get(selected_seed, "")
                full_path = str(logs_dir / dir_name) if dir_name else ""
            else:
                pattern = f"{model_name}_seed{selected_seed}"
                run_dir = _find_run_dir(logs_dir, pattern)
                full_path = str(run_dir) if run_dir else ""

            if Path(full_path).exists() and (Path(full_path) / "model_1499.pt").exists():
                cross_models.append((model_name, full_path))
                jobs.append(EvalJob(
                    name=f"cross_{model_name}", category="cross",
                    eval_task=EVAL_TASK_CROSS,
                    load_run=full_path, checkpoint="model_1499.pt",
                    seed=42,
                    output_json=str(eval_out_base / "cross" / f"cross_{model_name}.json"),
                ))
            else:
                log.warn(f"Cross-eval checkpoint 无效: {model_name} → {full_path}")

    log.info(f"发现 {len(jobs)} 个评估任务")
    return jobs


def _is_transient_error(stderr_path: Path) -> bool:
    """检查是否为已知瞬态错误"""
    if not stderr_path.exists():
        return False
    try:
        content = stderr_path.read_text(encoding="utf-8", errors="replace")
        return any(sig in content for sig in TRANSIENT_ERROR_SIGNATURES)
    except Exception:
        return False


# ── 评估执行 ─────────────────────────────────────────────────────────────

def run_single_eval(
    job: EvalJob,
    project_root: Path,
    log: DualLogger,
    ros2_mgr: Optional[Ros2PublisherManager],
    dry_run: bool = False,
) -> EvalResult:
    """执行单次评估，含重试"""
    eval_script = project_root / "scripts/go1-ros2-test/eval.py"
    log_out_dir = project_root / "logs/sweep/phase4_eval"
    log_out_dir.mkdir(parents=True, exist_ok=True)

    # 确保输出目录存在
    Path(job.output_json).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(eval_script),
        "--task", job.eval_task,
        "--num_envs", str(NUM_ENVS_EVAL),
        "--eval_steps", str(EVAL_STEPS),
        "--warmup_steps", str(WARMUP_STEPS),
        "--target_vx", str(TARGET_VX),
        "--pass_abs_err", str(PASS_ABS_ERR),
        "--pass_stable_ratio", str(PASS_STABLE_RATIO),
        "--stable_err_thresh", str(STABLE_ERR_THRESH),
        "--seed", str(job.seed),
        "--load_run", job.load_run,
        "--checkpoint", job.checkpoint,
        "--summary_json", job.output_json,
        "--headless",
    ]

    log.info(f"\n{'─'*50}")
    log.info(f"评估: {job.name} [{job.category}]")
    log.info(f"环境: {job.eval_task}")
    log.info(f"模型: {job.load_run}")
    log.info(f"输出: {job.output_json}")

    if dry_run:
        log.info("[DRY-RUN] 跳过")
        log.info(f"  CMD: {' '.join(cmd)}")
        return EvalResult(name=job.name, category=job.category, passed=True, error="dry-run")

    # 训练前确认 ROS2
    if ros2_mgr and not ros2_mgr.ensure_alive():
        log.error("ROS2 publisher 不可用，跳过")
        return EvalResult(name=job.name, category=job.category, passed=False, error="ROS2 down")

    for attempt in range(1, MAX_RETRY + 1):
        stdout_log = log_out_dir / f"{job.name}_attempt{attempt}_stdout.log"
        stderr_log = log_out_dir / f"{job.name}_attempt{attempt}_stderr.log"

        log.info(f"  尝试 {attempt}/{MAX_RETRY}...")
        start = time.time()

        try:
            with open(stdout_log, "w", encoding="utf-8") as fo, \
                 open(stderr_log, "w", encoding="utf-8") as fe:
                proc = subprocess.Popen(cmd, stdout=fo, stderr=fe, cwd=str(project_root))
                proc.wait()

            duration = time.time() - start
            rc = proc.returncode

            if rc == 0:
                # 验证输出 JSON 存在
                if Path(job.output_json).exists():
                    log.info(f"  [PASS] {job.name} ({duration:.0f}s)")
                else:
                    log.warn(f"  评估返回 0 但 JSON 未生成: {job.output_json}")
                return EvalResult(
                    name=job.name, category=job.category, passed=(rc == 0),
                    return_code=rc, duration_s=duration, attempts=attempt,
                    output_json=job.output_json if Path(job.output_json).exists() else None,
                )
            else:
                if _is_transient_error(stderr_log) and attempt < MAX_RETRY:
                    log.warn(f"  瞬态错误 (attempt {attempt}), 等待 10s 重试...")
                    time.sleep(10)
                    continue
                else:
                    err_tail = ""
                    try:
                        lines = stderr_log.read_text(encoding="utf-8", errors="replace").splitlines()
                        err_tail = "\n".join(lines[-5:])
                    except Exception:
                        pass
                    log.error(f"  [FAIL] rc={rc}")
                    log.error(f"  {err_tail}")
                    return EvalResult(
                        name=job.name, category=job.category, passed=False,
                        return_code=rc, duration_s=duration, attempts=attempt,
                        error=f"exit code {rc}",
                    )

        except KeyboardInterrupt:
            log.warn(f"  用户中断: {job.name}")
            try:
                proc.kill()
                proc.wait(timeout=5)
            except Exception:
                pass
            raise
        except Exception as e:
            log.error(f"  异常: {e}")
            return EvalResult(
                name=job.name, category=job.category, passed=False,
                duration_s=time.time() - start, attempts=attempt, error=str(e),
            )

    log.error(f"  [FAIL] {job.name} 在 {MAX_RETRY} 次重试后仍失败")
    return EvalResult(
        name=job.name, category=job.category, passed=False,
        attempts=MAX_RETRY, error=f"all {MAX_RETRY} retries exhausted",
    )


# ── 汇总 ─────────────────────────────────────────────────────────────────

def print_summary(results: list[EvalResult], log: DualLogger):
    """打印并保存汇总"""
    log.info(f"\n{'='*70}")
    log.info("EVAL SUMMARY")
    log.info(f"{'='*70}")
    log.info(f"| {'Name':<28} | {'Cat':<8} | {'RC':>3} | {'Time':>6} | {'Result':<8} |")
    log.info(f"|{'-'*30}|{'-'*10}|{'-'*5}|{'-'*8}|{'-'*10}|")

    for r in results:
        t = f"{r.duration_s:.0f}s" if r.duration_s > 0 else "—"
        status = "PASS" if r.passed else "FAIL"
        log.info(f"| {r.name:<28} | {r.category:<8} | {r.return_code:>3} | {t:>6} | {status:<8} |")

    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    total_min = sum(r.duration_s for r in results) / 60

    log.info(f"\n通过: {passed}/{len(results)}, 失败: {failed}")
    log.info(f"总耗时: {total_min:.1f}min")

    if failed > 0:
        log.error("失败列表:")
        for r in results:
            if not r.passed:
                log.error(f"  {r.name}: {r.error}")

    # 按类别统计
    categories = {}
    for r in results:
        cat = r.category
        if cat not in categories:
            categories[cat] = {"passed": 0, "total": 0}
        categories[cat]["total"] += 1
        if r.passed:
            categories[cat]["passed"] += 1
    log.info("\n按类别:")
    for cat, stats in categories.items():
        log.info(f"  {cat}: {stats['passed']}/{stats['total']}")

    # 保存 JSON
    summary = {
        "timestamp": datetime.now().isoformat(),
        "passed": passed, "failed": failed,
        "total": len(results), "total_minutes": round(total_min, 2),
        "by_category": categories,
        "results": [asdict(r) for r in results],
    }
    json_path = log.log_path.parent / "phase4_eval_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log.info(f"JSON summary: {json_path}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 4 Step 4.4b: 批量评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode", choices=["standard", "cross", "all"], required=True,
        help="评估模式: standard(31-33次), cross(4次抗扰动), all(全部)",
    )
    parser.add_argument("--skip-ros2", action="store_true",
                        help="跳过 ROS2 publisher 管理")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印命令，不实际评估")
    parser.add_argument("--resume", action="store_true",
                        help="断点续评：跳过已完成的评估")
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

    # 初始化日志
    log_dir = project_root / "logs" / "sweep" / "phase4_eval"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"phase4_eval_{args.mode}_{timestamp}.log"
    log = DualLogger(log_path)
    log.info(f"Project root: {project_root}")
    log.info(f"Mode: {args.mode}")

    # 发现评估任务
    jobs = discover_eval_jobs(project_root, args.mode, log)
    if not jobs and not args.dry_run:
        log.error("未发现任何可评估的 checkpoint，请先完成训练")
        log.close()
        sys.exit(1)

    log.info(f"评估任务: {len(jobs)} 个")

    # 会话状态
    session_path = log_dir / f"session_eval_{args.mode}.json"
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

    # 执行评估
    results: list[EvalResult] = []
    interrupted = False

    try:
        for i, job in enumerate(jobs):
            if args.resume and session.is_done(job.name):
                log.info(f"\n[SKIP] {job.name} 已完成 ({i+1}/{len(jobs)})")
                continue

            log.info(f"\n[PROGRESS] {i+1}/{len(jobs)}")
            result = run_single_eval(job, project_root, log, ros2_mgr, args.dry_run)
            results.append(result)

            if result.passed:
                session.completed.append(job.name)
            else:
                session.failed.append(job.name)
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
    all_passed = all(r.passed for r in results) if results else False
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
