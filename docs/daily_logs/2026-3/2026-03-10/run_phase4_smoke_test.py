#!/usr/bin/env python3
"""
Phase 4 Step 4.4: Smoke Test Script

验证 PPO CLI 覆盖 + DR 变体配置的正确性。
每组用 64 envs / 1 iter 快速运行，核验：
  1) 任务注册 + 训练正常启动并完成
  2) params/agent.yaml 中 PPO 参数与实验设计一致
  3) params/env.yaml 中 DR 事件配置正确
  4) 返回码 = 0

所有输出写入 logs/smoke/phase4_smoke/ 目录，每组独立日志文件。

使用方法:
    conda activate env_isaaclab
    # 完整运行（需 ROS2 publisher）
    python docs/daily_logs/2026-3/2026-03-10/run_phase4_smoke_test.py
    # 跳过 ROS2 publisher（已在其他终端启动）
    python docs/daily_logs/2026-3/2026-03-10/run_phase4_smoke_test.py --skip-ros2
    # Dry run（只打印命令）
    python docs/daily_logs/2026-3/2026-03-10/run_phase4_smoke_test.py --dry-run
    # 只跑 DR 测试
    python docs/daily_logs/2026-3/2026-03-10/run_phase4_smoke_test.py --only dr
    # 只跑 PPO 测试
    python docs/daily_logs/2026-3/2026-03-10/run_phase4_smoke_test.py --only ppo
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


# ── Test Matrix ──────────────────────────────────────────────────────────

@dataclass
class SmokeTestCase:
    """单个冒烟测试用例"""
    name: str                          # 显示名
    task_id: str                       # gym task ID
    run_name: str                      # --run_name
    extra_cli: list = field(default_factory=list)  # 额外 CLI 参数
    # 期望值（用于 agent.yaml 核验）
    expect_lr: float = 1e-3
    expect_clip: float = 0.2
    expect_ent: float = 0.01
    # DR 期望（用于 env.yaml 核验）
    expect_dr_key: Optional[str] = None  # env.yaml 中需搜索的关键字


_BASE_TASK = "Isaac-Velocity-Rough-Unitree-Go1-ROS2Cmd-v0"

PPO_TESTS = [
    SmokeTestCase("PPO-LR-Low",   _BASE_TASK, "smoke_ppo_lr_low",
                  ["--learning_rate", "5e-4"], expect_lr=5e-4),
    SmokeTestCase("PPO-LR-High",  _BASE_TASK, "smoke_ppo_lr_high",
                  ["--learning_rate", "3e-3"], expect_lr=3e-3),
    SmokeTestCase("PPO-Clip-Low", _BASE_TASK, "smoke_ppo_clip_low",
                  ["--clip_param", "0.1"], expect_clip=0.1),
    SmokeTestCase("PPO-Clip-High", _BASE_TASK, "smoke_ppo_clip_high",
                  ["--clip_param", "0.3"], expect_clip=0.3),
    SmokeTestCase("PPO-Ent-Low",  _BASE_TASK, "smoke_ppo_ent_low",
                  ["--entropy_coef", "0.005"], expect_ent=0.005),
    SmokeTestCase("PPO-Ent-High", _BASE_TASK, "smoke_ppo_ent_high",
                  ["--entropy_coef", "0.02"], expect_ent=0.02),
]

DR_TESTS = [
    SmokeTestCase("DR-Friction",
                  "Isaac-Velocity-Rough-Unitree-Go1-ROS2Cmd-DRFriction-v0",
                  "smoke_dr_friction",
                  expect_dr_key="static_friction_range"),
    SmokeTestCase("DR-Mass",
                  "Isaac-Velocity-Rough-Unitree-Go1-ROS2Cmd-DRMass-v0",
                  "smoke_dr_mass",
                  expect_dr_key="mass_distribution_params"),
    SmokeTestCase("DR-Push",
                  "Isaac-Velocity-Rough-Unitree-Go1-ROS2Cmd-DRPush-v0",
                  "smoke_dr_push",
                  expect_dr_key="push_robot"),
]


# ── Results ──────────────────────────────────────────────────────────────

@dataclass
class SmokeResult:
    """单次冒烟测试结果"""
    name: str
    passed: bool = False
    return_code: Optional[int] = None
    run_dir: Optional[str] = None
    duration_s: float = 0.0
    # 核验结果
    agent_yaml_ok: bool = False
    env_yaml_ok: bool = False
    lr_actual: Optional[float] = None
    clip_actual: Optional[float] = None
    ent_actual: Optional[float] = None
    dr_key_found: Optional[bool] = None
    error_msg: Optional[str] = None


# ── Logger ───────────────────────────────────────────────────────────────

class DualLogger:
    """同时输出到控制台和日志文件"""
    def __init__(self, log_path: Path):
        self.log_path = log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(log_path, "w", encoding="utf-8")
        self._write(f"=== Phase 4 Smoke Test Log ===\n")
        self._write(f"Started: {datetime.now().isoformat()}\n\n")

    def info(self, msg: str):
        line = f"[INFO] {msg}"
        print(line)
        self._write(line + "\n")

    def error(self, msg: str):
        line = f"[ERROR] {msg}"
        print(line, file=sys.stderr)
        self._write(line + "\n")

    def section(self, title: str):
        sep = "=" * 60
        self._write(f"\n{sep}\n{title}\n{sep}\n")
        print(f"\n{sep}\n{title}\n{sep}")

    def detail(self, msg: str):
        """只写日志文件，不打印到控制台"""
        self._write(msg + "\n")

    def _write(self, s: str):
        self._fh.write(s)
        self._fh.flush()

    def close(self):
        self._write(f"\nFinished: {datetime.now().isoformat()}\n")
        self._fh.close()




# ── YAML Verification ───────────────────────────────────────────────────

def _read_yaml_as_text(yaml_path: Path) -> str:
    """读取 YAML 文件为纯文本（避免依赖 pyyaml）"""
    if not yaml_path.exists():
        return ""
    return yaml_path.read_text(encoding="utf-8")


def _extract_yaml_value(text: str, key: str) -> Optional[str]:
    """从 YAML 文本中提取键值（简单正则匹配）"""
    import re
    pattern = rf"^\s*{re.escape(key)}\s*:\s*(.+)$"
    for line in text.splitlines():
        m = re.match(pattern, line)
        if m:
            return m.group(1).strip()
    return None


def verify_agent_yaml(run_dir: Path, tc: SmokeTestCase, log: DualLogger) -> dict:
    """核验 params/agent.yaml 中 PPO 参数"""
    result = {"ok": False, "lr": None, "clip": None, "ent": None}
    agent_yaml = run_dir / "params" / "agent.yaml"

    if not agent_yaml.exists():
        log.error(f"  agent.yaml 不存在: {agent_yaml}")
        return result

    text = _read_yaml_as_text(agent_yaml)
    log.detail("  --- agent.yaml PPO params ---")
    for line in text.splitlines():
        if any(k in line for k in ["learning_rate", "clip_param", "entropy_coef",
                                     "schedule", "desired_kl"]):
            log.detail(f"  {line}")

    lr_str = _extract_yaml_value(text, "learning_rate")
    clip_str = _extract_yaml_value(text, "clip_param")
    ent_str = _extract_yaml_value(text, "entropy_coef")

    try:
        result["lr"] = float(lr_str) if lr_str else None
        result["clip"] = float(clip_str) if clip_str else None
        result["ent"] = float(ent_str) if ent_str else None
    except (ValueError, TypeError) as e:
        log.error(f"  YAML 值解析失败: {e}")
        return result

    checks = []
    if result["lr"] is not None:
        ok = abs(result["lr"] - tc.expect_lr) / max(tc.expect_lr, 1e-10) < 0.01
        checks.append(ok)
        log.info(f"  lr: {result['lr']} (expect {tc.expect_lr}) [{'✓' if ok else '✗'}]")
    if result["clip"] is not None:
        ok = abs(result["clip"] - tc.expect_clip) / max(tc.expect_clip, 1e-10) < 0.01
        checks.append(ok)
        log.info(f"  clip: {result['clip']} (expect {tc.expect_clip}) [{'✓' if ok else '✗'}]")
    if result["ent"] is not None:
        ok = abs(result["ent"] - tc.expect_ent) / max(tc.expect_ent, 1e-10) < 0.01
        checks.append(ok)
        log.info(f"  ent: {result['ent']} (expect {tc.expect_ent}) [{'✓' if ok else '✗'}]")

    result["ok"] = all(checks) if checks else False
    return result


def verify_env_yaml(run_dir: Path, tc: SmokeTestCase, log: DualLogger) -> bool:
    """核验 params/env.yaml 中 DR 事件配置"""
    if tc.expect_dr_key is None:
        log.info("  (无 DR 核验项，跳过 env.yaml)")
        return True

    env_yaml = run_dir / "params" / "env.yaml"
    if not env_yaml.exists():
        log.error(f"  env.yaml 不存在: {env_yaml}")
        return False

    text = _read_yaml_as_text(env_yaml)
    # 记录 events 段到日志
    in_events = False
    for line in text.splitlines():
        if "events:" in line:
            in_events = True
        if in_events:
            log.detail(f"  {line}")
            if line.strip() and not line[0].isspace() and "events" not in line:
                in_events = False

    found = tc.expect_dr_key in text
    log.info(f"  DR key '{tc.expect_dr_key}' in env.yaml: [{'✓' if found else '✗'}]")
    return found


# ── Test Runner ──────────────────────────────────────────────────────────

def find_run_dir(logs_base: Path, run_name: str) -> Optional[Path]:
    """查找包含 run_name 的最新训练目录"""
    if not logs_base.exists():
        return None
    matching = sorted(
        [d for d in logs_base.iterdir() if d.is_dir() and run_name in d.name],
        key=lambda x: x.stat().st_mtime, reverse=True,
    )
    return matching[0] if matching else None


def run_single_test(
    tc: SmokeTestCase,
    project_root: Path,
    log: DualLogger,
    dry_run: bool = False,
) -> SmokeResult:
    """执行单个冒烟测试"""
    result = SmokeResult(name=tc.name)
    train_script = project_root / "scripts" / "go1-ros2-test" / "train.py"

    cmd = [
        sys.executable, str(train_script),
        "--task", tc.task_id,
        "--num_envs", "64",
        "--max_iterations", "1",
        "--headless",
        "--disable_ros2_tracking_tune",
        "--seed", "42",
        "--run_name", tc.run_name,
    ] + tc.extra_cli

    log.section(f"Test: {tc.name}")
    log.info(f"Task: {tc.task_id}")
    log.info(f"CMD: {' '.join(cmd)}")

    if dry_run:
        log.info("[DRY-RUN] 跳过实际执行")
        result.passed = True
        result.error_msg = "dry-run"
        return result

    # 将训练 stdout/stderr 写入独立日志
    test_log_dir = project_root / "logs" / "smoke" / "phase4_smoke"
    test_log_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = test_log_dir / f"{tc.run_name}_stdout.log"
    stderr_log = test_log_dir / f"{tc.run_name}_stderr.log"

    start = time.time()
    try:
        with open(stdout_log, "w", encoding="utf-8") as fout, \
             open(stderr_log, "w", encoding="utf-8") as ferr:
            proc = subprocess.run(cmd, stdout=fout, stderr=ferr, timeout=600)
        result.return_code = proc.returncode
        result.duration_s = time.time() - start

        log.info(f"Return code: {result.return_code} (耗时 {result.duration_s:.1f}s)")
        log.info(f"stdout log: {stdout_log}")
        log.info(f"stderr log: {stderr_log}")

        if result.return_code != 0:
            result.error_msg = f"exit code {result.return_code}"
            # 将 stderr 尾部写入主日志
            err_text = stderr_log.read_text(encoding="utf-8", errors="replace")
            tail = "\n".join(err_text.splitlines()[-30:])
            log.detail(f"  --- stderr tail ---\n{tail}")
            log.error(f"训练失败，详见 {stderr_log}")
            return result

    except subprocess.TimeoutExpired:
        result.duration_s = time.time() - start
        result.error_msg = "timeout (600s)"
        log.error("训练超时 (600s)")
        return result
    except Exception as e:
        result.duration_s = time.time() - start
        result.error_msg = str(e)
        log.error(f"执行异常: {e}")
        return result

    # 查找训练目录
    logs_base = project_root / "logs" / "rsl_rl" / "unitree_go1_rough"
    run_dir = find_run_dir(logs_base, tc.run_name)
    if run_dir is None:
        result.error_msg = "训练目录未找到"
        log.error(f"训练目录未找到 (搜索 '{tc.run_name}' in {logs_base})")
        return result

    result.run_dir = str(run_dir)
    log.info(f"Run dir: {run_dir.name}")

    # 核验 agent.yaml
    log.info("--- agent.yaml 核验 ---")
    agent_check = verify_agent_yaml(run_dir, tc, log)
    result.agent_yaml_ok = agent_check["ok"]
    result.lr_actual = agent_check["lr"]
    result.clip_actual = agent_check["clip"]
    result.ent_actual = agent_check["ent"]

    # 核验 env.yaml
    log.info("--- env.yaml 核验 ---")
    result.env_yaml_ok = verify_env_yaml(run_dir, tc, log)

    # 综合判定
    result.passed = (
        result.return_code == 0
        and result.agent_yaml_ok
        and result.env_yaml_ok
    )
    symbol = "✅ PASS" if result.passed else "❌ FAIL"
    log.info(f"Result: {symbol}")

    return result


# ── ROS2 Publisher (reuse from 补训 script) ──────────────────────────────

def start_ros2_publisher(project_root: Path) -> Optional[subprocess.Popen]:
    """启动 WSL ROS2 publisher，返回进程句柄"""
    ros2_script = project_root / "scripts/baseline-repro/Phase1-Baseline/run_ros2_cmd.sh"
    if not ros2_script.exists():
        print(f"[ERROR] ROS2 脚本不存在: {ros2_script}")
        return None

    # 路径转换
    path_str = str(project_root).replace('\\', '/')
    try:
        r = subprocess.run(
            ["wsl", "-d", "Ubuntu-22.04", "bash", "-c", f"wslpath -u '{path_str}'"],
            capture_output=True, text=True, timeout=5,
        )
        wsl_root = r.stdout.strip()
    except Exception:
        drive = path_str[0].lower()
        wsl_root = f"/mnt/{drive}{path_str[2:]}"

    wsl_script = f"{wsl_root}/scripts/baseline-repro/Phase1-Baseline/run_ros2_cmd.sh"
    cmd = ["wsl", "-d", "Ubuntu-22.04", "bash", "-lc",
           f"cd '{wsl_root}' && bash '{wsl_script}'"]

    print("[ROS2] 启动 WSL publisher (vx=1.0, 50Hz)...")
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
        )
        time.sleep(5)
        # 验证
        check = subprocess.run(
            ["wsl", "-d", "Ubuntu-22.04", "bash", "-c",
             "pgrep -f go1_cmd_script_node.py"],
            capture_output=True, text=True, timeout=5,
        )
        if check.stdout.strip():
            print(f"[ROS2] Publisher 已启动 (PID: {check.stdout.strip().splitlines()[0]})")
            return proc
        else:
            print("[ERROR] ROS2 publisher 启动失败")
            return None
    except Exception as e:
        print(f"[ERROR] ROS2 publisher 启动异常: {e}")
        return None


def stop_ros2_publisher(proc: Optional[subprocess.Popen]):
    """停止 WSL ROS2 publisher"""
    if proc is None:
        return
    print("[ROS2] 停止 publisher...")
    try:
        subprocess.run(
            ["wsl", "-d", "Ubuntu-22.04", "bash", "-c",
             "pkill -f go1_cmd_script_node.py 2>/dev/null || true"],
            timeout=5, capture_output=True,
        )
    except Exception:
        pass
    try:
        proc.terminate()
        proc.wait(timeout=3)
    except Exception:
        proc.kill()
    print("[ROS2] Publisher 已停止")


# ── Summary ──────────────────────────────────────────────────────────────

def print_summary(results: list, log: DualLogger):
    """打印汇总报告"""
    log.section("SUMMARY")
    passed = sum(1 for r in results if r.passed)
    total = len(results)

    header = f"| {'Test':<20} | {'Return':<6} | {'agent.yaml':<10} | {'env.yaml':<9} | {'Result':<8} |"
    sep = f"|{'-'*22}|{'-'*8}|{'-'*12}|{'-'*11}|{'-'*10}|"
    log.info(header)
    log.info(sep)

    for r in results:
        rc = str(r.return_code) if r.return_code is not None else "—"
        ay = "✓" if r.agent_yaml_ok else "✗"
        ey = "✓" if r.env_yaml_ok else "✗"
        res = "✅ PASS" if r.passed else "❌ FAIL"
        log.info(f"| {r.name:<20} | {rc:<6} | {ay:<10} | {ey:<9} | {res:<8} |")

    log.info("")
    log.info(f"Total: {passed}/{total} passed")

    if passed == total:
        log.info("🎉 All smoke tests PASSED")
    else:
        failed_names = [r.name for r in results if not r.passed]
        log.error(f"Failed tests: {', '.join(failed_names)}")
        for r in results:
            if not r.passed and r.error_msg:
                log.error(f"  {r.name}: {r.error_msg}")

    # 保存 JSON 汇总
    summary = {
        "timestamp": datetime.now().isoformat(),
        "passed": passed, "total": total,
        "results": [
            {"name": r.name, "passed": r.passed, "return_code": r.return_code,
             "agent_yaml_ok": r.agent_yaml_ok, "env_yaml_ok": r.env_yaml_ok,
             "lr": r.lr_actual, "clip": r.clip_actual, "ent": r.ent_actual,
             "dr_key_found": r.dr_key_found, "duration_s": r.duration_s,
             "error": r.error_msg, "run_dir": r.run_dir}
            for r in results
        ],
    }
    json_path = log.log_path.parent / "phase4_smoke_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log.info(f"\nJSON summary: {json_path}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 4 Step 4.4: Smoke Test (PPO CLI override + DR variants)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--only", choices=["ppo", "dr"], default=None,
        help="只运行 PPO 或 DR 测试子集",
    )
    parser.add_argument(
        "--skip-ros2", action="store_true",
        help="跳过 ROS2 publisher 启动（假设已在其他终端运行）",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只打印命令，不实际执行训练",
    )
    parser.add_argument(
        "--project-root", type=Path, default=None,
        help="项目根目录 (default: 自动检测)",
    )
    args = parser.parse_args()

    # 确定项目根目录
    if args.project_root:
        project_root = args.project_root
    else:
        script_path = Path(__file__).resolve()
        project_root = script_path.parents[4]  # docs/daily_logs/2026-3/2026-03-10/ -> root

    if not (project_root / "CLAUDE.md").exists():
        print(f"[ERROR] 项目根目录无效: {project_root}")
        sys.exit(1)

    # 选择测试集
    tests: list[SmokeTestCase] = []
    if args.only == "ppo":
        tests = PPO_TESTS
    elif args.only == "dr":
        tests = DR_TESTS
    else:
        tests = PPO_TESTS + DR_TESTS

    print(f"[INFO] 项目根目录: {project_root}")
    print(f"[INFO] 测试数量: {len(tests)} ({args.only or 'all'})")

    # 初始化日志
    log_dir = project_root / "logs" / "smoke" / "phase4_smoke"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"phase4_smoke_{timestamp}.log"
    log = DualLogger(log_path)
    log.info(f"Project root: {project_root}")
    log.info(f"Tests: {[t.name for t in tests]}")

    # ROS2 Publisher
    ros2_proc: Optional[subprocess.Popen] = None
    if not args.skip_ros2 and not args.dry_run:
        ros2_proc = start_ros2_publisher(project_root)
        if ros2_proc is None:
            log.error("ROS2 publisher 启动失败，退出")
            log.close()
            sys.exit(1)

    # 执行测试
    results: list[SmokeResult] = []
    try:
        for i, tc in enumerate(tests):
            log.info(f"\n[PROGRESS] {i+1}/{len(tests)}")
            result = run_single_test(tc, project_root, log, dry_run=args.dry_run)
            results.append(result)
    except KeyboardInterrupt:
        log.error("用户中断")
    finally:
        if ros2_proc is not None:
            stop_ros2_publisher(ros2_proc)

    # 汇总
    if results:
        print_summary(results, log)

    log.close()
    print(f"\n[INFO] 完整日志: {log_path}")

    # 退出码
    all_passed = all(r.passed for r in results) if results else False
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()