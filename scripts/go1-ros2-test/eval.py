# Canonical source: scripts/go1-ros2-test/eval.py
# Deployed to: (standalone script, not in robot_lab package)
# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""Script to evaluate an RL agent with RSL-RL in inference mode."""

"""Launch Isaac Sim Simulator first."""

import argparse
import json
import os
import sys
import time

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Evaluate an RL agent with RSL-RL (no learning updates)."
)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Velocity-Flat-Unitree-Go1-ROS2Cmd-Play-v0",
    help="Name of the task.",
)
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="Name of the RL agent configuration entry point.",
)
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to simulate."
)
parser.add_argument(
    "--seed", type=int, default=42, help="Seed used for the environment"
)
parser.add_argument(
    "--eval_steps", type=int, default=3000, help="Total environment steps to evaluate."
)
parser.add_argument(
    "--warmup_steps",
    type=int,
    default=300,
    help="Warmup steps ignored from final statistics.",
)
parser.add_argument(
    "--report_interval",
    type=int,
    default=250,
    help="Print interval for running statistics.",
)
parser.add_argument(
    "--target_vx",
    type=float,
    default=1.0,
    help="Target forward velocity used for pass/fail check.",
)
parser.add_argument(
    "--pass_abs_err",
    type=float,
    default=0.1,
    help="Pass threshold for mean vx absolute error.",
)
parser.add_argument(
    "--pass_stable_ratio",
    type=float,
    default=0.9,
    help="Pass threshold for stable ratio.",
)
parser.add_argument(
    "--stable_err_thresh",
    type=float,
    default=0.1,
    help="Error threshold used to count a step as stable.",
)
parser.add_argument(
    "--strict_pass",
    action="store_true",
    default=False,
    help="Return non-zero exit code when evaluation does not pass thresholds.",
)
parser.add_argument(
    "--summary_json",
    type=str,
    default=None,
    help="Optional output path to save evaluation summary JSON.",
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ROS2 command-driven task ids
_ROS2_TASK_IDS = {
    "Isaac-Velocity-Flat-Unitree-Go1-ROS2Cmd-v0",
    "Isaac-Velocity-Flat-Unitree-Go1-ROS2Cmd-Play-v0",
}


if args_cli.task in _ROS2_TASK_IDS and sys.platform == "win32":
    # Prevent child-process crash dialogs from blocking headless automation.
    try:
        import ctypes

        ctypes.windll.kernel32.SetErrorMode(
            0x0001 | 0x0002
        )  # SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX
    except Exception:
        pass

    ros_distro = os.environ.get("ROS_DISTRO", "humble")
    if ros_distro not in {"humble", "jazzy"}:
        ros_distro = "humble"
    os.environ["ROS_DISTRO"] = ros_distro
    os.environ.setdefault("ROS_DOMAIN_ID", "0")
    os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
    os.environ.setdefault("FASTDDS_BUILTIN_TRANSPORTS", "UDPv4")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    fastrtps_config = os.path.join(
        project_root, "configs", "ros2", "fastrtps_win_to_wsl.xml"
    )
    if os.path.isfile(fastrtps_config):
        os.environ.setdefault("FASTRTPS_DEFAULT_PROFILES_FILE", fastrtps_config)
        os.environ.setdefault("FASTDDS_DEFAULT_PROFILES_FILE", fastrtps_config)
        print(f"[ROS2] Using FastRTPS config: {fastrtps_config}")
    else:
        print(f"[WARNING] FastRTPS config not found: {fastrtps_config}")

    isaacsim_root = os.environ.get("ISAACSIM_PATH")
    if isaacsim_root:
        ros_lib_dir = os.path.join(
            isaacsim_root, "exts", "isaacsim.ros2.bridge", ros_distro, "lib"
        )
        if os.path.isdir(ros_lib_dir):
            path_entries = [p for p in os.environ.get("PATH", "").split(";") if p]
            if ros_lib_dir not in path_entries:
                os.environ["PATH"] = (
                    ";".join(path_entries + [ros_lib_dir])
                    if path_entries
                    else ros_lib_dir
                )
        else:
            print(f"[WARNING] ROS2 bridge library directory not found: {ros_lib_dir}")
    else:
        print(
            "[WARNING] ISAACSIM_PATH not set; cannot preconfigure ROS2 bridge library path"
        )

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [
            r".\isaaclab.bat",
            "-p",
            "-m",
            "pip",
            "install",
            f"rsl-rl-lib=={RSL_RL_VERSION}",
        ]
    else:
        cmd = [
            "./isaaclab.sh",
            "-p",
            "-m",
            "pip",
            "install",
            f"rsl-rl-lib=={RSL_RL_VERSION}",
        ]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import robot_lab.tasks  # noqa: F401  # isort: skip


@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
):
    """Run policy evaluation without learning updates."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 1
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )

    # Evaluation should match play behavior.
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.events.randomize_apply_external_force_torque = None
    env_cfg.events.push_robot = None
    env_cfg.curriculum.command_levels_lin_vel = None
    env_cfg.curriculum.command_levels_ang_vel = None

    # Disable terrain curriculum so evaluation uses a fixed terrain distribution.
    if getattr(env_cfg.curriculum, "terrain_levels", None) is not None:
        env_cfg.curriculum.terrain_levels = None
    if getattr(env_cfg.scene, "terrain", None) is not None:
        terrain_gen = getattr(env_cfg.scene.terrain, "terrain_generator", None)
        if terrain_gen is not None:
            terrain_gen.curriculum = False

    # Resolve checkpoint path.
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
        )
    print(f"[INFO] Using checkpoint: {resume_path}")

    # Create environment.
    env = gym.make(args_cli.task, cfg=env_cfg)

    # ROS2 bridge adapter setup for ROS2 command tasks.
    ros2_bridge_adapter = None
    if args_cli.task in _ROS2_TASK_IDS:
        try:
            from isaacsim.core.utils.extensions import enable_extension

            enable_extension("isaacsim.ros2.bridge")
            simulation_app.update()

            from robot_lab.ros2_bridge import (
                Ros2TwistBridgeCfg,
                Ros2TwistSubscriberGraphAdapter,
            )

            bridge_cfg = Ros2TwistBridgeCfg(
                topic_name="/go1/cmd_vel",
                queue_size=10,
                startup_mode="startup_blocking",
                startup_timeout_s=15.0,
                command_attr="ros2_latest_cmd_vel",
                command_stamp_attr="ros2_latest_cmd_stamp_s",
            )

            ros2_bridge_adapter = Ros2TwistSubscriberGraphAdapter(bridge_cfg)
            ros2_bridge_adapter.setup()
            ros2_bridge_adapter.attach(env)

            if bridge_cfg.startup_mode == "startup_blocking":
                success = ros2_bridge_adapter.wait_for_first_message(
                    timeout_s=bridge_cfg.startup_timeout_s
                )
                if not success:
                    print(
                        f"[WARNING] ROS2 bridge: No message received within "
                        f"{bridge_cfg.startup_timeout_s}s, continuing anyway..."
                    )
        except Exception as err:
            raise RuntimeError(
                f"Failed to setup ROS2 bridge for evaluation: {err}"
            ) from err

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    raw_env = env.unwrapped
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(
            env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
        )
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(
            env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
        )
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    command_term = raw_env.command_manager.get_term("base_velocity")
    term_manager = raw_env.termination_manager

    required_metric_keys = ("cmd_vx", "vx_meas", "vx_abs_err")
    for metric_key in required_metric_keys:
        if metric_key not in command_term.metrics:
            raise RuntimeError(
                f"Missing required metric '{metric_key}' in command term metrics."
            )

    has_base_contact_term = "base_contact" in term_manager.active_terms

    obs = env.get_observations()
    start_time = time.time()

    cmd_vx_samples: list[float] = []
    vx_meas_samples: list[float] = []
    vx_abs_err_samples: list[float] = []
    timeout_rate_samples: list[float] = []
    base_contact_rate_samples: list[float] = []

    for step_idx in range(args_cli.eval_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy_nn.reset(dones)

        if step_idx < args_cli.warmup_steps:
            continue

        cmd_vx = float(command_term.metrics["cmd_vx"].mean().item())
        vx_meas = float(command_term.metrics["vx_meas"].mean().item())
        vx_abs_err = float(command_term.metrics["vx_abs_err"].mean().item())
        timeout_rate = float(term_manager.time_outs.float().mean().item())
        if has_base_contact_term:
            base_contact_rate = float(
                term_manager.get_term("base_contact").float().mean().item()
            )
        else:
            base_contact_rate = 0.0

        cmd_vx_samples.append(cmd_vx)
        vx_meas_samples.append(vx_meas)
        vx_abs_err_samples.append(vx_abs_err)
        timeout_rate_samples.append(timeout_rate)
        base_contact_rate_samples.append(base_contact_rate)

        if (step_idx + 1) % args_cli.report_interval == 0:
            print(
                "[EVAL] "
                f"step={step_idx + 1} "
                f"cmd_vx={cmd_vx:.4f} "
                f"vx_meas={vx_meas:.4f} "
                f"vx_abs_err={vx_abs_err:.4f} "
                f"timeout_rate={timeout_rate:.4f} "
                f"base_contact_rate={base_contact_rate:.4f}"
            )

    elapsed_s = time.time() - start_time

    if len(vx_abs_err_samples) == 0:
        raise RuntimeError(
            "No effective evaluation samples collected. Reduce warmup_steps or increase eval_steps."
        )

    vx_abs_err_tensor = torch.tensor(vx_abs_err_samples, dtype=torch.float32)
    stable_ratio = float(
        (vx_abs_err_tensor <= args_cli.stable_err_thresh).float().mean().item()
    )

    summary = {
        "task": args_cli.task,
        "checkpoint": resume_path,
        "eval_steps": int(args_cli.eval_steps),
        "warmup_steps": int(args_cli.warmup_steps),
        "effective_samples": int(len(vx_abs_err_samples)),
        "target_vx": float(args_cli.target_vx),
        "mean_cmd_vx": float(sum(cmd_vx_samples) / len(cmd_vx_samples)),
        "mean_vx_meas": float(sum(vx_meas_samples) / len(vx_meas_samples)),
        "mean_vx_abs_err": float(sum(vx_abs_err_samples) / len(vx_abs_err_samples)),
        "p95_vx_abs_err": float(torch.quantile(vx_abs_err_tensor, 0.95).item()),
        "stable_ratio": stable_ratio,
        "stable_err_thresh": float(args_cli.stable_err_thresh),
        "mean_timeout_rate": float(
            sum(timeout_rate_samples) / len(timeout_rate_samples)
        ),
        "mean_base_contact_rate": float(
            sum(base_contact_rate_samples) / len(base_contact_rate_samples)
        ),
        "elapsed_seconds": float(elapsed_s),
    }

    summary["pass"] = bool(
        abs(summary["mean_cmd_vx"] - args_cli.target_vx) <= 0.05
        and summary["mean_vx_abs_err"] <= args_cli.pass_abs_err
        and summary["stable_ratio"] >= args_cli.pass_stable_ratio
    )
    summary["pass_criteria"] = {
        "abs(mean_cmd_vx-target_vx)<=": 0.05,
        "mean_vx_abs_err<=": float(args_cli.pass_abs_err),
        "stable_ratio>=": float(args_cli.pass_stable_ratio),
    }

    print("[EVAL] Summary")
    for key in (
        "mean_cmd_vx",
        "mean_vx_meas",
        "mean_vx_abs_err",
        "p95_vx_abs_err",
        "stable_ratio",
        "mean_timeout_rate",
        "mean_base_contact_rate",
        "pass",
    ):
        print(f"[EVAL] {key}: {summary[key]}")
    print("[EVAL_JSON] " + json.dumps(summary, ensure_ascii=True))

    if args_cli.summary_json:
        summary_json_path = os.path.abspath(args_cli.summary_json)
        os.makedirs(os.path.dirname(summary_json_path), exist_ok=True)
        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=True, indent=2)
        print(f"[EVAL] Summary JSON written to: {summary_json_path}")

    if args_cli.strict_pass and not summary["pass"]:
        raise RuntimeError("Evaluation did not meet pass criteria.")

    if ros2_bridge_adapter is not None:
        try:
            ros2_bridge_adapter.close()
        except Exception as err:
            print(f"[WARNING] Error closing ROS2 bridge adapter: {err}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
