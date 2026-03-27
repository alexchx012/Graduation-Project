#!/usr/bin/env python3
"""Phase MORL M9: physical metric evaluation script."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
GO1_SCRIPT_DIR = PROJECT_ROOT / "scripts" / "go1-ros2-test"

for path in (SCRIPT_DIR, GO1_SCRIPT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from scenario_defs import get_scenario_spec, list_scenarios

DEFAULT_TASK = "Isaac-Velocity-MORL-Unitree-Go1-ROS2Cmd-Play-v0"
ROS2_MORL_TASK_IDS = {
    "Isaac-Velocity-MORL-Unitree-Go1-ROS2Cmd-v0",
    "Isaac-Velocity-MORL-Unitree-Go1-ROS2Cmd-Play-v0",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a MORL policy and compute physical metrics.",
        conflict_handler="resolve",
    )
    parser.add_argument("--task", type=str, default=DEFAULT_TASK, help="Name of the task.")
    parser.add_argument(
        "--agent",
        type=str,
        default="rsl_rl_cfg_entry_point",
        help="Name of the RL agent configuration entry point.",
    )
    parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to simulate.")
    parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment.")
    parser.add_argument("--eval_steps", type=int, default=3000, help="Total environment steps to evaluate.")
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
        "--recovery_err_thresh",
        type=float,
        default=0.25,
        help="Velocity tracking error threshold used by recovery_time.",
    )
    parser.add_argument(
        "--recovery_min_stable_steps",
        type=int,
        default=1,
        help="Number of consecutive stable steps required to count a recovery.",
    )
    parser.add_argument(
        "--summary_json",
        type=str,
        default=None,
        help="Optional output path to save evaluation summary JSON.",
    )
    parser.add_argument("--load_run", type=str, default=None, help="Run directory to evaluate.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file name to load.")
    parser.add_argument(
        "--scenario",
        type=str,
        choices=list_scenarios(),
        default=None,
        help="Optional Phase 4 scenario override applied before env creation.",
    )
    parser.add_argument(
        "--allow_no_ros2",
        action="store_true",
        default=False,
        help="If set, allow evaluation to continue even when no ROS2 message is received. "
        "Without this flag, a ROS2 timeout is a fatal error.",
    )
    parser.add_argument(
        "--skip_ros2",
        action="store_true",
        default=False,
        help="Skip ROS2 bridge setup entirely. Use Isaac Lab internal command generator "
        "instead of WSL ROS2 publisher. This is the correct mode when the env config "
        "already uses the built-in random command generator.",
    )
    return parser


def infer_policy_id(load_run: str | None) -> str | None:
    if not load_run:
        return None
    normalized = load_run.rstrip("\\/")
    base_name = os.path.basename(normalized)
    match = re.search(r"(morl_p\d+_seed\d+)$", base_name)
    return match.group(1) if match else base_name


def _extract_velocity_metrics(command_term, commanded_velocity, root_lin_vel_b) -> tuple[float, float, float]:
    """Read tracked velocity metrics from the command term or derive them from tensors."""
    metrics = getattr(command_term, "metrics", {})
    required_metric_keys = ("cmd_vx", "vx_meas", "vx_abs_err")
    if all(metric_key in metrics for metric_key in required_metric_keys):
        return (
            float(metrics["cmd_vx"].mean().item()),
            float(metrics["vx_meas"].mean().item()),
            float(metrics["vx_abs_err"].mean().item()),
        )

    cmd_vx_tensor = commanded_velocity[:, 0]
    vx_meas_tensor = root_lin_vel_b[:, 0]
    return (
        float(cmd_vx_tensor.mean().item()),
        float(vx_meas_tensor.mean().item()),
        float((cmd_vx_tensor - vx_meas_tensor).abs().mean().item()),
    )


def build_summary_metadata(scenario_metadata: dict[str, object] | None = None) -> dict[str, object]:
    """Build a stable metadata block for scenario-aware summaries."""

    metadata = {
        "scenario_id": None,
        "scenario_name": None,
        "terrain_mode": None,
        "cmd_vx": None,
        "disturbance_mode": "none",
        "analysis_group": "legacy",
    }
    if scenario_metadata:
        metadata.update(scenario_metadata)
    return metadata


def _freeze_command_ranges(base_velocity_cfg, vx: float) -> None:
    """Convert the stochastic velocity command into a fixed-command protocol."""

    base_velocity_cfg.ranges.lin_vel_x = (vx, vx)
    base_velocity_cfg.ranges.lin_vel_y = (0.0, 0.0)
    base_velocity_cfg.ranges.ang_vel_z = (0.0, 0.0)

    if hasattr(base_velocity_cfg.ranges, "heading"):
        base_velocity_cfg.ranges.heading = (0.0, 0.0)
    if hasattr(base_velocity_cfg, "heading_command"):
        base_velocity_cfg.heading_command = False
    if hasattr(base_velocity_cfg, "rel_heading_envs"):
        base_velocity_cfg.rel_heading_envs = 0.0
    if hasattr(base_velocity_cfg, "rel_standing_envs"):
        base_velocity_cfg.rel_standing_envs = 0.0
    if hasattr(base_velocity_cfg, "debug_vis"):
        base_velocity_cfg.debug_vis = False


def _ensure_generator_terrain(terrain_cfg) -> object | None:
    """Return terrain generator after switching the importer back to generator mode."""

    if terrain_cfg is None:
        return None
    terrain_cfg.terrain_type = "generator"
    terrain_generator = getattr(terrain_cfg, "terrain_generator", None)
    if terrain_generator is not None:
        terrain_generator.curriculum = False
    return terrain_generator


def apply_scenario_overrides(
    env_cfg,
    scenario_id: str,
    *,
    terrain_gen_module=None,
    event_term_cls=None,
    base_mdp_module=None,
) -> dict[str, object]:
    """Apply stage-4 scenario overrides on top of the MORL play env config."""

    spec = get_scenario_spec(scenario_id)
    _freeze_command_ranges(env_cfg.commands.base_velocity, spec.command_vx)

    terrain_cfg = getattr(getattr(env_cfg, "scene", None), "terrain", None)

    if spec.terrain_mode == "plane":
        if terrain_cfg is not None:
            terrain_cfg.terrain_type = "plane"
            terrain_generator = getattr(terrain_cfg, "terrain_generator", None)
            if terrain_generator is not None:
                terrain_generator.curriculum = False
    elif spec.terrain_mode == "slope_up":
        if terrain_gen_module is None:
            raise ValueError("terrain_gen_module is required for slope scenarios.")
        terrain_generator = _ensure_generator_terrain(terrain_cfg)
        if terrain_generator is not None:
            terrain_generator.sub_terrains = {
                "hf_pyramid_slope": terrain_gen_module.HfPyramidSlopedTerrainCfg(
                    proportion=1.0,
                    slope_range=(0.364, 0.364),
                    platform_width=2.0,
                    border_width=0.25,
                )
            }
    elif spec.terrain_mode == "slope_down":
        if terrain_gen_module is None:
            raise ValueError("terrain_gen_module is required for downhill scenarios.")
        terrain_generator = _ensure_generator_terrain(terrain_cfg)
        if terrain_generator is not None:
            terrain_generator.sub_terrains = {
                "hf_pyramid_slope_inv": terrain_gen_module.HfInvertedPyramidSlopedTerrainCfg(
                    proportion=1.0,
                    slope_range=(0.364, 0.364),
                    platform_width=2.0,
                    border_width=0.25,
                )
            }
    elif spec.terrain_mode == "stairs_15cm":
        if terrain_gen_module is None:
            raise ValueError("terrain_gen_module is required for stairs scenarios.")
        terrain_generator = _ensure_generator_terrain(terrain_cfg)
        if terrain_generator is not None:
            terrain_generator.sub_terrains = {
                "pyramid_stairs": terrain_gen_module.MeshPyramidStairsTerrainCfg(
                    proportion=1.0,
                    step_height_range=(0.15, 0.15),
                    step_width=0.3,
                    platform_width=3.0,
                    border_width=1.0,
                    holes=False,
                )
            }
    else:
        raise ValueError(f"Unsupported terrain_mode: {spec.terrain_mode}")

    if spec.disturbance_mode == "velocity_push_equivalent":
        if event_term_cls is not None and base_mdp_module is not None:
            env_cfg.events.push_robot = event_term_cls(
                func=base_mdp_module.push_by_setting_velocity,
                mode="interval",
                interval_range_s=(10.0, 15.0),
                params={"velocity_range": {"x": (0.0, 0.0), "y": (-0.5, 0.5)}},
            )
        elif getattr(env_cfg, "events", None) is not None:
            env_cfg.events.push_robot = {"mode": "velocity_push_equivalent"}

    return build_summary_metadata(
        {
            "scenario_id": spec.scenario_id,
            "scenario_name": spec.scenario_name,
            "terrain_mode": spec.terrain_mode,
            "cmd_vx": spec.command_vx,
            "disturbance_mode": spec.disturbance_mode,
            "analysis_group": spec.analysis_group,
        }
    )


def main() -> None:
    parser = build_parser()

    if any(flag in sys.argv[1:] for flag in ("-h", "--help")):
        parser.print_help()
        print("\nNote: run this script inside env_isaaclab for full AppLauncher support.")
        return

    from isaaclab.app import AppLauncher
    import cli_args

    cli_args.add_rsl_rl_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    args_cli, hydra_args = parser.parse_known_args()
    if args_cli.load_run is None:
        parser.error("--load_run is required for MORL evaluation.")

    sys.argv = [sys.argv[0]] + hydra_args
    _bootstrap_windows_ros2_env(args_cli)

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    try:
        _run_with_sim(args_cli, simulation_app)
    finally:
        simulation_app.close()


def _run_with_sim(args_cli, simulation_app) -> None:
    import importlib.metadata as metadata
    import platform

    import gymnasium as gym
    import isaaclab.envs.mdp as base_mdp
    import isaaclab.terrains as terrain_gen
    from isaaclab.managers import EventTermCfg as EventTerm
    import torch
    from metrics import compute_path_length, compute_recovery_time, summarize_morl_metrics
    from packaging import version
    from rsl_rl.runners import DistillationRunner, OnPolicyRunner

    import cli_args
    import isaaclab.utils.math as math_utils
    import robot_lab.tasks  # noqa: F401
    from checkpoint_utils import resolve_eval_checkpoint_path
    from isaaclab.envs import (
        DirectMARLEnv,
        DirectMARLEnvCfg,
        DirectRLEnvCfg,
        ManagerBasedRLEnvCfg,
        multi_agent_to_single_agent,
    )
    from isaaclab_tasks.utils import get_checkpoint_path
    from isaaclab_tasks.utils.hydra import hydra_task_config
    from isaaclab.utils.assets import retrieve_file_path
    from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

    RSL_RL_VERSION = "3.0.1"
    installed_version = metadata.version("rsl-rl-lib")
    if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
        if platform.system() == "Windows":
            cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
        else:
            cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
        raise RuntimeError(
            "Unsupported RSL-RL version. Install with:\n" + " ".join(cmd)
        )

    @hydra_task_config(args_cli.task, args_cli.agent)
    def _main(
        env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
        agent_cfg: RslRlBaseRunnerCfg,
    ) -> None:
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        env_cfg.seed = agent_cfg.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        env_cfg.observations.policy.enable_corruption = False
        env_cfg.events.randomize_apply_external_force_torque = None
        env_cfg.events.push_robot = None
        env_cfg.curriculum.command_levels_lin_vel = None
        env_cfg.curriculum.command_levels_ang_vel = None
        if getattr(env_cfg.curriculum, "terrain_levels", None) is not None:
            env_cfg.curriculum.terrain_levels = None
        if getattr(env_cfg.scene, "terrain", None) is not None:
            terrain_generator = getattr(env_cfg.scene.terrain, "terrain_generator", None)
            if terrain_generator is not None:
                terrain_generator.curriculum = False

        scenario_metadata = build_summary_metadata()
        if args_cli.scenario:
            scenario_metadata = apply_scenario_overrides(
                env_cfg,
                args_cli.scenario,
                terrain_gen_module=terrain_gen,
                event_term_cls=EventTerm,
                base_mdp_module=base_mdp,
            )

        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Loading experiment from directory: {log_root_path}")

        resume_path = resolve_eval_checkpoint_path(
            log_root_path=log_root_path,
            load_run=agent_cfg.load_run,
            load_checkpoint=agent_cfg.load_checkpoint,
            checkpoint_arg=args_cli.checkpoint,
            retrieve_file_path_func=retrieve_file_path,
            get_checkpoint_path_func=get_checkpoint_path,
        )
        print(f"[INFO] Using checkpoint: {resume_path}")

        env = gym.make(args_cli.task, cfg=env_cfg)

        ros2_bridge_adapter = None
        skip_ros2 = getattr(args_cli, "skip_ros2", False)
        if args_cli.task in ROS2_MORL_TASK_IDS and not skip_ros2:
            from isaacsim.core.utils.extensions import enable_extension
            from robot_lab.ros2_bridge import Ros2TwistBridgeCfg, Ros2TwistSubscriberGraphAdapter

            enable_extension("isaacsim.ros2.bridge")
            simulation_app.update()

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
                    if getattr(args_cli, "allow_no_ros2", False):
                        print(
                            f"[WARNING] ROS2 bridge: No message received within "
                            f"{bridge_cfg.startup_timeout_s}s. --allow_no_ros2 is set, "
                            f"continuing with ZERO commands (results will be invalid)."
                        )
                    else:
                        raise RuntimeError(
                            f"ROS2 bridge: No message received on "
                            f"'{bridge_cfg.topic_name}' within "
                            f"{bridge_cfg.startup_timeout_s}s. "
                            f"Check that the WSL ROS2 publisher is running and DDS "
                            f"discovery is healthy. Use --allow_no_ros2 to override."
                        )

        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)

        raw_env = env.unwrapped
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
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
        has_base_contact_term = "base_contact" in term_manager.active_terms
        robot = raw_env.scene["robot"]
        step_dt = float(raw_env.step_dt)

        required_metric_keys = ("cmd_vx", "vx_meas", "vx_abs_err")
        missing_metric_keys = tuple(
            metric_key for metric_key in required_metric_keys if metric_key not in command_term.metrics
        )
        if missing_metric_keys:
            print(
                "[MORL_EVAL] base_velocity metrics missing "
                f"{missing_metric_keys}; falling back to direct tensor-derived tracking metrics."
            )

        obs = env.get_observations()
        start_time = time.time()

        commanded_xy_steps: list[torch.Tensor] = []
        actual_xy_steps: list[torch.Tensor] = []
        joint_torque_steps: list[torch.Tensor] = []
        joint_vel_steps: list[torch.Tensor] = []
        actions_steps: list[torch.Tensor] = []
        ang_vel_xy_steps: list[torch.Tensor] = []
        pose_fluctuation_steps: list[torch.Tensor] = []

        cmd_vx_samples: list[float] = []
        vx_meas_samples: list[float] = []
        vx_abs_err_samples: list[float] = []
        timeout_rate_samples: list[float] = []
        base_contact_rate_samples: list[float] = []

        total_done_episodes = 0
        timeout_episodes = 0

        for step_idx in range(args_cli.eval_steps):
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                policy_nn.reset(dones)

            if step_idx < args_cli.warmup_steps:
                continue

            commanded_velocity = raw_env.command_manager.get_command("base_velocity")
            root_lin_vel_b = robot.data.root_lin_vel_b

            commanded_xy_steps.append(commanded_velocity[:, :2].detach().cpu())
            actual_xy_steps.append(root_lin_vel_b[:, :2].detach().cpu())
            joint_torque_steps.append(robot.data.applied_torque.detach().cpu())
            joint_vel_steps.append(robot.data.joint_vel.detach().cpu())
            actions_steps.append(actions.detach().cpu())
            ang_vel_xy_steps.append(robot.data.root_ang_vel_b[:, :2].detach().cpu())
            pose_fluctuation_steps.append(_compute_pose_fluctuation(robot.data.root_quat_w).detach().cpu())

            cmd_vx, vx_meas, vx_abs_err = _extract_velocity_metrics(
                command_term=command_term,
                commanded_velocity=commanded_velocity,
                root_lin_vel_b=root_lin_vel_b,
            )
            timeout_rate = float(term_manager.time_outs.float().mean().item())
            if has_base_contact_term:
                base_contact_rate = float(term_manager.get_term("base_contact").float().mean().item())
            else:
                base_contact_rate = 0.0

            cmd_vx_samples.append(cmd_vx)
            vx_meas_samples.append(vx_meas)
            vx_abs_err_samples.append(vx_abs_err)
            timeout_rate_samples.append(timeout_rate)
            base_contact_rate_samples.append(base_contact_rate)

            done_mask = dones.bool()
            if done_mask.any():
                total_done_episodes += int(done_mask.sum().item())
                timeout_episodes += int((done_mask & term_manager.time_outs.bool()).sum().item())

            if (step_idx + 1) % args_cli.report_interval == 0:
                print(
                    "[MORL_EVAL] "
                    f"step={step_idx + 1} "
                    f"cmd_vx={cmd_vx:.4f} "
                    f"vx_meas={vx_meas:.4f} "
                    f"vx_abs_err={vx_abs_err:.4f} "
                    f"timeout_rate={timeout_rate:.4f} "
                    f"base_contact_rate={base_contact_rate:.4f}"
                )

        elapsed_s = time.time() - start_time
        if not vx_abs_err_samples:
            raise RuntimeError("No effective evaluation samples collected. Reduce warmup_steps or increase eval_steps.")

        commanded_xy = torch.stack(commanded_xy_steps, dim=0).transpose(0, 1)
        actual_xy = torch.stack(actual_xy_steps, dim=0).transpose(0, 1)
        joint_torque = torch.stack(joint_torque_steps, dim=0).transpose(0, 1)
        joint_vel = torch.stack(joint_vel_steps, dim=0).transpose(0, 1)
        action_seq = torch.stack(actions_steps, dim=0).transpose(0, 1)
        ang_vel_xy = torch.stack(ang_vel_xy_steps, dim=0).transpose(0, 1)
        pose_fluctuation = torch.stack(pose_fluctuation_steps, dim=0).transpose(0, 1)

        distance = compute_path_length(actual_xy, dt=step_dt)
        physical_metrics = summarize_morl_metrics(
            commanded_xy=commanded_xy,
            actual_xy=actual_xy,
            joint_torque=joint_torque,
            joint_vel=joint_vel,
            actions=action_seq,
            ang_vel_xy=ang_vel_xy,
            pose_fluctuation=pose_fluctuation,
            dt=step_dt,
            distance=distance,
        )

        recovery = compute_recovery_time(
            torch.tensor(vx_abs_err_samples, dtype=torch.float32),
            threshold=args_cli.recovery_err_thresh,
            dt=step_dt,
            min_stable_steps=args_cli.recovery_min_stable_steps,
        )
        recovery_time = None if torch.isnan(recovery) else float(recovery.item())

        summary = {
            "policy_id": infer_policy_id(agent_cfg.load_run),
            "task": args_cli.task,
            **scenario_metadata,
            "checkpoint": resume_path,
            "load_run": agent_cfg.load_run,
            "eval_steps": int(args_cli.eval_steps),
            "warmup_steps": int(args_cli.warmup_steps),
            "effective_steps": int(len(vx_abs_err_samples)),
            "effective_env_steps": int(commanded_xy.shape[0] * commanded_xy.shape[1]),
            "step_dt": step_dt,
            "mean_cmd_vx": float(sum(cmd_vx_samples) / len(cmd_vx_samples)),
            "mean_vx_meas": float(sum(vx_meas_samples) / len(vx_meas_samples)),
            "mean_vx_abs_err": float(sum(vx_abs_err_samples) / len(vx_abs_err_samples)),
            **physical_metrics,
            "success_rate": float(timeout_episodes / total_done_episodes) if total_done_episodes > 0 else None,
            "mean_base_contact_rate": float(sum(base_contact_rate_samples) / len(base_contact_rate_samples)),
            "mean_timeout_rate": float(sum(timeout_rate_samples) / len(timeout_rate_samples)),
            "recovery_time": recovery_time,
            "elapsed_seconds": float(elapsed_s),
        }

        print("[MORL_EVAL] Summary")
        for key in (
            "J_speed",
            "J_energy",
            "J_smooth",
            "J_stable",
            "success_rate",
            "mean_base_contact_rate",
            "mean_timeout_rate",
            "recovery_time",
        ):
            print(f"[MORL_EVAL] {key}: {summary[key]}")
        print("[MORL_EVAL_JSON] " + json.dumps(summary, ensure_ascii=True))

        # Defence-in-depth: reject zero-command results unless explicitly allowed
        # Skip this check when --skip_ros2 is set (using internal command generator)
        if (
            args_cli.task in ROS2_MORL_TASK_IDS
            and abs(summary["mean_cmd_vx"]) < 1e-6
            and not getattr(args_cli, "allow_no_ros2", False)
            and not getattr(args_cli, "skip_ros2", False)
        ):
            raise RuntimeError(
                "Evaluation completed but mean_cmd_vx ≈ 0.0, indicating no ROS2 "
                "commands were received during the run. The resulting JSON would be "
                "invalid. Use --allow_no_ros2 to override this check."
            )

        if args_cli.summary_json:
            summary_json_path = os.path.abspath(args_cli.summary_json)
            os.makedirs(os.path.dirname(summary_json_path), exist_ok=True)
            with open(summary_json_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=True, indent=2)
            print(f"[MORL_EVAL] Summary JSON written to: {summary_json_path}")

        if ros2_bridge_adapter is not None:
            try:
                ros2_bridge_adapter.close()
            except Exception as err:
                print(f"[WARNING] Error closing ROS2 bridge adapter: {err}")
        env.close()

    _main()


def _bootstrap_windows_ros2_env(args_cli) -> None:
    if args_cli.task not in ROS2_MORL_TASK_IDS or sys.platform != "win32" or getattr(args_cli, "skip_ros2", False):
        return

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

    fastrtps_config = os.path.join(str(PROJECT_ROOT), "configs", "ros2", "fastrtps_win_to_wsl.xml")
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
            path_entries = [entry for entry in os.environ.get("PATH", "").split(";") if entry]
            if ros_lib_dir not in path_entries:
                os.environ["PATH"] = ";".join(path_entries + [ros_lib_dir]) if path_entries else ros_lib_dir
        else:
            print(f"[WARNING] ROS2 bridge library directory not found: {ros_lib_dir}")
    else:
        print("[WARNING] ISAACSIM_PATH not set; cannot preconfigure ROS2 bridge library path")


def _compute_pose_fluctuation(root_quat_w):
    import isaaclab.utils.math as math_utils

    yaw_only = math_utils.yaw_quat(root_quat_w)
    return math_utils.quat_box_minus(root_quat_w, yaw_only)[:, :2]


if __name__ == "__main__":
    main()
