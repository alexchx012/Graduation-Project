"""Phase 4 analysis: Determine best PPO and best DR configurations.

Reads TensorBoard event files from all Phase 4 training runs,
extracts episode reward curves, and computes:
  - Last-100-iter mean reward per run (per seed)
  - Group mean ± std across 3 seeds
  - Ranking to identify best PPO and best DR

Usage:
    conda activate env_isaaclab
    python scripts/phase4-ppo-dr/analyze_best_ppo_dr.py
"""

import os
import glob
import json
import statistics
from collections import defaultdict

from tensorboard.backend.event_processing import event_accumulator


LOG_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "logs", "rsl_rl", "unitree_go1_rough")
LOG_ROOT = os.path.normpath(LOG_ROOT)

# --- Run registry: group_name -> list of (dir_name_substring, seed) ---
RUNS = {
    "baseline": [
        ("2026-03-08_16-46-27_baseline_rough_ros2cmd", 42),
        ("2026-03-10_10-59-57_baseline_rough_seed43", 43),
        ("2026-03-10_13-02-19_baseline_rough_seed44", 44),
    ],
    # PPO variants
    "ppo_lr_low": [
        ("2026-03-10_22-19-03_ppo_lr_low_seed42", 42),
        ("2026-03-11_00-01-36_ppo_lr_low_seed43", 43),
        ("2026-03-11_01-45-46_ppo_lr_low_seed44", 44),
    ],
    "ppo_lr_high": [
        ("2026-03-11_03-22-07_ppo_lr_high_seed42", 42),
        ("2026-03-11_05-03-37_ppo_lr_high_seed43", 43),
        ("2026-03-11_06-46-17_ppo_lr_high_seed44", 44),
    ],
    "ppo_clip_low": [
        ("2026-03-11_08-41-55_ppo_clip_low_seed42", 42),
        ("2026-03-12_07-50-58_ppo_clip_low_seed43", 43),
        ("2026-03-11_11-50-58_ppo_clip_low_seed44", 44),
    ],
    "ppo_clip_high": [
        ("2026-03-11_13-31-15_ppo_clip_high_seed42", 42),
        ("2026-03-11_15-12-02_ppo_clip_high_seed43", 43),
        ("2026-03-11_17-50-46_ppo_clip_high_seed44", 44),
    ],
    "ppo_ent_low": [
        ("2026-03-11_20-04-04_ppo_ent_low_seed42", 42),
        ("2026-03-11_21-45-36_ppo_ent_low_seed43", 43),
        ("2026-03-11_23-23-59_ppo_ent_low_seed44", 44),
    ],
    "ppo_ent_high": [
        ("2026-03-12_01-09-11_ppo_ent_high_seed42", 42),
        ("2026-03-12_03-06-44_ppo_ent_high_seed43", 43),
        ("2026-03-12_04-49-51_ppo_ent_high_seed44", 44),
    ],
    # DR variants
    "dr_friction": [
        ("2026-03-12_13-39-08_dr_friction_seed42", 42),
        ("2026-03-12_15-18-38_dr_friction_seed43", 43),
        ("2026-03-12_16-59-25_dr_friction_seed44", 44),
    ],
    "dr_mass": [
        ("2026-03-12_18-41-21_dr_mass_seed42", 42),
        ("2026-03-12_20-27-22_dr_mass_seed43", 43),
        ("2026-03-12_22-07-37_dr_mass_seed44", 44),
    ],
    "dr_push": [
        ("2026-03-12_23-51-09_dr_push_seed42", 42),
        ("2026-03-13_01-32-02_dr_push_seed43", 43),
        ("2026-03-13_03-08-00_dr_push_seed44", 44),
    ],
}

LAST_N_ITERS = 100  # Average over last 100 iterations for final performance


def extract_reward_scalars(logdir):
    """Extract episode reward scalar from a TensorBoard event log directory.

    Returns list of (step, value) tuples sorted by step.
    Tries common tag names used by rsl_rl.
    """
    ea = event_accumulator.EventAccumulator(
        logdir,
        size_guidance={event_accumulator.SCALARS: 0},  # load all
    )
    ea.Reload()
    tags = ea.Tags().get("scalars", [])

    # rsl_rl commonly logs episode reward under these tag names
    reward_tag_candidates = [
        "Episode/mean_reward",
        "Train/mean_reward",
        "Episode_Reward/mean",
        "mean_reward",
        "Reward/mean",
        "Loss/mean_reward",
    ]

    chosen_tag = None
    for candidate in reward_tag_candidates:
        if candidate in tags:
            chosen_tag = candidate
            break

    if chosen_tag is None:
        # Try partial match
        for t in tags:
            if "reward" in t.lower() and "mean" in t.lower():
                chosen_tag = t
                break

    if chosen_tag is None:
        print(f"  [WARN] No reward tag found in {logdir}")
        print(f"  Available tags: {tags[:20]}")
        return None, tags

    events = ea.Scalars(chosen_tag)
    data = [(e.step, e.value) for e in events]
    data.sort(key=lambda x: x[0])
    return data, chosen_tag


def compute_last_n_mean(data, n=LAST_N_ITERS):
    """Compute mean reward over the last N data points."""
    if not data or len(data) == 0:
        return None
    tail = data[-n:]
    return statistics.mean([v for _, v in tail])


def compute_convergence_iter(data, window=100, threshold=0.01):
    """Find the first iteration where the sliding window mean changes < threshold.

    Convergence iter = first step where 100-iter rolling mean plateau.
    """
    if not data or len(data) < window * 2:
        return None
    values = [v for _, v in data]
    steps = [s for s, _ in data]
    for i in range(window, len(values) - window):
        prev_mean = statistics.mean(values[i - window:i])
        curr_mean = statistics.mean(values[i:i + window])
        if prev_mean == 0:
            continue
        change = abs(curr_mean - prev_mean) / abs(prev_mean)
        if change < threshold:
            return steps[i]
    return None  # never converged


def cohens_d(group1, group2):
    """Compute Cohen's d effect size (two independent samples)."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return None
    m1, m2 = statistics.mean(group1), statistics.mean(group2)
    s1, s2 = statistics.stdev(group1), statistics.stdev(group2)
    pooled_std = ((s1**2 * (n1 - 1) + s2**2 * (n2 - 1)) / (n1 + n2 - 2)) ** 0.5
    if pooled_std == 0:
        return float("inf") if m1 != m2 else 0.0
    return (m1 - m2) / pooled_std


def main():
    print(f"Log root: {LOG_ROOT}")
    print(f"Averaging over last {LAST_N_ITERS} iterations\n")

    # Phase 1: Extract data
    group_results = {}  # group_name -> {seeds: [...], last_n_means: [...], convergence_iters: [...]}
    reward_tag_used = None

    for group_name, runs in RUNS.items():
        seed_means = []
        seed_conv_iters = []
        print(f"=== {group_name} ===")
        for dirname, seed in runs:
            logdir = os.path.join(LOG_ROOT, dirname)
            if not os.path.isdir(logdir):
                print(f"  [ERROR] Directory not found: {logdir}")
                seed_means.append(None)
                seed_conv_iters.append(None)
                continue

            data, tag = extract_reward_scalars(logdir)
            if data is None:
                seed_means.append(None)
                seed_conv_iters.append(None)
                continue

            if reward_tag_used is None:
                reward_tag_used = tag
                print(f"  [INFO] Using tag: '{tag}' (total {len(data)} data points)")

            last_n_mean = compute_last_n_mean(data, LAST_N_ITERS)
            conv_iter = compute_convergence_iter(data)
            seed_means.append(last_n_mean)
            seed_conv_iters.append(conv_iter)
            print(f"  seed={seed}: last-{LAST_N_ITERS}-mean={last_n_mean:.2f}, "
                  f"total_points={len(data)}, conv_iter={conv_iter}")

        valid_means = [m for m in seed_means if m is not None]
        group_results[group_name] = {
            "seed_means": seed_means,
            "valid_means": valid_means,
            "group_mean": statistics.mean(valid_means) if valid_means else None,
            "group_std": statistics.stdev(valid_means) if len(valid_means) >= 2 else None,
            "convergence_iters": seed_conv_iters,
        }
        if valid_means:
            gm = group_results[group_name]["group_mean"]
            gs = group_results[group_name]["group_std"]
            print(f"  >> Group mean: {gm:.2f} +/- {(gs if gs else 0):.2f}")
        print()

    # Phase 2: Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE: Last-100-iter Mean Episode Reward (mean +/- std across 3 seeds)")
    print("=" * 80)
    header = f"{'Config':<20} {'Mean':>10} {'Std':>10} {'Seed42':>10} {'Seed43':>10} {'Seed44':>10}"
    print(header)
    print("-" * 80)

    baseline_means = group_results.get("baseline", {}).get("valid_means", [])

    for group_name in RUNS:
        r = group_results[group_name]
        sm = r["seed_means"]
        gm = r["group_mean"]
        gs = r["group_std"]
        s42 = f"{sm[0]:.2f}" if sm[0] is not None else "N/A"
        s43 = f"{sm[1]:.2f}" if sm[1] is not None else "N/A"
        s44 = f"{sm[2]:.2f}" if sm[2] is not None else "N/A"
        gm_s = f"{gm:.2f}" if gm is not None else "N/A"
        gs_s = f"{gs:.2f}" if gs is not None else "N/A"
        print(f"{group_name:<20} {gm_s:>10} {gs_s:>10} {s42:>10} {s43:>10} {s44:>10}")

    # Phase 3: Ranking
    print("\n" + "=" * 80)
    print("PPO RANKING (by group mean, descending)")
    print("=" * 80)
    ppo_groups = {k: v for k, v in group_results.items() if k.startswith("ppo_")}
    ppo_sorted = sorted(ppo_groups.items(), key=lambda x: x[1]["group_mean"] or -999, reverse=True)
    for rank, (name, r) in enumerate(ppo_sorted, 1):
        gm = r["group_mean"]
        gs = r["group_std"]
        d = cohens_d(r["valid_means"], baseline_means) if baseline_means and r["valid_means"] else None
        d_str = f"{d:+.2f}" if d is not None else "N/A"
        print(f"  #{rank} {name:<20} mean={gm:.2f} +/- {(gs if gs else 0):.2f}  "
              f"Cohen's d vs baseline: {d_str}")

    best_ppo = ppo_sorted[0][0] if ppo_sorted else None
    print(f"\n  >>> BEST PPO: {best_ppo} <<<")

    print("\n" + "=" * 80)
    print("DR RANKING (by group mean, descending)")
    print("=" * 80)
    dr_groups = {k: v for k, v in group_results.items() if k.startswith("dr_")}
    dr_sorted = sorted(dr_groups.items(), key=lambda x: x[1]["group_mean"] or -999, reverse=True)
    for rank, (name, r) in enumerate(dr_sorted, 1):
        gm = r["group_mean"]
        gs = r["group_std"]
        d = cohens_d(r["valid_means"], baseline_means) if baseline_means and r["valid_means"] else None
        d_str = f"{d:+.2f}" if d is not None else "N/A"
        print(f"  #{rank} {name:<20} mean={gm:.2f} +/- {(gs if gs else 0):.2f}  "
              f"Cohen's d vs baseline: {d_str}")

    best_dr = dr_sorted[0][0] if dr_sorted else None
    print(f"\n  >>> BEST DR: {best_dr} <<<")

    # Phase 4: Baseline comparison
    print("\n" + "=" * 80)
    print("BASELINE vs BEST")
    print("=" * 80)
    bl = group_results.get("baseline", {})
    if bl.get("group_mean") is not None:
        print(f"  Baseline:  {bl['group_mean']:.2f} +/- {(bl['group_std'] if bl['group_std'] else 0):.2f}")
    if best_ppo:
        bp = group_results[best_ppo]
        print(f"  Best PPO ({best_ppo}):  {bp['group_mean']:.2f} +/- {(bp['group_std'] if bp['group_std'] else 0):.2f}")
        d = cohens_d(bp["valid_means"], baseline_means)
        print(f"    Cohen's d: {d:+.2f}" if d else "    Cohen's d: N/A")
    if best_dr:
        bd = group_results[best_dr]
        print(f"  Best DR  ({best_dr}):  {bd['group_mean']:.2f} +/- {(bd['group_std'] if bd['group_std'] else 0):.2f}")
        d = cohens_d(bd["valid_means"], baseline_means)
        print(f"    Cohen's d: {d:+.2f}" if d else "    Cohen's d: N/A")

    # Phase 5: Convergence info
    print("\n" + "=" * 80)
    print("CONVERGENCE ITERATION (first iter where 100-iter rolling mean change < 1%)")
    print("=" * 80)
    for group_name in RUNS:
        r = group_results[group_name]
        ci = r["convergence_iters"]
        ci_strs = [str(c) if c is not None else "N/A" for c in ci]
        print(f"  {group_name:<20}  seeds: {', '.join(ci_strs)}")

    # Save JSON
    output = {
        "reward_tag": reward_tag_used,
        "last_n_iters": LAST_N_ITERS,
        "best_ppo": best_ppo,
        "best_dr": best_dr,
        "groups": {},
    }
    for group_name, r in group_results.items():
        output["groups"][group_name] = {
            "seed_means": r["seed_means"],
            "group_mean": r["group_mean"],
            "group_std": r["group_std"],
            "convergence_iters": r["convergence_iters"],
        }

    out_path = os.path.join(LOG_ROOT, "..", "..", "eval", "phase4_analysis", "best_ppo_dr_analysis.json")
    out_path = os.path.normpath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[Saved] {out_path}")

    # Recommendation for step 4.6b
    print("\n" + "=" * 80)
    print("RECOMMENDATION FOR STEP 4.6b (Combo Experiment)")
    print("=" * 80)
    if best_ppo and best_dr:
        ppo_params = {
            "ppo_lr_low": "--learning_rate 5e-4",
            "ppo_lr_high": "--learning_rate 3e-3",
            "ppo_clip_low": "--clip_param 0.1",
            "ppo_clip_high": "--clip_param 0.3",
            "ppo_ent_low": "--entropy_coef 0.005",
            "ppo_ent_high": "--entropy_coef 0.02",
        }
        dr_tasks = {
            "dr_friction": "Isaac-Velocity-Rough-Unitree-Go1-ROS2Cmd-DR-Friction-v0",
            "dr_mass": "Isaac-Velocity-Rough-Unitree-Go1-ROS2Cmd-DR-Mass-v0",
            "dr_push": "Isaac-Velocity-Rough-Unitree-Go1-ROS2Cmd-DR-Push-v0",
        }
        ppo_cli = ppo_params.get(best_ppo, "???")
        dr_task = dr_tasks.get(best_dr, "???")
        print(f"  Best PPO = {best_ppo} -> CLI: {ppo_cli}")
        print(f"  Best DR  = {best_dr}  -> Task: {dr_task}")
        print()
        for seed in [42, 43, 44]:
            print(f"  python scripts/go1-ros2-test/train.py ^")
            print(f"    --task {dr_task} ^")
            print(f"    --num_envs 4096 --max_iterations 1500 --headless ^")
            print(f"    --disable_ros2_tracking_tune --seed {seed} ^")
            print(f"    {ppo_cli} --run_name combo_best_seed{seed}")
            print()


if __name__ == "__main__":
    main()
