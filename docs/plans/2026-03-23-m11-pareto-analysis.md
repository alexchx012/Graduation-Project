# M11 Pareto Analysis Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the `M11` Pareto/HV analysis pipeline for MORL evaluation results and generate frozen, reproducible analysis artifacts from `logs/eval/phase_morl/*.json`.

**Architecture:** Add a standalone analysis script under `scripts/phase_morl/` that loads the current 20 evaluation JSON files, aggregates them into policy-level points, normalizes the four minimization objectives with fixed frozen bounds, computes the Pareto front and exact hypervolume without new dependencies, and writes JSON plus figure artifacts. Protect the implementation with unit tests that lock down aggregation, normalization, Pareto selection, and HV behavior.

**Tech Stack:** Python, `numpy`, `matplotlib`, `pytest`

### Task 1: Freeze analysis contract

**Files:**
- Create: `docs/plans/2026-03-23-m11-pareto-analysis.md`
- Modify: `docs/daily_logs/2026-3/2026-03-14/phase_morl_plan.md` (reference only, no change required)

**Step 1: Confirm required inputs and outputs**

- Input directory: `logs/eval/phase_morl/`
- Input selection: only JSON files whose base name matches an active run directory under `logs/rsl_rl/unitree_go1_rough/`
- Primary objectives: `J_speed`, `J_energy`, `J_smooth`, `J_stable`
- Primary output JSON: `logs/eval/phase_morl/pareto_analysis.json`
- Primary figures: `docs/figures/pareto_front_pairwise.png`, `docs/figures/pareto_front_policy_summary.png`

**Step 2: Freeze normalization bounds**

- Use fixed manual bounds instead of adaptive candidate min/max
- Record bounds in script constants and in output JSON

**Step 3: Define policy-level aggregation**

- Policies with multiple seeds: aggregate by mean/std across seeds
- Policies with only `seed42`: treat the single point as the policy point
- Preserve run-level rows in output JSON for traceability

### Task 2: Write failing tests

**Files:**
- Create: `tests/unit/test_analyze_pareto.py`
- Test: `tests/unit/test_analyze_pareto.py`

**Step 1: Write the failing test for fixed-bound normalization**

Test that:
- normalization uses provided frozen bounds
- values are clipped into `[0, 1]`
- returned structure includes bounds and normalized objectives

**Step 2: Write the failing test for Pareto front extraction**

Test that:
- Pareto logic treats all four objectives as minimization
- dominated points are excluded
- front membership mask is stable

**Step 3: Write the failing test for exact hypervolume**

Test that:
- HV is computed from minimization-space normalized points
- a simple 2D/3D toy case returns the expected union volume

**Step 4: Write the failing test for policy aggregation and output payload**

Test that:
- multiple runs for one policy aggregate to mean/std
- single-seed policies keep `num_seeds=1`
- output JSON payload contains `run_level`, `policy_level`, `pareto_front`, `hypervolume`, `normalization_bounds`, `ref_point`

**Step 5: Run test to verify it fails**

Run:

```powershell
conda run -n env_isaaclab python -m pytest -q tests/unit/test_analyze_pareto.py
```

Expected: FAIL because `scripts/phase_morl/analyze_pareto.py` does not exist yet.

### Task 3: Implement minimal analysis script

**Files:**
- Create: `scripts/phase_morl/analyze_pareto.py`
- Modify: `tests/unit/test_analyze_pareto.py`

**Step 1: Add loader and parser**

Implement helpers for:
- discovering active run names
- loading matching eval JSON summaries
- parsing `policy_id`, `policy_name`, `seed`

**Step 2: Add policy aggregation**

Implement:
- `aggregate_policy_rows(...)`
- per-policy mean/std for the four primary objectives
- pass-through of `success_rate` and supplemental metrics for reporting only

**Step 3: Add fixed-bound normalization**

Implement:
- frozen bounds constant for the four minimization objectives
- normalization with clipping to `[0, 1]`
- bounds recorded into output JSON

**Step 4: Add Pareto and hypervolume**

Implement:
- exact Pareto front mask for minimization
- exact HV using inclusion-exclusion over dominated hyper-rectangles
- `ref_point` strictly worse than normalized worst point

**Step 5: Add CLI and artifact output**

Implement CLI that:
- reads from `logs/eval/phase_morl`
- writes `pareto_analysis.json`
- writes pairwise and summary figures to `docs/figures`

### Task 4: Verify on real data

**Files:**
- Run: `scripts/phase_morl/analyze_pareto.py`

**Step 1: Run unit tests**

```powershell
conda run -n env_isaaclab python -m pytest -q tests/unit/test_analyze_pareto.py
```

Expected: PASS

**Step 2: Run the analysis script**

```powershell
conda run -n env_isaaclab python scripts/phase_morl/analyze_pareto.py
```

Expected:
- `logs/eval/phase_morl/pareto_analysis.json` exists
- `docs/figures/pareto_front_pairwise.png` exists
- `docs/figures/pareto_front_policy_summary.png` exists

**Step 3: Sanity-check output**

Verify that:
- only the four minimization objectives enter Pareto/HV
- `success_rate` remains supplemental
- the output names the Pareto policies explicitly

### Task 5: Update tracking docs if status changes

**Files:**
- Modify: `docs/daily_logs/2026-3/2026-03-23/2026-3-23.md` (only if M11 is fully completed)

**Step 1: Update daily log**

If the script runs successfully and the artifacts are generated, mark `M11` complete in the daily log and add the produced artifact paths.
