# 阶段四详细计划（2026-04-04 修订版）

> 修订日期：2026-04-04
> 计划窗口：2026-04-04 至 2026-04-20
> 蓝本：`docs/daily_logs/2026-3/2026-03-25/phase4_detailed_plan.md`
> 事实依据：
> - `docs/daily_logs/2026-3/2026-03-21/2026-3-21.md` 至 `docs/daily_logs/2026-4/2026-04-04/2026-4-04.md`
> - 当前仓库中的 `scripts/phase_morl/*.py`、`tests/unit/*.py`、`logs/rsl_rl/`、`logs/eval/`
> 本版目的：将 3 月 25 日版更新为从 2026-04-04 起可直接执行的阶段四计划，并补齐 manifest、baseline 归一化、消融泛化、聚合/QC、测试与统计口径。

---

## 1. 近半个月实验进度结论

- [Fact] 2026-03-21 至 2026-03-23 期间，项目先沿着“禁用 ROS2、使用随机命令、做统一测试集评估 + Pareto/HV”的路线推进，并在 2026-03-23 得到了旧 `phase_morl` family 的 `policy-level Pareto front = P9,P10` 与 `HV=0.337518`。
- [Fact] 2026-03-27 已完成阶段四场景链路的关键工程前置：`scripts/phase_morl/scenario_defs.py`、`run_morl_eval.py --scenario`、`S1-S6` 元数据输出，以及 `S1/S4/S6` smoke。
- [Fact] 2026-03-27 同时证伪了 3 月 23 日那批旧 MORL checkpoint 的核心前提：在 `S1 + --skip_ros2` 固定命令协议下，`P1/P2/P3/P4/P9/P10` 几乎都不前进，而 rough baseline 正常。
- [Fact] 2026-03-30 的只读分析将失败原因收敛为 5 个联合作用因素：MORL 奖励不对称、`ang_vel_z=[0,0]` 免费午餐、fresh Adam 第一步过大、`clip_param=0.3` 偏大、critic value 失配。
- [Fact] 2026-03-31 已完成 v2 修复路线：MORL curriculum、`repair_forward_v2`、`clip_param=0.2`、ROS2 bridge 清理、`run_full_eval_matrix.py`、`run_morl_ablation.py`、以及 `analyze_pareto.py --scenario` 适配。
- [Fact] 2026-04-02 完成 v2 confirm sweep 的 20 次训练和 `S1` 全量评估，但暴露出新的 seed 敏感性：A 层 seed43/44 部分退化，导致当时的 `S1` Pareto/HV 被污染。
- [Fact] 2026-04-02 已定位该问题根因为 `--init_checkpoint` 丢弃 Adam optimizer state，于是新增 `--init_with_optimizer`，并让 `run_morl_confirm_sweep_v2.py` 与 `run_morl_ablation.py` 默认启用。
- [Fact] 2026-04-04 已完成 A 层 seed43/44 的重训、20/20 的干净 `S1` 正式评估，以及新的 `S1` Pareto/HV：当前 `policy-level Pareto front = P1,P2,P3`，`Hypervolume = 0.2666`。
- [Inference + Evidence] 因此，阶段四的正式输入应切换为 2026-03-31 至 2026-04-04 形成的干净 `phase_morl_v2` family；2026-03-23 的旧 `phase_morl` family 只保留为历史对照证据。

---

## 2. 阶段四目标

阶段四现在只做五件事：

1. 用当前干净的 `phase_morl_v2` 策略集完成 **6 个正式场景** 的同口径评估。
2. 用 **manifest 驱动** 统一 MORL、baseline、ablation 三类 run 的正式执行链。
3. 建立 **聚合层 + QC 层 + Pareto/HV 层** 三段式分析产线。
4. 在冻结后的锚点上完成 **两组 reward 消融实验**，验证能效目标与平滑目标的局部必要性。
5. 产出可直接写入论文第 5 章和答辩 PPT 的表格、图和结论。

---

## 3. 明确不做的事

- 不把 2026-03-23 的 `logs/eval/phase_morl/pareto_analysis.json` 或 `HV=0.337518` 当作阶段四正式结果；它只保留为旧路线历史证据。
- 不把旧 `phase_morl` family 与新 `phase_morl_v2` family 混入同一张官方主表。
- 不继续依赖目录名正则去猜 `family / policy / seed / checkpoint`。
- 不把 `P5-P9 × seed42` 写成强统计结论；它们仍然只用于 exploratory overlay。
- 不在还没完成六场景主矩阵之前，就写死“最终推荐策略”或“最终消融锚点”。

---

## 4. 当前资产与正式化策略

### 4.1 当前可用资产

| 资产 | 当前状态 | 用途 |
| --- | --- | --- |
| `P1/P2/P3/P4/P10 × seed42/43/44` 的 v2 checkpoint | ✅ 已有 15 个 | 阶段四官方确认层主表 |
| `P5/P6/P7/P8/P9 × seed42` 的 v2 checkpoint | ✅ 已有 5 个 | exploratory Pareto 补点 |
| `logs/eval/phase_morl_v2/morl_p*_S1.json` | ✅ 已有 20 个 | 当前干净 `S1` 正式结果 |
| `logs/eval/phase_morl_v2/pareto_analysis_S1.json` | ✅ 已有 | 当前干净 `S1` 的 Pareto/HV 参考 |
| `scripts/phase_morl/scenario_defs.py` | ✅ 已有 | `S1-S6` 场景定义 |
| `scripts/phase_morl/run_morl_eval.py --scenario` | ✅ 已有 | 场景级正式评估入口 |
| `scripts/phase_morl/run_full_eval_matrix.py` | ✅ 已有 | MORL v2 主矩阵 runner，当前仍按目录命名发现 run |
| `scripts/phase_morl/run_morl_ablation.py` | ✅ 已有 | 当前为 `P10` 专用消融训练入口 |
| `logs/rsl_rl/unitree_go1_rough/2026-03-08_16-46-27_baseline_rough_ros2cmd/` | ✅ 已有 | baseline seed42 / warm-start 基准 |
| `logs/rsl_rl/unitree_go1_rough/_archive_20260331_pre_v2_sweep/2026-03-10_10-59-57_baseline_rough_seed43/` | ✅ 已有但在 archive | baseline seed43 原始资产 |
| `logs/rsl_rl/unitree_go1_rough/_archive_20260331_pre_v2_sweep/2026-03-10_13-02-19_baseline_rough_seed44/` | ✅ 已有但在 archive | baseline seed44 原始资产 |

### 4.2 当前仓库约束

- [Fact] `run_full_eval_matrix.py` 当前只按 `morl_pX_seedY` 发现 run，并固定输出 `morl_*_S*.json`。
- [Fact] `analyze_pareto.py` 当前也只按 `morl_pX_seedY` 发现 active runs，并使用一套硬编码的全局归一化边界与 `ref_point`。
- [Fact] `run_morl_ablation.py` 当前只定义了 `P10-no-energy` / `P10-no-smooth`，没有锚点参数。
- [Fact] `run_morl_eval.py --task Isaac-Velocity-Rough-Unitree-Go1-ROS2Cmd-Play-v0 --load_run 2026-03-08_16-46-27_baseline_rough_ros2cmd --checkpoint model_1499.pt --scenario S1 --skip_ros2` 已在 2026-04-04 smoke 中成功跑通并写出 JSON，因此 baseline eval 的问题不在 task 兼容性，而在正式链的 manifest 化、归一化与批量接入。
- [Inference + Evidence] 因此 baseline 和 ablation 尚未进入统一的正式评估链；如果不先补 manifest、聚合与解析层，后续表图只能靠临时人工拼接。

### 4.3 正式注册文件

本版计划新增 3 个正式注册文件，作为阶段四执行链的唯一输入源：

1. `scripts/phase_morl/manifests/phase4_main_manifest.json`
   - 记录 MORL 主矩阵与 baseline 主矩阵
   - 每条记录至少包含：
     - `family`
     - `policy_id`
     - `canonical_seed`
     - `run_dir`
     - `checkpoint`
     - `output_stem`
     - `evidence_layer`
     - `official_hv_eligible`

2. `scripts/phase_morl/manifests/phase4_analysis_config.json`
   - 记录正式冻结后的分析配置
   - 至少包含：
     - `normalization_bounds`
     - `ref_point`
     - `official_policy_set`
     - `exploratory_policy_set`
     - `baseline_policy_set`

3. `scripts/phase_morl/manifests/phase4_ablation_manifest.json`
   - 在六场景主矩阵完成后生成
   - 记录最终锚点、full weights、两组 ablation weights、训练 seed、初始化 checkpoint 等

### 4.4 证据就绪度分层

| 层级 | 当前集合 | 用途 | 当前就绪度 |
| --- | --- | --- | --- |
| A 层（确认层） | `P1/P2/P3/P4/P10 × 3 seeds` | 官方场景主表、官方 Pareto/HV | 高 |
| B 层（探索层） | `P5/P6/P7/P8/P9 × seed42` | exploratory overlay、trade-off 补点 | 中 |
| C 层（消融层） | `anchor-full` 对比 `anchor-no-energy` / `anchor-no-smooth` | 局部必要性验证 | 低，待锚点冻结 |
| D 层（外部对照） | rough baseline `seed42/43/44` | MORL vs 非 MORL 对照 | 中，资产已存在但待归一化接入 |

说明：

- [Fact] A 层当前已经完整，B 层当前仍然只有单种子。
- [Fact] 当前干净 `S1` Pareto front 是 `P1/P2/P3`，不再是 3 月 23 日旧结果里的 `P9/P10`。
- [Inference + Evidence] 因此本版计划不再把 “`P10` 已在 Pareto front 上” 当作消融锚点的事实前提。
- [Inference + Evidence] 如果最终仍保留 `P10` 作为消融锚点，其理由只能是“六场景后它仍是最可解释的平衡点”，而不是“它在旧版或当前单场景里天然成立”。

---

## 5. 六场景定义与当前状态

### 5.1 场景定义

| 场景 ID | 名称 | terrain_mode | command_vx | disturbance_mode | analysis_group |
| --- | --- | --- | --- | --- | --- |
| `S1` | flat_mid_speed | `plane` | `1.0` | `none` | `main` |
| `S2` | flat_high_speed | `plane` | `1.5` | `none` | `stress` |
| `S3` | uphill_20deg | `slope_up` | `0.8` | `none` | `main` |
| `S4` | downhill_20deg | `slope_down` | `0.6` | `none` | `main` |
| `S5` | stairs_15cm | `stairs_15cm` | `0.5` | `none` | `main` |
| `S6` | lateral_disturbance | `plane` | `0.8` | `velocity_push_equivalent` | `stress` |

### 5.2 当前场景状态

- [Fact] `S1/S4/S6` 的真实 smoke 已在 2026-03-27 完成，说明场景注入链本身已经接上 Isaac Lab。
- [Fact] 当前干净 v2 family 只完成了 `S1` 的全量正式评估；`S2/S3/S4/S5/S6` 还没有用干净 v2 family 跑过正式矩阵。
- [Fact] `S6` 当前官方可复跑实现是 `velocity_push_equivalent`。
- [Don't know] 如果论文或导师要求 `S6` 必须严格落到 `100N` 力扰动，当前实现是否需要升级，仍要看后续要求与 Isaac Lab 事件 API 可行性。

---

## 6. 指标、统计口径与分析规则

### 6.1 主指标

沿用阶段三已冻结的 4 个主指标，统一按“越小越好”分析：

- `J_speed`
- `J_energy`
- `J_smooth`
- `J_stable`

### 6.2 辅助指标

- `success_rate`
- `mean_vx`
- `mean_base_contact_rate`
- `mean_timeout_rate`
- `recovery_time`（重点关注 `S6`）
- `pass_flag`（工程质检，不进入 Pareto 维度）

### 6.3 官方分析单元

- 官方主分析：A 层的 `policy-level` 聚合结果。
- exploratory overlay：B 层单种子点。
- 外部对照：baseline 单独成表，不混入 MORL 官方 HV。
- checkpoint-level：只用于 QC 与 seed 稳健性，不直接写成官方主结论。

### 6.4 统计口径冻结

1. **场景内主表**
   - A 层和 baseline：统一用 `across-seed mean ± std`
   - B 层：只报单点值，不报 `± std`

2. **跨场景 summary**
   - 统一输出：
     - `mean across scenes`
     - `best-scene`
     - `worst-scene`
     - `front-membership count`
   - 不输出“六场景混合 `± std`”

3. **Pareto/HV**
   - 单场景官方 Pareto/HV：只用 A 层 `policy-level` 结果
   - B 层只做 overlay
   - baseline 单独做对照表和对照图，不进入 MORL 官方 HV

4. **seed 稳健性标记**
   - 若 A 层某策略在某场景出现明显 seed 级塌缩，则该策略在该场景的官方结论中必须显式标记 `unstable`
   - 即便其 `policy-level mean` 看起来较优，也不能直接写成“稳健非支配解”

5. **bootstrap Pareto 稳健性**
   - 对 A 层 `policy-level` 结果补做 bootstrap Pareto 分析
   - 重采样必须尊重 seed 分组，不允许把 scene 波动和 seed 波动混在一起
   - 至少输出：
     - `P(on_front)`
     - `HV` 的 bootstrap 区间
   - 在论文和答辩材料中，不直接把单次 front 写成“稳健 Pareto 最优”，而是结合 bootstrap front frequency 表述

### 6.5 当前 `S1` 参考结论

- [Fact] 2026-04-04 当前干净 `S1` 的 `policy-level Pareto front = P1,P2,P3`。
- [Fact] 2026-04-04 当前干净 `S1` 的 `Hypervolume = 0.2666`。
- [Fact] `P2` 在 `J_energy`、`J_smooth`、`J_stable` 上表现强，`P1` 在 `J_speed` 上最优，`P3` 在 `J_stable` 上最优。
- [Inference + Evidence] 阶段四的多场景主线，至少应把 `P1/P2/P3` 视为当前已被 `S1` 支持的官方 shortlist；`P10` 是否保留为最终解释性锚点，要等六场景主矩阵结果。

### 6.6 边界冻结规则

- [Fact] 当前 `analyze_pareto.py` 仍使用硬编码 `normalization_bounds` 与 `ref_point`。
- [Inference + Evidence] 因此阶段四不能继续把“边界冻结”只写在文档里，而必须把它外化到 `phase4_analysis_config.json`。
- `S1` 的当前干净结果作为第一个已知样本。
- `S2-S6` 的边界在预跑 smoke 完成后冻结一次，写入 `phase4_analysis_config.json`，冻结后不得因正式结果回改。
- `S4-2` 的预跑必须显式覆盖 `S5` 与 `S6`。
- 若任一主指标在预跑中超出当前候选上界，或已接近候选上界到足以引发 clip 失真，则必须先更新 `phase4_analysis_config.json` 再启动正式矩阵。
- 冻结时必须记录每个场景的主指标最大观测值，作为后续 QC 与论文附录的可追溯证据。
- 不定义新的官方 `HV(global)`；官方层面只保留 `HV(S1)...HV(S6)` 与跨场景 descriptive summary。

---

## 7. 阶段四分步骤执行

### S4-0 建立 manifest 驱动的正式执行链

**目标**

- 让 MORL、baseline、ablation 都通过 manifest 进入同一条正式链。

**本步必须完成的工程改动**

- 修改 `run_full_eval_matrix.py`：从“目录名正则发现 run”改为“读取 `phase4_main_manifest.json`”
- 修改 `analyze_pareto.py` 或新增 `analyze_phase4_pareto.py`：从“目录名正则发现 run”改为“读取 manifest / 聚合 CSV”
- 新建：
  - `scripts/phase_morl/manifests/phase4_main_manifest.json`
  - `scripts/phase_morl/manifests/phase4_analysis_config.json`
- 新增单测：
  - `tests/unit/test_phase4_manifest.py`
  - `tests/unit/test_run_full_eval_matrix_manifest.py`
  - `tests/unit/test_phase4_analysis_config.py`

**通过标准**

- `run_full_eval_matrix.py --manifest ... --dry-run` 可正确列出 MORL 与 baseline 条目
- `analyze_pareto.py` 或替代脚本不再依赖 `morl_pX_seedY` 正则
- 若计划复用现有 `S1` 的 20 个 JSON，必须先完成 schema 验证：
  - 比对现有 `S1` JSON 字段与聚合层所需字段
  - 若字段不足，则在“补字段”与“重跑 S1”之间二选一后再进入 `S4-3`

### S4-1 完成 baseline 归一化接入

**目标**

- 将 baseline 3 seeds 变成与 MORL 主矩阵同等级的正式输入。

**本步必须完成的内容**

- 在 `phase4_main_manifest.json` 中为 baseline 显式冻结：
  - `canonical_seed`
  - `run_dir`
  - `checkpoint=model_1499.pt`
  - `output_stem=baseline_seed42/43/44`
  - `source_state=active/archive`
- 不再依赖目录名差异去推断 seed 身份

**新增测试**

- `tests/unit/test_phase4_baseline_manifest.py`

**通过标准**

- baseline 3 条 manifest 记录都能通过 dry-run
- baseline seed42/43/44 的输出命名统一为 `baseline_seed*_S*.json`

### S4-2 跑剩余场景预跑并冻结分析配置

**目标**

- 在启动全矩阵之前，用少量代表点把 `S2-S6` 跑通，并正式冻结 `analysis_config`。

**建议预跑集合**

- `P1 seed42`
- `P2 seed42`
- `P10 seed42`
- `baseline seed42`

**预跑矩阵**

- `4 runs × 5 scenes = 20` 次评估

**本步关注点**

- `scenario_id / terrain_mode / cmd_vx / disturbance_mode` 是否写对
- `S2/S3/S5/S6` 是否都能稳定产出 JSON
- `S6` 是否确实带扰动
- `eval_steps=3000` 是否足以累计稳定统计
- `S5/S6` 的主指标是否触发归一化边界外溢或 clip 风险

**产出**

- `20` 个预跑 JSON
- 冻结版 `phase4_analysis_config.json`
- `preflight_metric_maxima.json`

**通过标准**

- `20/20` JSON 生成成功
- `analysis_config` 冻结一次并入库

### S4-3 跑 MORL v2 六场景主矩阵

**目标**

- 先完成 MORL 主体结果，不等消融。

**正式集合**

- A 层：`P1/P2/P3/P4/P10 × seed42/43/44 = 15`
- B 层：`P5/P6/P7/P8/P9 × seed42 = 5`

**矩阵规模**

- 若直接复用现有 `S1`：新增 `20 × 5 = 100` 次评估
- 若因 schema 或 analysis_config 变化需要重跑 `S1`：总计 `20 × 6 = 120` 次评估

**通过标准**

- MORL 主矩阵完整
- `S1` 结果来源单一，不出现“旧退化数据 + 新干净数据”混写

### S4-4 跑 baseline 六场景对照矩阵

**目标**

- 补齐外部对照组，否则阶段四只剩 MORL 内部互比。

**矩阵规模**

- `3 baseline seeds × 6 scenes = 18` 次评估

**建议输出**

- `baseline_seed42_S1.json` ... `baseline_seed44_S6.json`

**通过标准**

- baseline `18` 个 JSON 全部落到阶段四正式结果目录
- baseline 可进入后续聚合与对照表，但不进入 MORL 官方 HV

### S4-5 建立聚合层与 QC 层

**目标**

- 在 Pareto/HV 分析之前，先把正式链的表、质检和分层做完。

**新增脚本**

- `scripts/phase_morl/aggregate_phase4_results.py`
- `scripts/phase_morl/check_phase4_qc.py`

**新增测试**

- `tests/unit/test_phase4_aggregate_results.py`
- `tests/unit/test_phase4_qc.py`

**最低产出**

- `checkpoint_level.csv`
- `policy_level_confirmatory.csv`
- `policy_level_exploratory.csv`
- `baseline_control.csv`
- `qc_report.md`

**QC 必查项**

- 是否混入旧 `phase_morl` family
- `S1` 是否全部来自干净 family
- manifest 中的 `policy_id / canonical_seed / checkpoint / output_stem` 是否一致
- `P5-P9` 是否被强制标记为 exploratory
- 是否存在场景元数据缺字段、错字段、零命令回退

### S4-6 做六场景 Pareto/HV，并冻结消融锚点

**目标**

- 在主矩阵和 baseline 对照齐备后，再冻结最终消融锚点。

**官方分析**

- 六个场景各自产出一个官方 Pareto/HV
- 官方 Pareto/HV 只用 A 层 `policy-level` 结果
- B 层只做 overlay
- baseline 只做对照图和对照表
- 补做 bootstrap Pareto 稳健性分析，只针对 A 层 confirmatory set

**新增或扩展脚本**

- `scripts/phase_morl/analyze_phase4_pareto.py` 或扩展后的 `analyze_pareto.py`
- `scripts/phase_morl/render_phase4_figures.py`

**新增测试**

- `tests/unit/test_phase4_pareto_manifest.py`
- `tests/unit/test_phase4_render_figures.py`

**本步必须新增的产出**

- `confirmatory_scenario_hv.json`
- `front_membership_frequency.csv`
- `robustness_summary.csv`
- `bootstrap_front_membership.csv`
- `bootstrap_hv_ci.json`
- `phase4_hv_bar.png`
- `phase4_pareto_s1.png` ... `phase4_pareto_s6.png`
- `phase4_ablation_manifest.json`

**锚点冻结规则**

1. 若 `P10` 在六场景 summary 中仍然是最可解释的平衡点，则可继续作为锚点
2. 若 `P10` 在多数核心场景中被明显支配，则在 A 层中选择更合理的锚点
3. 锚点一旦写入 `phase4_ablation_manifest.json`，后续不得回改

### S4-7 泛化消融 runner，并完成消融训练

**目标**

- 将当前 `P10` 专用消融脚本改成“读取 ablation manifest 的通用 runner”。

**本步必须完成的工程改动**

- 修改 `run_morl_ablation.py`：从写死 `P10-no-energy / P10-no-smooth` 改为读取 `phase4_ablation_manifest.json`
- 若锚点不是 `P10`，按 manifest 自动生成：
  - `anchor-full`
  - `anchor-no-energy`
  - `anchor-no-smooth`

**新增测试**

- `tests/unit/test_run_morl_ablation_manifest.py`
- `tests/unit/test_phase4_ablation_weights.py`

**训练要求**

- 默认 `2 ablations × 3 seeds = 6` 次训练
- 每个 run 训练后先做 `S1 + --skip_ros2` smoke 验收

**验收底线**

- `mean_vx > 0.8`
- `success_rate` 保持可用
- 不出现 2026-04-02 那种 seed43/44 全面退化

### S4-8 跑消融六场景评估与聚合

**目标**

- 把消融组放进与主矩阵完全一致的正式链。

**矩阵规模**

- 消融组：`2 × 3 × 6 = 36` 次评估
- 对照组：复用锚点 full model 的 `3 × 6 = 18` 次评估

**通过标准**

- 消融与 full anchor 能按场景逐列对比
- 聚合层与 QC 层能识别 `family=ablation`
- 能明确回答“去掉能效目标/平滑目标后，哪个主指标先恶化、是否出现副作用”

### S4-9 收敛论文与答辩材料

**最低产物**

- 表 1：A 层 5 个策略的六场景主指标表
  - 每格定义为：`across-seed mean ± std`
- 表 2：B 层 exploratory 补点表
  - 只报单点，不报 `± std`
- 表 3：baseline 对照表
  - 每格定义为：`across-seed mean ± std`
- 表 4：消融对照表
  - 每格定义为：`across-seed mean ± std`
- 图 1：六场景 Pareto 总览
- 图 2：六场景 HV 柱状图
- 图 3：策略-场景热力图
- 图 4：消融对比图

**图表升级要求**

- 单场景 Pareto 图：关键 trade-off 点必须有文字标注
- baseline：在对照图或对照表中必须显式出现
- 多场景：至少有 1 张总览型热力图或矩阵图，避免只给单场景图
- 误差表达：在适合的 2D 对照图中可加入 `±1σ` 误差棒；若图面过载，则改用表格承载误差

**文字结论最低要回答**

1. `P1/P2/P3/P4/P10` 在六场景下分别擅长什么
2. 哪些策略在场景级 4 维指标上形成稳健非支配解
3. MORL 相对 rough baseline 是否有清晰增益，以及增益在哪些场景成立
4. 去掉能效或平滑目标后，会破坏哪些行为特征
5. 最终论文推荐策略是谁，理由是什么

**表述约束**

- 当某策略只在单次 point estimate 下进入 front，而 bootstrap front frequency 较低时，不写成“稳健非支配解”
- 官方结论优先使用：
  - `front_membership count`
  - `bootstrap P(on_front)`
  - 是否存在 `unstable` 场景标记

**论文章节叙事结构**

1. 策略性能画像（事实层）
2. Trade-off 机制分析（机制层）
3. MORL vs baseline 的定位与增益边界（对比层）
4. 消融解释（因果层）
5. 最终推荐与适用场景（决策层）

**SOTA 叙事定位**

- 不新增 PGMORL / MORL-D 复现实验作为阶段四主任务
- 但在论文与答辩材料中必须增加一段方法定位说明：
  - 当前方法属于离散权重 profile 的可解释型 MORL 路线
  - 与连续权重自适应类方法相比，重点是可复现、可解释、易做工程部署与局部消融
  - 当前贡献聚焦于 curriculum 化训练链路、reward-terrain trap 规避、以及场景级 confirmatory evaluation

**手动 reward tuning 的处理方式**

- 不将现有 flat 手动调优 checkpoint 直接并入当前阶段四主 manifest
- `reward_engineering.md` 中已存在的手动调权重实验，只作为论文背景与替代方案讨论素材
- 若后续需要与手动调优做同口径公平对照，必须新增 rough/manual-tuning 专门实验包，而不是复用现有 flat checkpoint

**消融训练协议说明**

- 当前默认协议：从 baseline warm-start 重新训练 ablation 模型，不是从锚点 full model 继续微调
- 若后续要改成“从锚点 full model 微调”，必须在 `phase4_ablation_manifest.json` 中显式记录，并单独说明与当前协议不同

---

## 8. 测试与验证规则

### 8.1 TDD 适用范围

- `run_full_eval_matrix.py` 的 manifest 化
- baseline 归一化 manifest
- analysis config 外化
- 聚合层/QC 层脚本
- 消融 runner 泛化
- Pareto/HV 分析脚本的 manifest 化

### 8.2 最低验证命令

```powershell
pytest -q tests/unit/test_phase4_manifest.py
pytest -q tests/unit/test_run_full_eval_matrix_manifest.py
pytest -q tests/unit/test_phase4_baseline_manifest.py
pytest -q tests/unit/test_phase4_aggregate_results.py tests/unit/test_phase4_qc.py
pytest -q tests/unit/test_run_morl_ablation_manifest.py tests/unit/test_phase4_ablation_weights.py
python scripts/phase_morl/run_full_eval_matrix.py --manifest scripts/phase_morl/manifests/phase4_main_manifest.json --dry-run
python scripts/phase_morl/aggregate_phase4_results.py --manifest scripts/phase_morl/manifests/phase4_main_manifest.json --dry-run
python scripts/phase_morl/analyze_phase4_pareto.py --manifest scripts/phase_morl/manifests/phase4_main_manifest.json --scenario S1
```

### 8.3 真实 smoke 最低要求

- 1 条 MORL manifest 记录 smoke
- 1 条 baseline manifest 记录 smoke
- 1 条 ablation manifest 记录 smoke
- 1 条聚合层 + Pareto 层贯通 smoke

---

## 9. 新版排期

| 日期 | 任务 | 产出 |
| --- | --- | --- |
| 2026-04-04 至 2026-04-05 | `S4-0` manifest 主链 + baseline 归一化 + 单测 | `phase4_main_manifest.json`、`phase4_analysis_config.json`、manifest 单测 |
| 2026-04-06 | `S4-2` 预跑 `S2-S6` + 冻结 analysis config | `20` 个预跑 JSON、冻结版 `analysis_config` |
| 2026-04-07 至 2026-04-08 | `S4-3` MORL 主矩阵 | `S2-S6` 正式 JSON（必要时重跑 `S1`） |
| 2026-04-09 | `S4-4` baseline 对照矩阵 | baseline `18` 个 JSON |
| 2026-04-10 至 2026-04-11 | `S4-5` 聚合层 + QC 层 + 单测 | `checkpoint_level.csv`、`policy_level_confirmatory.csv`、`baseline_control.csv`、`qc_report.md` |
| 2026-04-12 | `S4-6` 六场景 Pareto/HV + bootstrap 稳健性 + 锚点冻结 | 场景 Pareto/HV、`front_membership_frequency.csv`、`bootstrap_hv_ci.json`、`phase4_ablation_manifest.json` |
| 2026-04-13 | `S4-7` 消融 runner 泛化 + 单测 + `S1` smoke | 通用 `run_morl_ablation.py`、消融 smoke 结果 |
| 2026-04-14 至 2026-04-15 | `S4-7` 消融训练 + `S4-8` 消融评估 | `6` 个 ablation checkpoint、`36` 个消融 JSON |
| 2026-04-16 至 2026-04-20 | `S4-9` 论文图表与结论整理 | 论文主表、图、结论草稿 |

---

## 10. 风险与对策

| 风险 | 影响 | 对策 |
| --- | --- | --- |
| 旧 `phase_morl` 与新 `phase_morl_v2` 混写 | 主表失真 | 所有正式输出统一由 manifest 驱动，按 `family` 显式分层 |
| baseline seed43/44 在 archive，seed42 命名不同 | baseline 身份混乱 | 用 `phase4_main_manifest.json` 显式冻结 `canonical_seed / checkpoint / output_stem` |
| 当前 eval/pareto 脚本仍依赖 `morl_pX_seedY` | baseline 与 ablation 无法进入正式链 | 先完成 manifest 化，再跑正式数据 |
| `P10` 在六场景后不再适合作锚点 | 现有消融 runner 失效 | 将 `run_morl_ablation.py` 泛化为 manifest 驱动，而不是写死 `P10` |
| `S6` 严格 `100N` 仍未落地 | 结果表述可能被质疑 | 第一版主结果写成 `velocity_push_equivalent`；若后续必须 `100N`，单独追加补充实验 |
| `S3/S5` 场景中 MORL 策略明显失效 | 推荐结论只能覆盖部分场景 | 在 `S4-6` 中显式标记 `non-generalizable` 场景，不将其作为推荐策略的支撑场景 |
| B 层只有单 seed | 容易误写成强结论 | 统一标 exploratory，只做 overlay，不报 `± std` |
| 场景内/跨场景统计混淆 | 结论易被误读 | 场景内只报 `across-seed mean ± std`；跨场景不用混合 `± std` |
| 5 个 A 层策略在 4 维 Pareto 空间中对噪声敏感 | 单次 front 结论不稳 | 对 confirmatory set 补做 bootstrap front frequency 与 HV 区间分析 |
| 旧 `S1` JSON schema 与 manifest 化聚合层不一致 | 复用旧结果失败 | 在 `S4-0` 先做 schema 验证，不通过则补字段或重跑 `S1` |
| `S5/S6` 指标超出候选边界导致 clip | Pareto/HV 排序失真 | 在 `S4-2` 预跑中验证并先更新 `analysis_config`，再跑正式矩阵 |
| 消融训练再次暴露 seed43/44 敏感性 | 消融结论不稳定或无法完成 | 在 `S4-7` 的 `S1` smoke 中设硬验收阈值；若 seed43/44 明显退化，先暂停调查再继续 |
| 聚合/QC/图表脚本缺失 | 排期失真 | 将其单列为工程任务和测试任务，不再隐含在分析步骤里 |

---

## 11. 阶段四完成判据

- `phase4_main_manifest.json`、`phase4_analysis_config.json`、`phase4_ablation_manifest.json` 全部冻结并可追溯。
- MORL v2 主矩阵六场景结果完整，至少形成 `20 × 6` 的可追溯结果集。
- baseline 对照组六场景结果完整，形成 `3 × 6` 的同口径外部对照。
- 聚合层与 QC 层产物全部生成，并通过单测与真实 smoke。
- 官方六场景 Pareto/HV 全部完成，且没有新的混合 `HV(global)`。
- bootstrap front frequency 与 `HV` 区间全部生成，并能支持“稳健非支配解”的表述。
- `P5-P9` 只出 exploratory 补图，不混入官方主表。
- 通用消融 runner、两组消融训练、`S1` 快速验收、六场景正式评估全部完成。
- 论文第 5 章和答辩 PPT 所需表图都可回溯到具体 manifest、JSON、CSV 与分析配置。
- 最终结论能明确区分：
  - [Fact] 已被直接数据支持的结论
  - [Inference + Evidence] 基于多场景模式做出的解释
  - [Assumption] 尚未被额外实验加固的工作假设
  - [Don't know] 当前仍未被验证的问题

---

## 12. 备注

- [Inference + Evidence] 2026-04-04 更适合作为当前阶段四的正式执行基线，而不是整个阶段四工作的唯一时间起点。
- [Fact] `S1` 当前已经不是“待做事项”，而是新的阶段四参考场景。
- [Assumption] 如果 `S1` 现有 JSON 的 schema 已满足 manifest 化后的聚合需求，则可以直接复用，避免重复仿真。
- [Assumption] baseline 归档目录中的 seed43/44 资产内容完整，可通过 manifest 正常接回正式链。
- [Don't know] 六场景汇总后，最终最值得做局部消融的是不是 `P10`，目前还不能提前写死。
