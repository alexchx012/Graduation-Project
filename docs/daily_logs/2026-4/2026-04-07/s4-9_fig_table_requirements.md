# S4-9 图表来源与制作要求（4 图 + 4 表）

## 1. 目的

本文件用于冻结 `S4-9` 论文/答辩材料中的 **4 个图** 和 **4 个表** 的数据来源、展示要求与制作约束，
避免后续在写作阶段重复讨论“每个图表该从哪里取数、该怎么画/怎么排”。

对应计划来源：

- `docs/daily_logs/2026-4/2026-04-04/phase4_detailed_plan.md`
- `docs/daily_logs/2026-4/2026-04-04/phase4_plan_review.md`

## 2. 通用约束

### 2.1 场景顺序

所有图表统一使用以下场景顺序：

1. `S1` `flat_mid_speed`
2. `S2` `flat_high_speed`
3. `S3` `uphill_20deg`
4. `S4` `downhill_20deg`
5. `S5` `stairs_15cm`
6. `S6` `lateral_disturbance`

### 2.2 策略分组口径

- **A 层 confirmatory set**：`P1 / P2 / P3 / P4 / P10`
- **B 层 exploratory set**：`P5 / P6 / P7 / P8 / P9`
- **baseline**：`baseline`
- **ablation**：`anchor-full(P10) / P10-no-energy / P10-no-smooth`

### 2.3 统计口径

- 表 1、表 3、表 4：使用 `across-seed mean ± std`
- 表 2：只报单点，不报 `± std`
- 官方 Pareto/HV 只使用 **A 层 confirmatory set**
- B 层只能作为 `exploratory` 补点，不能混入官方 front/HV
- baseline 必须显式保留 `seed43 = degenerate archived control` 的说明
- 消融解释只写 **local necessity**，不写“充分性”或“联合最优”

### 2.4 统一字段优先级

优先使用以下字段：

- 速度跟踪：`mean_vx_meas`、`mean_vx_abs_err`、`J_speed`
- 能效：`J_energy`
- 平滑：`J_smooth`
- 稳定/安全：`J_stable`、`success_rate`
- 恢复：`recovery_time`

### 2.5 当前产物就绪度

| 编号 | 名称 | 主要数据源 | 当前状态 |
| ---- | ---- | ---------- | -------- |
| 表 1 | A 层 5 策略六场景主指标表 | `logs/eval/phase_morl_v2/aggregated/policy_level_confirmatory.csv` | 数据已就绪 |
| 表 2 | B 层 exploratory 补点表 | `logs/eval/phase_morl_v2/aggregated/policy_level_exploratory.csv` | 数据已就绪 |
| 表 3 | baseline 对照表 | `logs/eval/phase_morl_v2/aggregated/baseline_control.csv` | 数据已就绪 |
| 表 4 | 消融对照表 | `logs/eval/phase_morl_v2/ablation/ablation_comparison.csv` | 数据已就绪 |
| 图 1 | 六场景 Pareto 总览 | `docs/figures/phase4_pareto_s1.png` ~ `phase4_pareto_s6.png` | 基础子图已就绪 |
| 图 2 | 六场景 HV 柱状图 | `docs/figures/phase4_hv_bar.png` / `confirmatory_scenario_hv.json` | 已就绪 |
| 图 3 | 策略-场景热力图 | `policy_level_confirmatory.csv` + `policy_level_exploratory.csv` | 需新增排版/绘制 |
| 图 4 | 消融对比图 | `ablation_comparison.csv` | 需新增排版/绘制 |

---

## 3. 四个表

### 表 1：A 层 5 个策略的六场景主指标表

**信息来源**

- 主数据：
  - `logs/eval/phase_morl_v2/aggregated/policy_level_confirmatory.csv`
- 可选注释数据：
  - `logs/eval/phase_morl_v2/pareto/front_membership_frequency.csv`
  - `logs/eval/phase_morl_v2/pareto/robustness_summary.csv`

**核心字段**

- 主键：`policy_id`、`scenario_id`
- 主要数值：
  - `mean_vx_meas`
  - `mean_vx_abs_err`
  - `J_speed`
  - `J_energy`
  - `J_smooth`
  - `J_stable`
  - `success_rate`
- 波动字段：
  - 对应的 `*_std`

**做表要求**

- 只允许出现 A 层 5 个策略：`P1/P2/P3/P4/P10`
- 六个场景必须全部出现，顺序固定为 `S1-S6`
- 每个数值单元必须写成 `mean ± std`
- 表中至少要保留 4 个主指标：
  - `J_speed`
  - `J_energy`
  - `J_smooth`
  - `J_stable`
- 若版面允许，建议同时保留 `mean_vx_meas`
- 不允许把 B 层 exploratory 点混入表 1
- 若需要在表内提示“稳健非支配”，必须引用：
  - `front_count`
  - `bootstrap_mean_p_on_front`
  - `unstable_scenarios`
  不能只根据单次 point estimate 标注

**推荐排版**

- 行：场景 `S1-S6`
- 列：5 个策略
- 每个策略下的子列：`mean_vx_meas / J_speed / J_energy / J_smooth / J_stable`

### 表 2：B 层 exploratory 补点表

**信息来源**

- 主数据：
  - `logs/eval/phase_morl_v2/aggregated/policy_level_exploratory.csv`

**核心字段**

- `policy_id`
- `scenario_id`
- `mean_vx_meas`
- `J_speed`
- `J_energy`
- `J_smooth`
- `J_stable`
- `success_rate`

**做表要求**

- 只允许出现 B 层 5 个策略：`P5/P6/P7/P8/P9`
- 明确标注为 `exploratory only` 或 `not included in official front/HV`
- 由于当前只有单 seed，**不写 `± std`**
- 不允许把表 2 的结果写成“官方 Pareto 结论”
- 表 2 的作用是补充说明：
  - B 层是否提供新的 trade-off 点
  - B 层是否只适合作为 overlay

**推荐排版**

- 行：`P5-P9`
- 列：`S1-S6`
- 每格：单点主指标摘要，或分场景展开为单行

### 表 3：baseline 对照表

**信息来源**

- 主数据：
  - `logs/eval/phase_morl_v2/aggregated/baseline_control.csv`
- 补充说明：
  - `logs/eval/phase_morl_v2/aggregated/qc_report.md`

**核心字段**

- `scenario_id`
- `mean_vx_meas`
- `mean_vx_abs_err`
- `J_speed`
- `J_energy`
- `J_smooth`
- `J_stable`
- `success_rate`
- `degenerate_seed_ids`
- `narrative_effective_seed_ids`
- `paper_note`

**做表要求**

- 必须使用 `across-seed mean ± std`
- 必须显式保留 `seed43` 的退化说明
- 不能把 baseline 当成“3 个健康 seeds 的稳定均值”
- 表中或表下注必须出现：
  - `degenerate archived control`
  - `effective narrative should use seed(s) 42,44`
- 如果版面紧张，至少保留：
  - `mean_vx_meas`
  - `J_speed`
  - `J_energy`
  - `success_rate`
  - 一列 `note`

**推荐排版**

- 行：`S1-S6`
- 列：`mean_vx_meas / J_speed / J_energy / J_smooth / J_stable / success_rate / note`

### 表 4：消融对照表

**信息来源**

- 主数据：
  - `logs/eval/phase_morl_v2/ablation/ablation_comparison.csv`
- 补充数据：
  - `logs/eval/phase_morl_v2/ablation/policy_level_ablation.csv`
  - `logs/eval/phase_morl_v2/ablation/qc_report.md`

**核心字段**

- 主键：
  - `scenario_id`
  - `ablation_policy_id`
- anchor 侧：
  - `anchor_mean_vx_meas`
  - `anchor_J_speed`
  - `anchor_J_energy`
  - `anchor_J_smooth`
  - `anchor_J_stable`
- ablation 侧：
  - `ablation_mean_vx_meas`
  - `ablation_J_speed`
  - `ablation_J_energy`
  - `ablation_J_smooth`
  - `ablation_J_stable`
- 差值侧：
  - `delta_mean_vx_meas`
  - `delta_J_speed`
  - `delta_J_energy`
  - `delta_J_smooth`
  - `delta_J_stable`

**做表要求**

- 必须同时展示：
  - `anchor-full (P10)`
  - `P10-no-energy`
  - `P10-no-smooth`
- 统计口径必须为 `across-seed mean ± std`
- 表中必须出现 `delta`，不能只摆绝对值
- 重点回答两个问题：
  - 去掉 `energy` 后，哪些行为先恶化
  - 去掉 `smooth` 后，哪些行为先恶化
- 优先保留以下指标：
  - `mean_vx_meas`
  - `J_speed`
  - `J_energy`
  - `J_smooth`
  - `J_stable`
- 若场景恢复行为重要，可追加 `recovery_time`
- 表述上必须使用“necessity test”语气，不能写成“证明最优”

**推荐排版**

- 行：`S1-S6 × 2 ablations`
- 列：`anchor`、`ablation`、`delta`
- 每个指标单独一列，或拆为正文主表 + 附表

---

## 4. 四个图

### 图 1：六场景 Pareto 总览

**信息来源**

- 现成基础子图：
  - `docs/figures/phase4_pareto_s1.png`
  - `docs/figures/phase4_pareto_s2.png`
  - `docs/figures/phase4_pareto_s3.png`
  - `docs/figures/phase4_pareto_s4.png`
  - `docs/figures/phase4_pareto_s5.png`
  - `docs/figures/phase4_pareto_s6.png`
- 底层数据：
  - `logs/eval/phase_morl_v2/aggregated/policy_level_confirmatory.csv`
  - `logs/eval/phase_morl_v2/pareto/confirmatory_scenario_hv.json`
- 生成脚本：
  - `scripts/phase_morl/analyze_phase4_pareto.py`

**做图要求**

- 该图应作为 **一个编号图** 出现，但内部使用 `2 × 3` 子图布局
- 子图顺序固定为 `S1-S6`
- 只显示 A 层 confirmatory set：`P1/P2/P3/P4/P10`
- 每个场景子图必须有：
  - 场景名
  - 坐标轴标签
  - Pareto 成员高亮
- 关键 trade-off 点必须加文字标注，至少覆盖：
  - `P1`
  - `P2`
  - `P3`
  - `P10`
- 若图面过载，不强制加入误差棒；误差信息可转由表格承载
- 图注必须说明：
  - 官方 front/HV 只基于 A 层
  - B 层 exploratory 不在该图中承担官方结论角色

### 图 2：六场景 HV 柱状图

**信息来源**

- 主数据：
  - `logs/eval/phase_morl_v2/pareto/confirmatory_scenario_hv.json`
- 可选误差数据：
  - `logs/eval/phase_morl_v2/pareto/bootstrap_hv_ci.json`
- 现成图：
  - `docs/figures/phase4_hv_bar.png`
- 生成脚本：
  - `scripts/phase_morl/analyze_phase4_pareto.py`

**做图要求**

- 横轴固定为 `S1-S6`
- 纵轴为 `Hypervolume`
- 推荐在柱顶或图注中标出具体 HV 数值
- 推荐加入 bootstrap `95% CI`
  - 若加入误差棒，来源必须是 `bootstrap_hv_ci.json`
- 颜色风格要统一，不按策略上色
- 图注必须明确：
  - HV 只基于 A 层 confirmatory set
  - 不能出现混合 A/B 层的 `HV(global)` 表述

### 图 3：策略-场景热力图

**信息来源**

- 主数据：
  - `logs/eval/phase_morl_v2/aggregated/policy_level_confirmatory.csv`
  - `logs/eval/phase_morl_v2/aggregated/policy_level_exploratory.csv`
- 可选稳健性注释：
  - `logs/eval/phase_morl_v2/pareto/bootstrap_front_membership.csv`
  - `logs/eval/phase_morl_v2/pareto/front_membership_frequency.csv`
  - `logs/eval/phase_morl_v2/pareto/robustness_summary.csv`

**做图要求**

- 该图建议做成 `2 × 2` 面板热力图
- 四个面板分别对应：
  - `J_speed`
  - `J_energy`
  - `J_smooth`
  - `J_stable`
- 行为 `10` 个策略：`P1-P10`
- 列为 `6` 个场景：`S1-S6`
- 必须显式区分：
  - A 层 `P1/P2/P3/P4/P10`
  - B 层 `P5-P9`
- 建议做法：
  - A 层使用正常标签
  - B 层用浅色标签、虚线框或脚注标明 `exploratory`
- 每个面板独立颜色尺度，且图注中说明“数值不可跨指标直接比大小”
- 色彩语义必须统一为：
  - **更优（更小）** 的 `J_*` 数值对应更显著的“好”色
- 如果需要在热力图上叠加稳健信息，优先叠加：
  - `P(on_front)` 或 `front_count`
  不要发明新的综合分数当作官方结论

### 图 4：消融对比图

**信息来源**

- 主数据：
  - `logs/eval/phase_morl_v2/ablation/ablation_comparison.csv`
- 补充数据：
  - `logs/eval/phase_morl_v2/ablation/policy_level_ablation.csv`
  - `logs/eval/phase_morl_v2/ablation/qc_report.md`

**做图要求**

- 图 4 的核心任务是解释：
  - 去掉 `energy` 目标后，哪些行为先恶化
  - 去掉 `smooth` 目标后，哪些行为先恶化
- 推荐做成 `2 × 2` 多面板图
- 最低建议面板：
  1. `delta_mean_vx_meas`
  2. `delta_J_energy`
  3. `delta_J_smooth`
  4. `delta_J_stable`
- 若版面允许，可额外补 `delta_J_speed`
- 横轴固定为 `S1-S6`
- 颜色固定为两组：
  - `P10-no-energy`
  - `P10-no-smooth`
- 图中必须有零基线，便于读者判断“相对 anchor-full 是变好还是变坏”
- 若加入误差棒，只能使用已有 `anchor_*_std` 与 `ablation_*_std`
  或明确注明是“描述性误差”，不能写成显著性结论
- 图注必须明确：
  - anchor 为 `P10`
  - 当前消融是 `local necessity test`
  - 不代表对 reward 结构的充分性证明

---

## 5. 建议输出落点

### 表格

- 表 1：由 `policy_level_confirmatory.csv` 整理为论文主表
- 表 2：由 `policy_level_exploratory.csv` 整理为 exploratory 附表
- 表 3：由 `baseline_control.csv` 整理为 baseline 对照表
- 表 4：由 `ablation_comparison.csv` 整理为消融对照表

### 图形

- 图 1：建议输出为 `docs/figures/phase4_pareto_overview.png`
- 图 2：已存在 `docs/figures/phase4_hv_bar.png`
- 图 3：建议输出为 `docs/figures/phase4_policy_scene_heatmap.png`
- 图 4：建议输出为 `docs/figures/phase4_ablation_comparison.png`

## 6. 结论

当前 `S4-9` 的 4 表 4 图中：

- **表 1-4 的数据源已经全部齐备**
- **图 1 和图 2 的基础产物已经存在**
- **图 3 和图 4 还需要在 `S4-9` 阶段补绘图脚本或手工排版**

因此，现阶段最合理的顺序是：

1. 先整理表 1-4
2. 直接复用或微调图 1、图 2
3. 最后新增图 3、图 4 的绘图实现
