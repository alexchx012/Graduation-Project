# Phase MORL v2 Figures

本文件夹包含 S4-9 阶段论文/答辩材料所需的 **4 个主图形文件**，用于展示 MORL v2 实验的评估结果。

所有图形采用高分辨率（DPI 150-200）生成，适合放入论文和演示文稿。

---

## 🖼️ 文件清单

| 文件名 | 大小 | 类型 | 布局 | 状态 |
|--------|------|------|------|------|
| `phase4_pareto_overview.png` | 691 KB | 图 1：六场景 Pareto 总览 | 2×3 子图 | ✅ 已生成 |
| `phase4_hv_bar.png` | 35 KB | 图 2：六场景 HV 柱状图 | 单图 | ✅ 已存在 |
| `phase4_policy_scene_heatmap.png` | 334 KB | 图 3：策略-场景热力图 | 2×2 面板 | ✅ 已生成 |
| `phase4_ablation_comparison.png` | 126 KB | 图 4：消融对比图 | 2×2 面板 | ✅ 已生成 |

**基础素材**（保留用于参考）：
- `phase4_pareto_s1.png` ~ `phase4_pareto_s6.png`（6 个单场景 Pareto 子图）

---

## 📊 图形详细说明

### 图 1：六场景 Pareto 总览

**文件名**：`phase4_pareto_overview.png`

**内容**：
- 展示 A 层 confirmatory set（P1, P2, P3, P4, P10）在 6 个场景下的 Pareto front
- 每个子图对应一个场景的 Pareto 前沿

**布局**：
- 2×3 子图（6 个场景）
- 场景顺序：S1（左上）→ S2 → S3 → S4（左下）→ S5 → S6（右下）

**场景映射**：
- S1: Flat Mid Speed（平地中速）
- S2: Flat High Speed（平地高速）
- S3: Uphill 20deg（上坡 20°）
- S4: Downhill 20deg（下坡 20°）
- S5: Stairs 15cm（楼梯 15cm）
- S6: Lateral Disturbance（侧向扰动）

**图形参数**：
- 尺寸：24×16 英寸
- DPI：200
- 子图间距：0.15（紧凑布局）
- 边距：left=0.05, right=0.98, top=0.95, bottom=0.05

**数据来源**：
- 组合自 6 个单场景 Pareto 子图（phase4_pareto_s1~s6.png）
- 原始数据：`logs/eval/phase_morl_v2/aggregated/policy_level_confirmatory.csv`

**论文使用建议**：
- 建议作为正文主图（Figure 1）
- 标题：Pareto Fronts Across Six Scenarios (Confirmatory Set)
- 说明：展示 A 层策略在不同场景下的 trade-off，突出 P10 的 Pareto 成员资格

---

### 图 2：六场景 HV 柱状图

**文件名**：`phase4_hv_bar.png`

**内容**：
- 展示 6 个场景的 Hypervolume（HV）对比
- 量化不同场景的 Pareto front 质量

**布局**：
- 单图
- 横轴：S1-S6
- 纵轴：Hypervolume

**特点**：
- 只基于 A 层 confirmatory set（P1-P4, P10）
- 可能包含 bootstrap 95% CI 误差棒
- 颜色统一（不按策略上色）

**数据来源**：
- `logs/eval/phase_morl_v2/pareto/confirmatory_scenario_hv.json`
- 可选误差数据：`logs/eval/phase_morl_v2/pareto/bootstrap_hv_ci.json`

**生成脚本**：
- `scripts/phase_morl/analyze_phase4_pareto.py`

**论文使用建议**：
- 建议作为正文主图（Figure 2）
- 标题：Hypervolume Comparison Across Scenarios
- 说明：HV 越高表示 Pareto front 质量越好，S2（高速）和 S5（楼梯）最具挑战性

**注意事项**：
- HV 只基于 A 层 confirmatory set
- 不能出现混合 A/B 层的 HV(global) 表述

---

### 图 3：策略-场景热力图

**文件名**：`phase4_policy_scene_heatmap.png`

**内容**：
- 展示 10 个策略（P1-P10）在 6 个场景下的 4 个目标性能
- 便于跨策略、跨场景对比单一目标的表现

**布局**：
- 2×2 面板（4 个目标）
- 左上：J_speed（速度跟踪代价）
- 右上：J_energy（能效代价）
- 左下：J_smooth（平滑代价）
- 右下：J_stable（稳定代价）

**热力图规格**：
- 行：10 个策略（P1-P10）
- 列：6 个场景（S1-S6）
- 颜色：YlOrRd_r（黄-橙-红反转，越红越优）
- 数值标注：每个单元格显示具体数值（保留 2 位小数）

**特殊标记**：
- 蓝色虚线（y=4.5）分隔 A 层和 B 层
- 左侧标注：
  - "A-layer"（蓝色，P1-P4, P10）
  - "B-layer (exploratory)"（红色，P5-P9）

**图形参数**：
- 尺寸：16×12 英寸
- DPI：150
- 颜色映射：每个面板独立色标

**数据来源**：
- A 层：`logs/eval/phase_morl_v2/aggregated/policy_level_confirmatory.csv`
- B 层：`logs/eval/phase_morl_v2/aggregated/policy_level_exploratory.csv`

**论文使用建议**：
- 建议作为附录图（Appendix Figure A1）
- 标题：Policy-Scenario Performance Heatmap (4 Objectives)
- 说明：提供完整的 10×6×4 数据矩阵，便于读者查阅具体数值

**注意事项**：
- 每个面板独立颜色尺度，数值不可跨指标直接比大小
- 色彩语义：更优（更小）的 J_* 对应更显著的"好"色（红色）
- B 层数据只作为 exploratory 补充，不能混入官方结论

---

### 图 4：消融对比图

**文件名**：`phase4_ablation_comparison.png`

**内容**：
- 展示去除能效目标（no-energy）和平滑目标（no-smooth）后的性能变化
- 证明这两个目标在 P10 中的局部必要性（local necessity）

**布局**：
- 2×2 面板（4 个指标）
- 左上：Delta Mean Velocity (m/s)
- 右上：Delta J_energy
- 左下：Delta J_smooth
- 右下：Delta J_stable

**图形类型**：
- 分组柱状图
- 横轴：S1-S6
- 两组柱：
  - P10-no-energy（蓝色，steelblue）
  - P10-no-smooth（橙色，coral）
- 零基线：黑色虚线

**Delta 定义**：
- Delta = Ablation - Anchor
- 负值：性能恶化（去掉目标后变差）
- 正值：性能改善（去掉目标后变好）

**图形参数**：
- 尺寸：16×12 英寸
- DPI：150
- 透明度：0.8

**数据来源**：
- `logs/eval/phase_morl_v2/ablation/ablation_comparison.csv`
- 字段：`delta_mean_vx_meas`, `delta_J_energy`, `delta_J_smooth`, `delta_J_stable`

**论文使用建议**：
- 建议作为正文主图（Figure 3）
- 标题：Ablation Study: Impact of Removing Objectives
- 说明：
  - 重点展示 Delta J_energy 和 Delta J_smooth 面板
  - 去除能效目标后，J_energy 显著恶化（Delta < 0）
  - 去除平滑目标后，J_smooth 显著恶化（Delta < 0）

**注意事项**：
- 消融实验只证明 **local necessity**（局部必要性）
- 不能写成"证明最优"或"充分性证明"
- 不能写成"联合最优"

**正确表述**：
```
消融实验（图 3）表明，去除能效目标后，J_energy 显著恶化（Δ < 0），
证明了能效目标在 P10 中的局部必要性（local necessity）。
```

**错误表述**（避免）：
```
❌ 消融实验证明了能效目标的充分性
❌ 消融实验证明了 P10 的最优性
❌ 消融实验证明了多目标联合最优
```

---

## 🎨 图形设计原则

### 颜色选择

1. **热力图（图 3）**：
   - 使用 YlOrRd_r（黄-橙-红反转）
   - 语义：越红越优（代价越小越好）
   - 避免使用 RdYlGn（红-黄-绿），因为"红色=差"的直觉与"代价小=好"冲突

2. **柱状图（图 4）**：
   - P10-no-energy：steelblue（蓝色）
   - P10-no-smooth：coral（橙色）
   - 零基线：黑色虚线
   - 避免使用红色（可能暗示"错误"）

### 布局原则

1. **多面板图**：
   - 使用 2×2 布局（4 个指标）
   - 子图间距适中（hspace=0.3, wspace=0.3）
   - 统一标题字体（12pt，粗体）

2. **单图**：
   - 使用 tight_layout 自动调整边距
   - 保留足够的轴标签空间

### 分辨率要求

- **论文正文图**：DPI ≥ 150
- **演示文稿图**：DPI ≥ 150
- **海报图**：DPI ≥ 200

---

## 📝 论文使用建议

### 正文主图（建议放入正文）

1. **图 1：Pareto 总览** → Figure 1
   - 展示 A 层策略在不同场景下的 trade-off
   - 突出 P10 在多数场景中的 Pareto 成员资格

2. **图 2：HV 柱状图** → Figure 2
   - 量化不同场景的 Pareto front 质量
   - 说明哪些场景更具挑战性

3. **图 4：消融对比图** → Figure 3
   - 证明能效和平滑目标的局部必要性
   - 重点展示 Delta J_energy 和 Delta J_smooth 面板

### 附录补充图（建议放入附录）

- **图 3：策略-场景热力图** → Appendix Figure A1
  - 提供完整的 10×6×4 数据矩阵
  - 便于读者查阅具体数值

---

## 🔍 数据溯源

### 上游数据文件

| 图形 | 数据源 | 路径 |
|------|--------|------|
| 图 1 | policy_level_confirmatory.csv | `logs/eval/phase_morl_v2/aggregated/` |
| 图 2 | confirmatory_scenario_hv.json | `logs/eval/phase_morl_v2/pareto/` |
| 图 3 | policy_level_confirmatory.csv + exploratory.csv | `logs/eval/phase_morl_v2/aggregated/` |
| 图 4 | ablation_comparison.csv | `logs/eval/phase_morl_v2/ablation/` |

### 生成脚本

- **脚本路径**：`logs/eval/phase_morl_v2/figures/generate_figures.py`
- **执行命令**：`conda run -n env_isaaclab python logs/eval/phase_morl_v2/figures/generate_figures.py`
- **依赖库**：pandas, matplotlib, numpy

### 基础素材

- **单场景 Pareto 子图**：`docs/figures/phase4_pareto_s1.png` ~ `s6.png`
- **生成脚本**：`scripts/phase_morl/analyze_phase4_pareto.py`

---

## ⚠️ 重要注意事项

### 1. A 层 vs B 层的使用限制

**A 层（confirmatory set）**：
- 策略：P1, P2, P3, P4, P10
- 种子数：3（42, 43, 44）
- 用途：官方 Pareto front、HV 计算、正文结论

**B 层（exploratory set）**：
- 策略：P5, P6, P7, P8, P9
- 种子数：1（42）
- 用途：探索性补充，不能混入官方结论

**正确使用**：
```
图 1 和图 2 只显示 A 层策略（P1-P4, P10）。
图 3 同时显示 A 层和 B 层，但明确标注 B 层为 exploratory。
```

### 2. 消融实验的解释语气

**限制**：
- 消融实验只证明 **local necessity**（局部必要性）
- 不能写成"证明最优"或"充分性证明"
- 不能写成"联合最优"

**正确表述**：
```
消融实验（图 4）表明，去除能效目标后，J_energy 显著恶化（Δ < 0），
证明了能效目标在 P10 中的局部必要性（local necessity）。
```

### 3. HV 的计算口径

**限制**：
- HV 只基于 A 层 confirmatory set
- 不能出现混合 A/B 层的 HV(global) 表述

**正确表述**：
```
图 2 展示了基于 A 层 confirmatory set（P1-P4, P10）的 Hypervolume。
```

---

## 📅 版本历史

- **v2.0** (2026-04-08): 重新生成图 1（优化布局，增大子图尺寸）
- **v1.0** (2026-04-08): 初始版本（生成图 1, 3, 4）

---

## 📧 相关文档

- 表格文档：`logs/eval/phase_morl_v2/tables/README.md`
- 图表需求文档：`docs/daily_logs/2026-4/2026-04-07/s4-9_fig_table_requirements.md`
- 阶段四详细计划：`docs/daily_logs/2026-4/2026-04-04/phase4_detailed_plan.md`
