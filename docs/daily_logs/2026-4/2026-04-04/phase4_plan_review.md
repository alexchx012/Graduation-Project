# 阶段四计划审查报告

> 审查日期：2026-04-04
> 审查对象：`docs/daily_logs/2026-4/2026-04-04/phase4_detailed_plan.md`
> 审查方法：5 个独立审查视角（事实准确性、工程可行性、统计方法论、排期风险、论文可交付性）

---

## 1. 事实准确性审查 — 全部通过

所有 [Fact] 和 [Inference + Evidence] 标注的声明均已对照实际文件验证，**零事实错误**。

| 声明 | 验证文件 | 结果 |
|------|----------|------|
| Pareto front = P1,P2,P3; HV = 0.2666 | `logs/eval/phase_morl_v2/pareto_analysis_S1.json` | ✅ 精确匹配 |
| 旧 Pareto P9,P10; HV = 0.337518 | `logs/eval/phase_morl/_archive_20260331_v1_evals/pareto_analysis.json` | ✅ 精确匹配 |
| 20/20 干净 S1 eval JSON | `logs/eval/phase_morl_v2/morl_p*_S1.json` | ✅ 20 个文件 |
| 15 个 A 层 + 5 个 B 层 checkpoint | `logs/rsl_rl/unitree_go1_rough/` | ✅ 全部存在 |
| Baseline seed42 路径 | `.../2026-03-08_16-46-27_baseline_rough_ros2cmd/` | ✅ 存在，含 model_1499.pt |
| Baseline seed43/44 archive 路径 | `_archive_20260331_pre_v2_sweep/2026-03-10_*` | ✅ 两个目录均存在 |
| S1/S4/S6 smoke 于 3/27 完成 | `docs/daily_logs/2026-3/2026-03-27/2026-3-27.md` | ✅ Section 5 记录 |
| `FROZEN_NORMALIZATION_BOUNDS` / `ref_point` | `scripts/phase_morl/analyze_pareto.py:29-37` | ✅ 与代码一致 |
| `scenario_defs.py` 定义 S1-S6 | `scripts/phase_morl/scenario_defs.py` | ✅ 6 个 ScenarioSpec |

---

## 2. 工程可行性审查 — 3 个高风险点

### 2.1 [P0] Baseline eval task 不兼容

**问题**：Baseline 使用 `Isaac-Velocity-Rough-Unitree-Go1-ROS2Cmd-Play-v0`（ROS2 task），MORL 使用 `...-Play-v2`（内部命令 task）。`run_morl_eval.py` 当前无法评估非 MORL checkpoint：

- `DEFAULT_TASK` 硬编码为 MORL Play v0
- `ROS2_MORL_TASK_IDS` 只包含 MORL task
- 策略网络结构可能不同（MORL v2 有 scaffold reward 不同的 observation processing）

**影响**：如果不解决，S4-1（baseline 归一化）和 S4-4（baseline 对照矩阵）**完全被阻塞**。

**建议方案**：

| 方案 | 工作量 | 风险 |
|------|--------|------|
| A: 新建 `run_baseline_eval.py` | 2-3h | 低，但增加维护负担 |
| B: 重构 `run_morl_eval.py` 为 task-agnostic | 4-6h | 中，需验证 MORL/baseline 兼容性 |
| C: 直接用 `scripts/go1-ros2-test/eval.py` + `--scenario` 适配 | 3-4h | 低，复用现有代码 |

**推荐方案 B**：在 `run_morl_eval.py` 中添加 `--task` 参数支持非 MORL task，metrics fallback 逻辑已有（`_extract_velocity_metrics` 的 tensor fallback），只需确保 `gym.make()` 能正确创建 baseline env。

### 2.2 [中高] 旧 S1 数据复用风险

**问题**：如果 manifest 化后 JSON schema 变化（新增 `family`、`evidence_layer` 等字段），现有 20 个 S1 eval JSON 无法直接被聚合层消费。

**建议**：

- 在 S4-0 中添加 schema 验证：比对现有 JSON 字段与聚合脚本期望字段
- 如果不一致，评估重跑成本（20 × 6min = 2h）vs 补字段成本
- 聚合脚本设计为容忍缺失字段（用 `dict.get()` + 默认值）

### 2.3 [中] 脚本增殖过度

**问题**：计划新增 4 个分析脚本：

| 新脚本 | 功能 | 与现有脚本重叠 |
|--------|------|---------------|
| `aggregate_phase4_results.py` | JSON → CSV 聚合 | `analyze_pareto.py` 已有 `aggregate_policy_rows()` |
| `check_phase4_qc.py` | 数据质检 | 可作为聚合前的验证步骤 |
| `analyze_phase4_pareto.py` | 按场景 Pareto/HV | `analyze_pareto.py --scenario` 已支持 |
| `render_phase4_figures.py` | 生成论文图表 | `analyze_pareto.py` 已有 `save_*_figure()` |

**建议**：合并为 1 个 `phase4_analysis.py` 编排脚本，内部调用模块化函数：

```
phase4_analysis.py aggregate  → 读 manifest + JSON → 输出 CSV
phase4_analysis.py validate   → QC 检查
phase4_analysis.py pareto     → 按场景 Pareto/HV
phase4_analysis.py render     → 论文图表
phase4_analysis.py all        → 以上全部
```

节省约 500 行代码和 3 个测试文件。

### 2.4 其他工程评估

| 项目 | 评估 | 工作量 |
|------|------|--------|
| Manifest 化 `run_full_eval_matrix.py` | 直接可做，~100 行改动 | 低 |
| 消融 runner 泛化 | ~80 行改动，从硬编码 P10 改为读 manifest | 低 |
| 单元测试（8+ 个） | 沿用现有 `importlib` 模式，无需仿真 | 中（8-10h） |
| `analyze_pareto.py` 归一化外化 | 将 `FROZEN_NORMALIZATION_BOUNDS` 移到 config JSON | 低（1h） |

---

## 3. 统计方法论审查 — 3 个关键问题

### 3.1 [P0] A/B 层混入同一个 Pareto 计算

**问题**：当前 `pareto_analysis_S1.json` 将 A 层（3 seeds 均值，有 std）和 B 层（1 seed 单点，std=0）放入同一个 Pareto/HV 计算。

**统计缺陷**：

- B 层单点没有置信区间，无法判断其位置是真实还是 seed 噪声
- 单点 std=0 使其在 Pareto 支配判断中获得不公平优势（看起来"确定性最优"）
- HV 计算混合了不同置信水平的数据

**建议**：

1. **官方 HV 只用 A 层** 5 个 policy-level 均值（P1/P2/P3/P4/P10）
2. B 层（P5-P9）单独做 overlay 图，标注 "exploratory, single-seed"
3. 在论文中明确说明："Official Hypervolume is computed from the confirmatory set (5 policies × 3 seeds). Exploratory policies (single seed) are shown for reference but excluded from statistical comparisons."

### 3.2 [P1] 5 点 4 维 Pareto 稳定性

**问题**：A 层只有 5 个 policy，4 维空间中 5 点的 Pareto front 对测量噪声极度敏感。当前 front = P1,P2,P3（3/5 = 60% 非支配），这在 4D 中偏高且可能不稳定。

**具体数据（归一化后 S1）**：

| Policy | J_speed_norm | J_energy_norm | J_smooth_norm | J_stable_norm |
|--------|-------------|---------------|---------------|---------------|
| P1 | 0.079 | 0.034 | 0.681 | 0.647 |
| P2 | 0.082 | 0.029 | 0.634 | 0.612 |
| P3 | 0.090 | 0.035 | 0.647 | 0.575 |
| P4 | 0.086 | 0.035 | 0.680 | 0.623 |
| P10 | 0.083 | 0.034 | 0.683 | 0.636 |

P4 被 P1 支配仅因为 J_smooth 一个维度的微小差异（0.680 vs 0.681），这可能在 ±1σ 范围内翻转。

**建议**：

1. 添加 bootstrap Pareto 分析（1000 次重采样，尊重 seed 分组）
2. 报告每个 policy 出现在 front 上的频率：`P(on_front)`
3. 示例格式：

| Policy | P(on_front) | 95% CI |
|--------|------------|--------|
| P1 | 98% | [94%, 100%] |
| P2 | 100% | [97%, 100%] |
| P3 | 85% | [78%, 92%] |
| P4 | 12% | [6%, 18%] |
| P10 | 8% | [3%, 14%] |

4. **结论措辞**：不要写 "P1/P2/P3 是 Pareto 最优"，而是 "P1/P2 在 >95% bootstrap 样本中保持非支配；P3 在 85% 样本中保持非支配"。

### 3.3 [P1] S5/S6 归一化边界风险

**问题**：当前冻结边界：

| 指标 | 边界 | S1 实际范围 | S5 潜在范围 |
|------|------|-----------|-----------|
| J_speed | [0, 1.2] | [0.09, 0.13] | 可能 > 0.5（跌倒） |
| J_energy | [0, 2500] | [68, 96] | 可能 > 500（爬台阶） |
| J_stable | [0, 0.5] | [0.22, 0.43] | 可能 > 0.5（clip 失真） |

如果 S5（台阶）导致 J_stable > 0.5，归一化值被 clip 到 1.0，丢失策略间的排序信息。

**建议**：

1. S4-2 预跑**必须包含 S5**（至少 1 policy × 1 seed）
2. 如果任何指标超出当前边界 30%，考虑：
   - 方案 A：扩大全局边界（影响所有场景的归一化）
   - 方案 B：按场景分别归一化（增加复杂度但更准确）
3. 冻结边界时记录日志："Bounds verified against S1-S6 pre-run data on [date]; max observed = [values]"

### 3.4 其他统计建议

| 建议 | 优先级 | 说明 |
|------|--------|------|
| 跨场景不做混合 HV(global) | ✅ 计划已正确规避 | 语义不同的场景不应合并 |
| 加场景难度指标 | P3 | `Difficulty(s) = mean_HV(s) / max_possible_HV(s)` |
| 加策略稳健性指标 | P2 | `Robustness(p) = 1 - std(HV_rank across S1-S6)` |
| 消融只测必要性，不测充分性 | 已知限制 | 论文中用 "necessity test" 措辞，不声称 "jointly optimal" |

---

## 4. 排期与风险审查

### 4.1 排期调整建议

| 步骤 | 计划天数 | 风险点 | 建议天数 |
|------|----------|--------|----------|
| S4-0/S4-1 | 2d (04-04~05) | Baseline eval 重构未计入 | **3d** (04-04~06) |
| S4-2 | 1d (04-06) | 需检查 S5 metric 范围 | 1d (04-07) |
| S4-3 | 2d (04-07~08) | 120 evals × 6-8min = 12-16h GPU | 2d (04-08~09) |
| S4-4 | 1d (04-09) | **可与 S4-3 并行** | 并行 (04-08~09) |
| S4-5 | 2d (04-10~11) | 聚合+QC，如合并脚本可缩短 | 1.5d (04-10~11) |
| S4-6 | 1d (04-12) | 需同时开始 S4-7 泛化 | 1d (04-11) |
| S4-7/S4-8 | 3d (04-13~15) | 消融训练 6h + 评估 36 runs | 2.5d (04-12~14) |
| S4-9 | 5d (04-16~20) | 图表+写作，无导师 review 缓冲 | 4d (04-15~18) + 2d 缓冲 |

**建议并行化**：

```
Track 1 (critical): S4-0/1 → S4-2 → S4-3 → S4-5 → S4-6 → S4-7/8 → S4-9
Track 2 (parallel):                    S4-4 (与 S4-3 并行)
```

**建议完成目标**：04-18（留 04-19~20 缓冲）

### 4.2 计划遗漏的风险

| 风险 | 影响 | 对策 |
|------|------|------|
| **S3/S5 MORL policy 泛化失败** | MORL 训练仅在 rough 地形，未在 20° 坡/15cm 台阶上训练。如果 vx < 0.3 则结论受限 | S4-6 中若发现失败场景，标记为 non-generalizable，不作为推荐策略的支撑场景 |
| **Manifest schema 设计迭代** | 如果 JSON 结构返工，S4-0 从 2d 变 3d | 先写最小 schema（3 个必填字段），通过 dry-run 验证后再扩展 |
| **消融训练 seed 敏感性复发** | `--init_with_optimizer` 修复了 MORL 训练的问题，但消融是新场景 | S4-7 中若 seed43/44 vx < 0.5，暂停调查再继续 |

---

## 5. 论文可交付性审查 — 2 个严重缺口

### 5.1 [P1] 缺少 MORL SOTA 对比

**问题**：计划没有与 PGMORL、MORL/D 等方法做任何对比。答辩委员会几乎必问："你的方法比现有 MORL 方法好在哪？"

**建议（二选一）**：

| 方案 | 工作量 | 效果 |
|------|--------|------|
| **实验对比**：实现 PGMORL 在同一环境 | 2-3 天 | 最强，有直接数据 |
| **叙事对比**：论文中加定位段落 | 0.5 天 | 可接受，解释设计取舍 |

叙事对比示例："Unlike PGMORL's continuous weight adaptation (suited for online policy switching), our approach uses discrete weight profiles for interpretability and reproducibility. Within the discrete-weight class, our contribution is the curriculum-based MORL training and reward-terrain trap avoidance."

### 5.2 [P1] 缺少手动 reward tuning 基线

**问题**：`reward_engineering.md` 第 5-7 章描述了手动调 reward 权重的实验（如 `track_lin_vel_xy_exp` 权重提升到 3.5）。计划只拿 rough baseline 做对照，没有把手动调优变体放进来。

**为什么重要**：MORL 的替代方案不是 "不做 reward engineering"，而是 "手动做 reward engineering"。如果不对比，无法论证 MORL 的价值。

**建议**：

1. 从 `reward_engineering.md` 第 5 章提取已有的手动调优 checkpoint（如 `track_vel_high`、`action_rate_high`）
2. 将它们加入 `phase4_main_manifest.json` 作为 `family=manual_tuning`
3. 在 Table 3（baseline 对照表）中增加 1-2 行手动调优变体

### 5.3 图表质量

当前 Pareto 图（`pareto_front_pairwise_S1.png`、`pareto_front_policy_summary_S1.png`）评估：

| 方面 | 当前状态 | 论文要求 |
|------|----------|----------|
| 误差可视化 | 无 | 需要 ±1σ 误差棒或置信区域 |
| Baseline 对照 | 无 | 需要 baseline 作为参考点叠加 |
| 多场景视图 | 仅 S1 | 需要 6 场景缩略图或热力图 |
| 标注 | 最小化 | 需要关键 trade-off 文字标注 |

**建议新增图表**：

| 图 | 内容 | 目的 |
|------|------|------|
| 策略-场景热力图 | 10 policies × 6 scenarios × 4 metrics | 快速看全局模式 |
| HV 柱状图 | 6 个场景的 HV 并列 | 比较场景间 trade-off 丰富度 |
| Parallel Coordinates | 4D 指标连线图 | 直观展示高维 trade-off |
| 权重-指标相关图 | 训练权重 vs 物理指标散点 | 回答 "为什么" |

### 5.4 叙事深度

**问题**：S4-9 的产出定义是"表格 + 图 + 结论"，但论文需要三层叙事：

| 层次 | 当前覆盖 | 建议 |
|------|----------|------|
| **事实层**（是什么） | ✅ 表格数据 | 无需变化 |
| **机制层**（为什么） | ❌ 缺失 | 解释 trade-off 产生原因：权重如何影响梯度方向 → 策略行为 → 物理指标 |
| **含义层**（所以呢） | ❌ 缺失 | P2 比 P1 慢 2% 但省电 6%，对实际部署意味着什么 |

**建议 S4-9 叙事结构**：

```
5.1 策略性能画像（事实层）
5.2 Trade-off 机制分析（机制层）
5.3 MORL vs Baseline 定位（对比层）
5.4 消融解释（因果层）
5.5 最终推荐与适用场景（决策层）
```

### 5.5 其他论文建议

| 建议 | 优先级 | 说明 |
|------|--------|------|
| 消融设计只测必要性 | 已知限制 | 论文中明确说 "necessity test"，不声称 "jointly optimal" |
| 加 2-factor 消融（去掉 energy + smooth） | P3 | 测试交互效应，可选 |
| B 层声明 | P2 | 明确标注 "exploratory, not included in official front" |
| 明确消融训练协议 | P2 | 写明 "从头训练" 还是 "从锚点微调" |

---

## 6. 综合建议优先级

### P0 — 阻塞性问题（必须在 S4-0 前解决）

| # | 事项 | 来源 | 行动 |
|---|------|------|------|
| 1 | Baseline eval task 不兼容 | 工程 | 重构 `run_morl_eval.py` 支持 `--task` 非 MORL，或写独立 baseline eval |
| 2 | A/B 层不能混入同一个 HV | 统计 | 修改分析逻辑：官方 HV 只用 A 层，B 层做 overlay |

### P1 — 严重缺陷（必须在 S4-9 前解决）

| # | 事项 | 来源 | 行动 |
|---|------|------|------|
| 3 | 缺少 MORL SOTA 对比 | 论文 | 至少加叙事对比段落（0.5 天） |
| 4 | 缺少手动调优基线 | 论文 | 提取 Ch5-7 checkpoint 作为额外对照行 |
| 5 | S5/S6 归一化边界预检 | 统计 | S4-2 预跑后验证 metric 范围 |
| 6 | 5 点 Pareto bootstrap CI | 统计 | 在 S4-6 中加 1000 次 bootstrap |

### P2 — 改进建议（提升论文质量）

| # | 事项 | 来源 | 行动 |
|---|------|------|------|
| 7 | 脚本合并为 1 个编排脚本 | 工程 | 减少维护负担 |
| 8 | Pareto 图升级（误差棒 + baseline） | 论文 | 升级为论文级图表 |
| 9 | 排期留 2d 缓冲 | 项目 | 目标 04-18 完成 |
| 10 | 叙事加机制解释层 | 论文 | S4-9 增加 "为什么" 分析 |

### P3 — 可选增强

| # | 事项 | 来源 | 行动 |
|---|------|------|------|
| 11 | 场景难度指标 | 统计 | 简单计算，不需额外实验 |
| 12 | 策略稳健性指标 | 统计 | 跨场景 HV rank 的稳定性 |
| 13 | Parallel Coordinates 可视化 | 论文 | 4D trade-off 直观展示 |
| 14 | 2-factor 消融（energy + smooth 同时去除） | 论文 | 测试交互效应 |
