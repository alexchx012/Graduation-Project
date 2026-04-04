# 阶段四计划审查回复

> 回复日期：2026-04-04
> 回复对象：`docs/daily_logs/2026-4/2026-04-04/phase4_plan_review.md`
> 当前被回复版本：`docs/daily_logs/2026-4/2026-04-04/phase4_detailed_plan.md`
> 回复原则：逐条判断“接受 / 部分接受 / 不接受”，并给出当前证据基础上的技术理由。

---

## 0. 总体结论

- [Fact] `事实准确性审查` 整体接受。
- [Fact] `工程可行性审查` 中，`2.1 baseline eval task 不兼容` 已被后续 smoke 证伪；`2.2` 接受；`2.3` 作为实现风格建议部分接受。
- [Fact] `统计方法论审查` 中，`3.1` 已被当前计划吸收；`3.2`、`3.3` 接受并已回写；`3.4` 中部分建议保留为可选增强。
- [Fact] `排期与风险审查` 中，`4.2` 的 3 条风险均有效，其中 2 条已回写，1 条已在更早步骤中被覆盖。
- [Fact] `论文可交付性审查` 中，叙事结构、图表升级、消融协议说明接受；SOTA 对比只接受叙事定位，不接受追加复现实验；手动 reward tuning 基线不接受按现有 flat checkpoint 直接并入主 manifest 的建议。

---

## 1. 事实准确性审查

### 1.1 整体结论

**处理结果：接受**

**理由**

- 当前 review 对 `Pareto front = P1,P2,P3`、`HV = 0.2666`、baseline 路径、A/B 层 checkpoint 数量、`scenario_defs.py`、`FROZEN_NORMALIZATION_BOUNDS` 等事实核对基本准确。
- 这些事实与当前仓库文件和日志一致，没有发现需要反驳的地方。

---

## 2. 工程可行性审查

### 2.1 [P0] Baseline eval task 不兼容

**处理结果：不接受**

**理由**

- 后续验证已经直接推翻这条判断。
- `run_morl_eval.py` 支持显式 `--task` 参数，不是只能跑 MORL task。
- `Isaac-Velocity-Rough-Unitree-Go1-ROS2Cmd-Play-v0` 已注册且可被 `run_morl_eval.py` 调用。
- 2026-04-04 的 `smoke.log` 已经直接证明以下命令可以成功运行并写出 JSON：

```powershell
python scripts/phase_morl/run_morl_eval.py `
  --task Isaac-Velocity-Rough-Unitree-Go1-ROS2Cmd-Play-v0 `
  --load_run 2026-03-08_16-46-27_baseline_rough_ros2cmd `
  --checkpoint model_1499.pt `
  --scenario S1 `
  --skip_ros2 `
  --num_envs 8 `
  --eval_steps 40 `
  --warmup_steps 5 `
  --summary_json logs/eval/_compat_smoke/baseline_seed42_S1_smoke.json `
  --headless
```

- 因此 baseline eval 的真正问题不在 task compatibility，而在：
  - manifest 化
  - baseline seed / checkpoint / output 命名归一化
  - 正式批量链接入

**当前处理**

- 已在 `phase4_detailed_plan.md` 中改写为：
  - baseline eval 已验证通过
  - 后续工作重点转为 manifest 与批量接入

### 2.2 [中高] 旧 S1 数据复用风险

**处理结果：接受**

**理由**

- 这是有效的工程前置检查项。
- 如果 manifest 化后的聚合层需要新的字段，而现有 `S1` JSON 没有，就会出现“主矩阵能复用，聚合层却吃不下”的问题。
- 当前计划原先只把这件事写成了隐含假设，review 这里给出了更明确的工程落点。

**当前处理**

- 已在 `S4-0` 中新增：
  - 旧 `S1` JSON schema 验证
  - 若字段不足，则在“补字段”和“重跑 `S1`”之间二选一

### 2.3 [中] 脚本增殖过度

**处理结果：部分接受**

**理由**

- review 的核心担心成立：如果把 manifest、聚合、QC、Pareto、渲染拆成过多脚本，确实会增加维护面。
- 但“必须合并成一个 `phase4_analysis.py`”并不是当前的硬缺陷，只是实现风格偏好。
- 当前计划选择“分层脚本 + 明确职责”的原因是：
  - 更便于 TDD
  - 更容易在某一层单独做 smoke 和回归
  - 更适合 manifest 驱动的分阶段验证

**当前处理**

- 不把“合并为单一编排脚本”写进主计划。
- 保留为实现阶段的可选重构方向。

### 2.4 其他工程评估

#### 2.4.1 Manifest 化 `run_full_eval_matrix.py`

**处理结果：接受**

**理由**

- 与当前计划一致，已作为 `S4-0` 主任务。

#### 2.4.2 消融 runner 泛化

**处理结果：接受**

**理由**

- 与当前计划一致，已作为 `S4-7` 主任务。

#### 2.4.3 单元测试（8+ 个）

**处理结果：接受**

**理由**

- 与当前计划一致，测试现已单列，不再隐含在实验排期中。

#### 2.4.4 `analyze_pareto.py` 归一化外化

**处理结果：接受**

**理由**

- 与当前计划一致，已转为 `phase4_analysis_config.json`。

---

## 3. 统计方法论审查

### 3.1 [P0] A/B 层混入同一个 Pareto 计算

**处理结果：接受，但已被当前计划吸收**

**理由**

- 该批评对“当前代码现状”成立：旧 `pareto_analysis_S1.json` 里的 A/B 层确实是混算的。
- 但对“当前计划版本”而言，这条已被吸收：
  - 官方 Pareto/HV 只用 A 层 confirmatory set
  - B 层只做 exploratory overlay

**当前处理**

- 不再作为新增缺陷。
- 作为实现时必须兑现的统计规则保留。

### 3.2 [P1] 5 点 4 维 Pareto 稳定性

**处理结果：接受**

**理由**

- 当前计划虽然已经加入了 `unstable` 标记，但还不足以支撑“稳健非支配解”这种表述。
- A 层只有 5 个 policy，4 维空间下前沿对微小扰动敏感，这是合理担心。

**当前处理**

- 已新增 bootstrap Pareto 稳健性分析要求：
  - 输出 `P(on_front)`
  - 输出 `HV` 的 bootstrap 区间
  - 在论文中以 bootstrap front frequency 而不是单次 point estimate 作为“稳健非支配解”依据

### 3.3 [P1] S5/S6 归一化边界风险

**处理结果：接受**

**理由**

- 当前 `analyze_pareto.py` 的归一化边界仍是硬编码全局边界。
- 如果 `S5/S6` 的指标超界被 clip，Pareto/HV 排序会失真。

**当前处理**

- 已在计划中新增：
  - `S4-2` 必须显式覆盖 `S5`、`S6`
  - `preflight_metric_maxima.json`
  - 若任一主指标接近或超出候选边界，则先更新 `phase4_analysis_config.json`，再启动正式矩阵

### 3.4 其他统计建议

#### 3.4.1 跨场景不做混合 `HV(global)`

**处理结果：接受，但已被当前计划吸收**

**理由**

- 当前计划已经明确规避新的官方 `HV(global)`。

#### 3.4.2 场景难度指标

**处理结果：暂不接受，保留为可选增强**

**理由**

- 这是增量分析，不是当前阶段四闭环所必需。
- 当前优先级仍是先把正式主链跑通。

#### 3.4.3 策略稳健性指标

**处理结果：部分接受**

**理由**

- “稳健性”这个目标接受。
- 但当前计划已经通过：
  - bootstrap `P(on_front)`
  - `front_membership count`
  - `unstable` 标记
 形成了更直接的稳健性表达。
- 因此不再额外引入新的 `Robustness(p)` 指标作为主结论必要条件。

#### 3.4.4 消融只测必要性，不测充分性

**处理结果：接受，但已被当前计划吸收**

**理由**

- 当前计划和论文表述已经按“necessity / local necessity”约束。

---

## 4. 排期与风险审查

### 4.1 排期调整建议

**处理结果：部分接受**

**理由**

- review 对排期偏紧的担心有道理。
- 但其中最重的一项假设是“baseline eval task 不兼容”，这已被 smoke 证伪，因此整段排期不需要原样照搬。
- 当前计划已经通过增加显式工程任务来吸收主要复杂度：
  - manifest
  - schema 验证
  - bootstrap
  - 聚合/QC

**当前处理**

- 不按 review 建议整体重排到 04-18 目标。
- 保留当前排期，但承认它仍需要实际执行时动态校准。

### 4.2 计划遗漏的风险

#### 4.2.1 `S3/S5` MORL policy 泛化失败

**处理结果：接受**

**理由**

- 这是有效风险。
- 如果策略在 `S3/S5` 中明显失效，最终推荐结论只能覆盖部分场景。

**当前处理**

- 已加入风险表：
  - 失败场景需显式标记为 `non-generalizable`
  - 不作为推荐策略的支撑场景

#### 4.2.2 Manifest schema 设计迭代

**处理结果：接受，但已被当前计划吸收**

**理由**

- 已通过 `S4-0` 的 schema 验证与最小 manifest 路线吸收。

#### 4.2.3 消融训练 seed 敏感性复发

**处理结果：接受**

**理由**

- 这是有效风险。
- 即使主矩阵已经修复，ablation 仍是新训练链，存在复发可能。

**当前处理**

- 已加入风险表：
  - 在 `S4-7` 的 `S1` smoke 中设硬阈值
  - 若 seed43/44 明显退化，先暂停调查再继续

---

## 5. 论文可交付性审查

### 5.1 [P1] 缺少 MORL SOTA 对比

**处理结果：部分接受**

**理由**

- 接受“论文层需要方法定位”这个要求。
- 不接受“将 PGMORL / MORL-D 复现实验加入当前阶段四主任务”的建议。
- 当前仓库内没有现成 PGMORL / MORL-D 复现实验基础，额外追加会显著冲击阶段四主线。

**当前处理**

- 已在 `S4-9` 中新增 **SOTA 叙事定位**：
  - 说明当前方法属于离散权重 profile 的可解释型 MORL
  - 明确与连续权重自适应类方法的差别和取舍

### 5.2 [P1] 缺少手动 reward tuning 基线

**处理结果：部分接受**

**理由**

- 接受“手动 reward tuning 是合理替代方案，需要在论文里讨论”。
- 不接受“把现有 flat 手动调优 checkpoint 直接加入当前阶段四主 manifest”的建议。

**技术理由**

- 现有 `track_vel_high` / `action_rate_high` 属于 `unitree_go1_flat` 任务线，不是当前 rough / MORL 阶段四主线。
- flat 与 rough 的观测结构、场景复杂度、训练目标不同，直接并入当前六场景主对照会形成不公平比较。

**当前处理**

- 已在 `S4-9` 中明确：
  - flat 手动调优实验只作为论文背景与替代方案讨论素材
  - 若要做公平对照，必须新增 rough/manual-tuning 专门实验包

### 5.3 图表质量

**处理结果：接受**

**理由**

- 当前计划原本只列了图表名，review 对图表质量的要求合理。

**当前处理**

- 已新增图表升级要求：
  - baseline 必须显式出现
  - 多场景必须有总览图
  - 关键 trade-off 点加标注
  - 2D 对照图在适合时可加入 `±1σ`

### 5.4 叙事深度

**处理结果：接受**

**理由**

- 这是当前计划原先缺的部分。
- 光有表图和结论，不足以支持论文答辩中的“为什么”和“所以呢”。

**当前处理**

- 已在 `S4-9` 中新增 5 层叙事结构：
  1. 策略性能画像
  2. Trade-off 机制分析
  3. MORL vs baseline 定位
  4. 消融解释
  5. 最终推荐与适用场景

### 5.5 其他论文建议

#### 5.5.1 消融设计只测必要性

**处理结果：接受，但已被当前计划吸收**

#### 5.5.2 加 2-factor 消融

**处理结果：暂不接受**

**理由**

- 这是可选增强，不进入当前阶段四主计划。
- 当前优先级仍是先完成单因素局部必要性验证。

#### 5.5.3 B 层声明

**处理结果：接受，但已被当前计划吸收**

#### 5.5.4 明确消融训练协议

**处理结果：接受**

**理由**

- review 的要求合理。
- 当前计划之前没有明确写“是重新训练还是从 full anchor 微调”。

**当前处理**

- 已补充为：
  - 当前默认协议是从 baseline warm-start 重新训练 ablation 模型
  - 若改成从 full anchor 微调，必须在 `phase4_ablation_manifest.json` 中显式记录

---

## 6. 汇总处理结果

### 6.1 已接受并已回写到当前计划

1. 旧 `S1` 数据复用前先做 schema 验证
2. bootstrap Pareto 稳健性分析
3. `S5/S6` 归一化边界预检与 `analysis_config` 冻结
4. `S3/S5` 泛化失败风险
5. 消融训练 seed 敏感性复发风险
6. 图表升级要求
7. 论文叙事结构
8. SOTA 叙事定位
9. 消融训练协议说明

### 6.2 已接受，但当前计划中原本已覆盖

1. 事实准确性审查整体通过
2. 官方 HV 只用 A 层，B 层做 overlay
3. 不做混合 `HV(global)`
4. B 层显式标注 exploratory
5. 消融只测必要性，不测充分性
6. manifest schema 设计迭代风险

### 6.3 未接受或仅作为可选增强保留

1. `2.1` baseline eval task 不兼容
   - 已被 2026-04-04 baseline smoke 证伪
2. 将多个分析脚本强制合并为单一 `phase4_analysis.py`
   - 视角不同，当前不作为缺陷
3. 新增 PGMORL / MORL-D 复现实验
   - 不进入当前阶段四主任务
4. 将 flat 手动调优 checkpoint 直接并入当前阶段四主 manifest
   - 不具备同口径公平性
5. 追加 2-factor 消融
   - 仅保留为后续可选增强

