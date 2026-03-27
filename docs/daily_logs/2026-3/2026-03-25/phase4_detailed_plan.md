# 阶段四详细计划（修订版）

> 修订日期：2026-03-27
> 计划窗口：2026-03-25 至 2026-04-21
> 依据文件：`docs/daily_logs/毕设阶段.md`、`docs/daily_logs/2026-3/2026-03-08` 至 `2026-03-23` 全部关键日志
> 本版目的：将旧版草案重构为与当前 MORL 实际进展一致的正式阶段四执行计划

---

## 2. 阶段四目标

阶段四只做四件事：

1. 对当前 MORL 策略集做 **6 个正式场景** 的对比评估。
2. 在 **场景级 4 维物理指标** 上重做 Pareto Front 与 Hypervolume 分析。
3. 围绕一个已确认的折中策略做 **两组 reward 消融实验**：
   - 去掉能效目标
   - 去掉平滑目标
4. 产出可直接写入论文第 5 章/答辩 PPT 的表格、图和结论。

---

## 3. 明确不做的事

- 不把 2026-03-23 的统一测试集 HV=`0.337518` 直接当作阶段四最终 HV；该数值只作为阶段三参考基线。
- 不引入新的主目标定义；阶段四沿用阶段三已经冻结的 4 个物理主指标口径。

---

## 4. 资产冻结与证据分层

### 4.1 冻结输入资产

| 资产                                           | 当前状态      | 用途                                               |
| ---------------------------------------------- | ------------- | -------------------------------------------------- |
| `P1-P10` MORL checkpoint                     | ✅ 已有       | 阶段四正式评估输入                                 |
| 代表性策略 `P1/P2/P3/P4/P10 × seed42/43/44` | ✅ 已有 15 个 | 统计主表、场景均值 ± 标准差                       |
| 内部点策略 `P5-P9 × seed42`                 | ✅ 已有 5 个  | 探索性 Pareto 补点，不做强统计结论                 |
| rough baseline `seed42/43/44`                | ✅ 已有 3 个  | 外部对照组，回答 “MORL 相比非 MORL 基线是否更好” |
| `logs/eval/phase_morl/*.json`                | ✅ 已有       | 统一测试集参考，不替代场景评估                     |
| `logs/eval/phase_morl/pareto_analysis.json`  | ✅ 已有       | 阶段三 HV 参考线                                   |
| `docs/reward_engineering.md` 第 8 章         | ✅ 已有       | 提供阶段三收尾叙事，阶段四只补实验结果             |

### 4.2 证据强度分层

| 层级             | 策略集合                                                | 用途                                    | 结论强度 |
| ---------------- | ------------------------------------------------------- | --------------------------------------- | -------- |
| A 层（确认层）   | `P1/P2/P3/P4/P10 × 3 seeds`                          | 描述性统计主表、官方 Pareto/HV          | 中       |
| B 层（探索层）   | `P5/P6/P7/P8/P9 × seed42`                            | 填充 Pareto 形状、发现 trade-off 点     | 中       |
| C 层（消融层）   | `P10-full` 对比 `P10-no-energy` / `P10-no-smooth` | 局部验证：在 `P10` 邻域去掉目标会怎样 | 中       |
| D 层（外部对照） | rough baseline `seed42/43/44`                         | MORL vs 非 MORL 基线对照                | 中       |

说明：

- [Fact] `P5-P9` 当前只有 `seed42`。
- [Inference + Evidence] 因此阶段四主表必须以 `P1/P2/P3/P4/P10` 为统计主体，`P5-P9` 只能标注为 exploratory，不应进入论文主 HV，也不应写成“统计显著优于/劣于”。
- [Inference + Evidence] 外部对照至少保留 rough baseline；否则“六场景对比实验”会退化成 MORL 内部权重互比，无法回答 MORL 相比非 MORL 基线的增益。

---

## 5. 六场景正式定义

### 5.1 场景总体原则

- 所有场景都基于 **MORL PLAY 环境** 做评估，保证 observation/action 维度与 MORL checkpoint 完全兼容。
- 除场景变量本身外，其余物理参数默认继承当前 MORL/rough baseline 配置，不额外引入摩擦、质量、DR 扰动变化。
- 阶段四正式评估默认 **沿用 2026-03-23 已冻结的 `--skip_ros2` 口径**，即在 `run_morl_eval.py` 内通过确定性场景命令覆盖生成固定 `vx`，而不是重新切回 ROS2 publisher。
- ROS2 publisher 只保留两类用途：
  - bridge/smoke 调试
  - 如果后续专门新增“ROS2 场景协议”实验，则单独命名，不与正式六场景结果混写

### 5.2 场景表

| 场景 ID | 定义       | 目标命令   | 分布属性         | 实现备注                                          |
| ------- | ---------- | ---------- | ---------------- | ------------------------------------------------- |
| `S1`  | 平地中速   | `vx=1.0` | 参考场景 / 近 ID | 作为六场景基准线                                  |
| `S2`  | 平地高速   | `vx=1.5` | 高速 OOD         | 检查速度上限与失稳模式                            |
| `S3`  | 上坡 20°  | `vx=0.8` | 地形泛化         | 可复用阶段三 `Slope20` 参数                     |
| `S4`  | 下坡 20°  | `vx=0.6` | 地形泛化         | 当前仓库无现成 MORL 下坡配置，需新增              |
| `S5`  | 台阶 15 cm | `vx=0.5` | 高难度台阶场景   | 是否 OOD 需按 `pyramid_stairs` 训练范围单独说明 |
| `S6`  | 侧向扰动   | `vx=0.8` | 抗扰动泛化       | 先冻结实现口径，再决定能否严格落到“100N 力”     |

### 5.3 已有资产与缺口

- [Fact] 现有 `rough_env_cfg_terrain_eval.py` 已有：
  - `Slope10`
  - `Slope20`
  - `Stairs10`
  - `Stairs15`
- [Fact] 现有 `rough_env_cfg_dr_variants.py` 里 `DRPush` 使用的是“速度扰动事件”，不是“100N 侧向力”。
- [Fact] 现有 `scripts/phase_morl/run_morl_eval.py` 会在建环境前无条件关闭 `push_robot` 和 `randomize_apply_external_force_torque`；如果不先改这段逻辑，`S6` 只会得到“没有扰动的假场景”。
- [Fact] 当前仓库没有：
  - MORL 专用的 `S4` 下坡 20° 场景
  - MORL 专用的平地 `S1/S2` 固定场景配置
  - 与“100N 侧向推力”完全等价的现成 `S6` 配置
- [Fact] 当前仓库也没有：
  - `run_morl_eval.py` 级别的 `--scenario` 参数
  - terrain/flat/downhill/push 的 runtime override hook
  - 场景元数据写入 summary JSON 的 schema
- [Inference] 阶段四第一步必须先把“场景实现口径 + 评估脚本注入点 + JSON schema”一起冻结，再跑批量实验。
- [Don't know] Isaac Lab 当前事件 API 是否直接支持稳定、可复现的 100N 横向力注入；如果不支持，需要做等效扰动标定并在文档中明确说明。

---

## 6. 指标与数据口径

### 6.1 主指标

沿用阶段三已冻结的 4 个物理主指标，全部按 **越小越好**：

- `J_speed`
- `J_energy`
- `J_smooth`
- `J_stable`

### 6.2 辅助指标

- `success_rate`
- `mean_base_contact_rate`
- `mean_timeout_rate`
- `recovery_time`（`S6` 重点关注）
- `pass_flag`（仅作工程质检，不作为 Pareto 维度）

### 6.3 评估单元

正式场景评估统一采用当前 MORL 评估脚本口径：

- `num_envs=64`
- `eval_steps=3000`
- `warmup_steps=300`
- 每个 `checkpoint × scene` 至少覆盖 **10 个完整 episode**；若 `3000` 步不足以累计 10 个完成 episode，则延长 `eval_steps` 或重复 rollout，避免与 `毕设阶段.md` 的“10+ episodes”要求脱节
- 描述性统计的独立单位固定为 **训练 seed / checkpoint**，不是 env 数量

说明：

- [Fact] 2026-03-23 的统一测试集评估中，20 个 checkpoint 总耗时为 `46.65 min`。
- [Inference + Evidence] 按相同量级估算，`(20 MORL + 3 baseline) × 6 scenarios = 138` 次场景评估约为 `5.4h` 纯仿真时间，实际再加场景切换和 JSON 汇总，按 `7-9h` 规划更稳妥。

### 6.4 场景级 Pareto/HV 口径

阶段四的 HV 只保留一个官方层级：

1. **官方 HV**：`HV(S1)...HV(S6)`，只在 **确认层 policy-level 聚合结果** 上计算。
2. **探索性补图**：`P5-P9` 只作为 exploratory overlay，不进入论文主 HV。
3. **全局层**：不再定义新的官方 `HV(global)`；全局只保留 descriptive robustness summary，例如六场景均值表、ID/OOD 分组汇总或排名积分。

约束：

- 主分析对象固定为 4 维最小化指标。
- checkpoint-level 结果只用于 QC，不作为官方 Pareto/HV 计算单元。
- 官方 Pareto/HV 只用 `P1/P2/P3/P4/P10` 的 policy-level 聚合结果。
- `P5-P9` 进入图时必须打上“single-seed exploratory”标记。
- 场景边界遵循以下预注册规则：
  - 先尝试沿用 2026-03-23 的 stage3 边界；
  - 若 `rough baseline / P1 / P10` 在 `S1/S2/S5/S6` 先导 smoke 中有任一维度超出 stage3 边界的 `90%` 以上，则启用单独的 stage4-specific 边界；
  - stage4-specific 边界只允许由 `{rough baseline, P1, P10}` 的先导 smoke 冻结一次，冻结后不得因正式结果回改。
- HV 只作描述性比较，不做 p 值式强统计结论；同时输出：
  - `Pareto-membership frequency`
  - `leave-one-seed-out HV sensitivity`

---

## 7. 消融实验设计

### 7.1 消融锚点选择

- [Fact] `P10` 已有 `seed42/43/44`，且在 2026-03-23 的 policy-level Pareto front 中。
- [Inference + Evidence] `P10` 是阶段四最合适的消融锚点：
  - 已有 3-seed 基线，无需补基线训练
  - 是折中策略，不是单目标极端点
  - 适合作为 **局部消融** 的最低成本入口

但要明确：

- [Inference] 仅围绕 `P10` 的消融，最多能回答“在 `P10` 这类折中点附近，去掉目标会怎样”，不能直接外推成“该目标在全部 MORL 配方里都必要”。

### 7.2 两组强制消融

| 消融 ID   | 基础权重                       | 变更后权重                 | 目的                                   |
| --------- | ------------------------------ | -------------------------- | -------------------------------------- |
| `ABL-E` | `P10 = [0.2, 0.2, 0.2, 0.4]` | `[0.25, 0.0, 0.25, 0.5]` | 去掉能效目标，观察能耗与其他目标联动   |
| `ABL-S` | `P10 = [0.2, 0.2, 0.2, 0.4]` | `[0.25, 0.25, 0.0, 0.5]` | 去掉平滑目标，观察动作抖动与稳定性联动 |

说明：

- 只改 4 个主目标权重，辅助约束保持不变。
- 默认每个消融配置补 `seed42/43/44`，共 `2 × 3 = 6` 次训练。
- 对照组直接使用已有 `P10-full × 3 seeds`。
- 最终文稿默认把这两组结果表述为 **P10 邻域的局部必要性验证**。

### 7.3 增强版第二锚点（推荐，不是首轮硬门槛）

若首轮 `P10` 局部消融结果不够稳定，或导师要求把“局部结论”升级为更一般的“目标必要性”结论，则追加第二锚点：

- 候选锚点优先级：`P7`（较均衡） > `P9`（平滑偏好端）

对应追加训练：

- `anchor_full × seed42/43/44`
- `anchor_no_energy × seed42/43/44`
- `anchor_no_smooth × seed42/43/44`

这部分属于增强包，不写入首轮完成判据，但在时间允许时优先做。

### 7.4 消融评估矩阵

- 对照组：`P10-full × 3 seeds × 6 scenarios = 18` 次场景评估
- 消融组：`2 ablations × 3 seeds × 6 scenarios = 36` 次场景评估
- 合计：`54` 次场景评估

训练时间估算：

- [Fact] 2026-03-17 的 10 次确认层训练总耗时约 `18.7h`。
- [Inference + Evidence] 单次 MORL 1500 iter 训练约 `1.87h`；6 次消融训练约 `11-13h`，按 `1.5` 天窗口安排。

---

## 8. 阶段四分步骤执行

### S4-1 先冻结场景实现层

**目标**

- 给 `S1-S6` 定义唯一、可复跑的实现口径。

**建议文件**

- `scripts/phase_morl/run_morl_eval.py`
- `scripts/phase_morl/run_morl_scenario_eval_all.ps1`（新建）
- `scripts/phase_morl/scenario_defs.py`（新建）
- `tests/unit/test_morl_scenario_defs.py`（新建）
- 如 runtime override 不够稳定，再补：
  - `src/go1-ros2-test/envs/morl_scenario_eval_cfg.py`
  - `robot_lab/.../morl_scenario_eval_cfg.py`
  - `src/go1-ros2-test/envs/__init__.py` 注册
  - 相关白名单/解析逻辑更新

**本步必须显式落地的代码改动**

- 在 `run_morl_eval.py` 新增 `--scenario`
- 新增 `apply_scenario_overrides(env_cfg, scenario_id)` 钩子，负责：
  - `S1/S2` 平地化
  - `S3` 上坡 20°
  - `S4` 下坡 20°
  - `S5` 台阶 15 cm
  - `S6` 侧向扰动
  - `--skip_ros2` 下的固定命令注入
- 扩展 summary JSON schema，强制写入：
  - `scenario_id`
  - `scenario_name`
  - `terrain_mode`
  - `cmd_vx`
  - `disturbance_mode`
  - `analysis_group`
- 删除或重构当前无条件禁用 push 的逻辑，否则 `S6` 不成立

**产出**

- 场景配置表
- `S1/S4/S6` 三个 smoke JSON
- 冻结版 `scenario_defs.py`

**通过标准**

- `S1`、`S4`、`S6` 都能从 MORL checkpoint 正常启动
- 输出 JSON 字段完整
- 场景命令值与目标一致
- `S6` 扰动实现方式写清楚是“真实 100N”还是“标定等效推扰”

### S4-2 跑完整六场景评估矩阵

**目标**

- 先把现有 `20` 个 MORL checkpoint 与 `3` 个 rough baseline checkpoint 全部跑完 6 个场景。

**矩阵**

- `23 checkpoints × 6 scenarios = 138` 次正式评估

**建议输出路径**

- `logs/eval/phase_morl_stage4/raw/<scenario_id>/<run_name>.json`
- `logs/eval/phase_morl_stage4/index.csv`

**通过标准**

- `138/138` JSON 全部落盘
- 每条记录都带：
  - `scenario_id`
  - `policy_id`
  - `train_seed`
  - `cmd_vx`
  - `analysis_group`
  - 4 个主指标
  - 辅助指标

### S4-3 做数据质检与聚合

**目标**

- 在分析前先清理数据质量问题。

**检查项**

- 场景参数是否正确写入 JSON
- 是否出现 ROS2 超时回退
- 是否有异常零值/缺字段
- `P5-P9` 是否被正确标记为 `single_seed_only`

**建议输出**

- `logs/eval/phase_morl_stage4/aggregated/checkpoint_level.csv`
- `logs/eval/phase_morl_stage4/aggregated/policy_level.csv`
- `logs/eval/phase_morl_stage4/aggregated/control_group.csv`
- `logs/eval/phase_morl_stage4/qc_report.md`

**官方分析单元冻结**

- 官方单元：policy-level confirmatory MORL set（`P1/P2/P3/P4/P10`）
- 外部对照：rough baseline 单独成表，不混入官方 MORL HV
- exploratory：`P5-P9` 只出补图，不混入官方 MORL HV
- checkpoint-level：仅用于 QC 与 seed 敏感性，不写入主结论

### S4-4 做场景级 Pareto 与 HV

**目标**

- 给出阶段四核心结果：六场景 trade-off 结构。

**分析层次**

1. checkpoint-level exploratory
2. policy-level aggregated
3. confirmatory subset (`P1/P2/P3/P4/P10`)

**建议输出**

- `logs/eval/phase_morl_stage4/pareto/confirmatory_scenario_hv.json`
- `logs/eval/phase_morl_stage4/pareto/exploratory_overlay.json`
- `logs/eval/phase_morl_stage4/pareto/front_membership_frequency.csv`
- `logs/eval/phase_morl_stage4/pareto/hv_leave_one_seed_out.json`
- `logs/eval/phase_morl_stage4/pareto/robustness_summary.csv`
- `docs/figures/phase4_pareto_s1.png` ... `phase4_pareto_s6.png`
- `docs/figures/phase4_hv_bar.png`
- `docs/figures/phase4_policy_summary.png`
- `docs/figures/phase4_baseline_vs_morl.png`

**通过标准**

- 六个场景各有 Pareto 结果和 HV
- 没有新的官方 `HV(global)` 混入口径
- 明确区分“官方确认层结果”“外部对照”“探索性补点”
- 输出 `front membership frequency` 与 `leave-one-seed-out HV sensitivity`

### S4-5 实现并启动两组消融训练

**目标**

- 只做最小必要的新训练：`ABL-E` 与 `ABL-S`。

**建议文件**

- `scripts/phase_morl/run_morl_ablation_sweep.py`（新建）
- `tests/unit/test_run_morl_ablation_sweep.py`（新建）
- `scripts/phase_morl/analyze_pareto.py`（扩展解析逻辑）
- `scripts/phase_morl/run_morl_eval.py`（扩展 policy/run name 解析）

**训练矩阵**

- `ABL-E × seeds 42/43/44`
- `ABL-S × seeds 42/43/44`

**通过标准**

- `6/6` checkpoint 产出完整
- `env.yaml` 中主目标权重与计划一致
- 辅助约束未被误改
- `ABL-*` 命名不会破坏现有 `policy_id` 解析、Pareto 聚合与画图逻辑

### S4-6 对消融模型跑六场景评估

**目标**

- 将消融结果放入与正式场景评估完全一致的分析管线。

**矩阵**

- `54` 次评估（含 `P10-full` 对照组）

**建议输出**

- `logs/eval/phase_morl_stage4_ablation/raw/`
- `logs/eval/phase_morl_stage4_ablation/aggregated/`

**通过标准**

- 对照组与消融组可直接按场景逐列比较
- `ABL-E`、`ABL-S` 至少能回答：
  - 去掉该目标后，直接对应指标是否恶化
  - 其他三个目标是否出现补偿或副作用

### S4-7 汇总结论并形成论文材料

**目标**

- 将阶段四结果收敛成可写入论文/答辩的最终材料。

**最低产物**

- 表 1：`P1/P2/P3/P4/P10` 六场景主指标均值 ± 标准差
- 表 2：`P5-P9` 六场景 exploratory 补点表
- 表 3：`P10-full / ABL-E / ABL-S` 六场景对比表
- 图 1：六场景主指标热力图
- 图 2：六场景 HV 柱状图
- 图 3：六场景 Pareto 总览图
- 图 4：消融对比图

**文稿落点**

- 论文第 5 章“实验与结果分析”
- 答辩 PPT 的“场景实验结果”“Pareto/HV”“消融实验”三页

---

## 9. 预计时间排期

| 日期                     | 任务                          | 产出                                           |
| ------------------------ | ----------------------------- | ---------------------------------------------- |
| 2026-03-25 至 2026-03-28 | `S4-1` 场景实现冻结 + smoke | `scenario_defs.py`、3 个 smoke JSON          |
| 2026-03-29 至 2026-04-02 | `S4-2` 六场景全矩阵评估     | `138` 个 raw JSON                            |
| 2026-04-03 至 2026-04-05 | `S4-3` 数据质检与聚合       | `checkpoint_level.csv`、`policy_level.csv` |
| 2026-04-06 至 2026-04-08 | `S4-4` 场景级 Pareto/HV     | 场景 Pareto/HV JSON + 图                       |
| 2026-04-09 至 2026-04-12 | `S4-5` 两组消融训练         | `6` 个 ablation checkpoint                   |
| 2026-04-13 至 2026-04-15 | `S4-6` 消融六场景评估       | `54` 个 ablation JSON                        |
| 2026-04-16 至 2026-04-21 | `S4-7` 结果整理与写作材料   | 论文主表、图、结论草稿                         |

---

## 10. 风险与对策

| 风险                                                                                       | 影响                     | 对策                                                                                           |
| ------------------------------------------------------------------------------------------ | ------------------------ | ---------------------------------------------------------------------------------------------- |
| `S4` 下坡 20° 当前没有现成 MORL 配置                                                    | 批量评估无法开跑         | 先做 runtime override；不稳定再补专用 PLAY 配置                                                |
| `S6` 的“100N 推力”与现有 `DRPush` 事件口径不一致，且当前 eval 脚本会把 push 直接关掉 | 场景失真或根本跑不起来   | 先修 `run_morl_eval.py` 的 push 禁用逻辑，再做扰动标定，统一写“侧向扰动”或“等效侧向扰动” |
| `P5-P9` 只有单种子                                                                       | 容易被误写成强结论       | 所有图表强制加 exploratory 标签                                                                |
| 归一化边界如果边算边改                                                                     | HV 不可比较              | 只允许用 `{baseline, P1, P10}` 先导 smoke 冻结一次，写入 JSON 配置                           |
| 场景命令路径从 `--skip_ros2` 切回 ROS2                                                   | 场景差异与命令源差异混淆 | 正式评估坚持 `--skip_ros2`；ROS2 只做 smoke                                                  |
| `S2` 高速和 `S6` 侧向扰动属于压力测试                                                  | 结果看起来“差”         | 在结论中明确它们属于鲁棒性/极限测试，不按训练内性能解释                                        |

---

## 11. 阶段四完成判据

- 六场景正式评估全部完成，至少有 `138` 个有效 raw JSON。
- rough baseline 外部对照组已纳入同口径六场景评估。
- `P1/P2/P3/P4/P10` 的六场景统计主表完成。
- 六场景各自的官方 HV 完成，且没有新的混合 `HV(global)` 口径。
- `ABL-E` 与 `ABL-S` 两组消融训练、评估和对照分析完成。
- 论文第 5 章所需表格与图全部有源数据可追溯。
- 最终结论能清楚回答四个问题：
  1. 不同 MORL 权重偏好在六场景下分别擅长什么。
  2. 哪些策略在场景级 4 维指标上形成非支配解。
  3. 去掉能效或平滑目标后，会破坏哪些行为特征。
  4. 哪个策略最适合作为论文最终推荐策略。

---

## 12. 备注

- [Fact] 当前阶段四的真正输入是 MORL 阶段产物，不是旧的 PPO/DR 产物。
- [Inference] 最稳妥的执行顺序是：先把六场景评估体系搭起来，再做最小量消融训练，而不是反过来。
- [Assumption] MORL checkpoint 在“仅改场景、不改 observation/action 维度”的 PLAY 环境中可直接复用。
- [Assumption] 阶段四官方主分析是 **policy-level confirmatory + descriptive**，不是显著性统计。
- [Don't know] 如果 `S6` 必须严格落到“100N 力”而非“等效侧向扰动”，可能需要额外查 Isaac Lab 事件 API，再决定是 runtime override 还是新增专用场景配置。
