<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<tbody>
<tr>
<td><ol>
<li><p>选题依据（选题的背景、意义、目的、国内外现状分析等，不少于1000字。）</p></li>
</ol>
<p>1.1　课题研究背景和意义</p>
<p>四足机器人是模仿四足动物的仿生机器人。它通过四条机械腿实现行走、奔跑、跳跃等动作。和轮式机器人相比，四足机器人能跨越障碍、攀爬台阶，在非结构化环境中有明显优势。近年来，四足机器人在技术和产业化方面都取得了显著进展。根据中商产业研究院的报告，2023年中国四足机器人市场规模达4.7亿元，同比增长42.68%。2024年市场规模约为6.6亿元，预计2025年将达到8.5亿元<sup>[1]</sup>。销量方面，2023年约1.8万台，2024年约2万台。这些数据表明四足机器人具有广阔的市场前景。</p>
<p>四足机器人的运动控制一直是个难题。传统方法主要依赖动力学建模和轨迹优化。这类方法对建模精度要求高，实际部署时难以适应复杂多变的环境。随着深度学习和强化学习的发展，研究人员开始用数据驱动的方法来解决运动控制问题<sup>[2]</sup>。强化学习让机器人通过与环境交互自主学习控制策略，不需要精确的动力学模型。在处理高维状态空间和非线性系统方面，强化学习展现出很大潜力。</p>
<p>但是现有的强化学习方法存在一个问题。四足机器人运动控制中同时存在多个优化目标，比如速度跟踪精度、运动稳定性、能量消耗、动作平滑度等。这些目标之间往往相互冲突。目前的做法是把多个目标用人工设定的权重加权求和，变成一个标量奖励。这种方法有明显缺点：权重怎么选没有理论依据，换个任务可能就要重新调参，而且看不出不同目标之间到底怎么权衡<sup>[3]</sup>。</p>
<p>多目标强化学习（Multi-Objective Reinforcement Learning,
MORL）提供了一个解决思路。它可以同时优化多个目标，给出一组Pareto最优策略，让用户根据需要来选择<sup>[4]</sup>。这种方法能够显式建模目标之间的权衡关系，为不同任务场景提供可调节的控制策略。因此，研究基于多目标强化学习的四足机器人运动控制，既有理论价值，也有实际应用意义。</p>
<p>1.2　国内外研究现状</p>
<p>1.2.1　国外研究现状</p>
<p>在强化学习方法兴起之前，四足机器人运动控制主要依靠模型预测控制（MPC）等方法。Di
Carlo等人提出了一种基于凸MPC的运动控制方法，在MIT Cheetah
3平台上验证<sup>[5]</sup>。他们将机器人简化为单刚体模型，通过线性化将非凸优化问题转化为凸二次规划问题。这套方法让Cheetah
3实现了稳定的行走和小跑。Kim等人进一步提出了全身脉冲控制（WBIC）方法<sup>[6]</sup>，在Mini
Cheetah上实现了后空翻等高动态动作。但传统方法对模型精度依赖性强，计算量大，对未知扰动的鲁棒性不够好。</p>
<p>深度强化学习为运动控制提供了新路径。Hwangbo等人在ANYmal机器人上展示了深度强化学习的能力<sup>[2]</sup>。他们使用Actor-Critic算法训练神经网络控制器，策略能让机器人以约1.5m/s的速度行走，并在多种地形上保持稳定。仿真中训练的策略能直接迁移到实体机器人使用。Rudin等人提出了大规模并行训练方法<sup>[7]</sup>，利用GPU并行仿真上千个机器人实例，将训练时间从几天压缩到几分钟。Kumar等人提出了快速电机适应（RMA）方法<sup>[8]</sup>，包含基础策略网络和适应模块，让机器人可以在线适应负载变化和地形变化。</p>
<p>Tan等人研究了仿真到现实的迁移问题<sup>[9]</sup>，提出了域随机化（Domain
Randomization）方法。基本思路是在训练时随机化仿真环境的各种参数，让策略在多种条件下都能工作。这项技术现在已经成为四足机器人强化学习的标配。Fu等人研究了能量效率与步态之间的关系<sup>[10]</sup>，发现当以最小化能耗为目标时，机器人会自发学习到类似动物的步态。</p>
<p>多目标强化学习的理论研究已有一定积累。Hayes等人发表了多目标强化学习的实践指南<sup>[3]</sup>，Roijers等人对多目标序贯决策问题进行了系统综述<sup>[4]</sup>，将MORL方法分为单策略方法和多策略方法。Liu等人在IEEE
Trans.
SMC上发表了多目标强化学习的综合概述<sup>[11]</sup>。讨论了方法选择和应用场景。Mossalam等人将深度神经网络引入MORL，提出了Multi-Objective
DQN<sup>[12]</sup>。Nguyen等人提出了一个可扩展的多目标深度强化学习框架<sup>[13]</sup>。</p>
<p>1.2.2　国内研究现状</p>
<p>国内在四足机器人领域的研究也取得了不少进展。宇树科技（Unitree）是国内四足机器人的代表企业，其Go1、B1等产品在市场上有较高知名度。在学术研究方面，国内高校也在积极开展四足机器人的强化学习控制研究。Wang等人提出了基于CPG的分层运动控制框架，采用深度强化学习训练四足机器人<sup>[14]</sup>。Tan等人提出了一种分层强化学习框架，实现了四足机器人的敏捷运动<sup>[15]</sup>。</p>
<p>在多目标优化领域，国内学者也有相关研究。Zhang Linzi等人在Procedia
Computer
Science上发表了多目标强化学习的概念和应用综述<sup>[16]</sup>。但总体来看，将多目标强化学习系统性地应用于四足机器人运动控制的研究还比较少，存在研究空白。</p></td>
</tr>
</tbody>
</table>

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<tbody>
<tr>
<td><ol start="2">
<li><p>研究（或设计）的主要内容和方法1.研究目标和研究内容2.拟采取的研究方法、技术路线和/或实验方案</p></li>
</ol>
<ol type="1">
<li><p><strong>研究目标和研究内容</strong></p></li>
</ol>
<p>本课题面向四足机器人运动控制任务，研究并实现一种多目标强化学习的运动控制策略学习方法。核心工作是在仿真环境中完成训练与评测，验证方法的有效性。首先是构建多目标问题建模，需要定义机器人的状态空间和动作空间，然后设计多目标奖励函数。奖励函数要涵盖速度跟踪精度、运动稳定性、能量消耗、动作平滑度这几个关键指标。在此基础上，还要建立一套可解释的多目标评价体系，让不同目标的性能表现能够被量化和比较。</p>
<p>算法实现方面，本课题基于rsl_rl开源库和PPO算法<sup>[17]</sup>来开展工作，要实现不少于4个关键奖励函数，并设计权重向量采样与标量化求解机制。通过这套机制，可以对比单目标和多目标两条求解路线的效果差异。</p>
<p>实验平台也是研究内容的重要组成部分，本课题基于Isaac
Sim物理引擎搭建仿真环境，配置Unitree
Go1机器人模型。为了测试策略的泛化能力，需要开发多种地形，包括平地、坡地（10°/20°）和台阶（10cm/15cm）。同时集成域随机化模块，提升控制策略在不同条件下的鲁棒性。</p>
<ol start="2" type="1">
<li><p><strong>拟采取的研究方法、技术路线和/或实验方案</strong></p></li>
</ol>
<p>整个研究按照由简单到复杂的思路开展。首先搭建强化学习训练环境，基于Isaac
Sim和rsl_rl框架配置Go1机器人的平地快走任务。这一阶段的重点是熟悉PPO算法原理，理解训练范式，详细分析velocity
tracking
reward各项含义。单目标基线训练完成后，就有了后续多目标设计的参照。</p>
<p>环境搭建完成后，下一步是扩展地形和增强鲁棒性。具体要实现不同坡度的坡地和不同高度的台阶，同时调整PPO超参数，实现Domain
Randomization，随机化的参数包括质量、摩擦系数、电机延迟等。这一阶段完成时，单目标方法在所有地形上都应该能够正常工作，并通过W&amp;B可视化面板记录完整的训练过程。</p>
<p>有了单目标基线之后，就可以着手多目标强化学习的实现。在rsl_rl中设计4个Reward函数，分别对应速度跟踪、能耗、稳定性和动作平滑。然后实现权重向量采样机制，让系统能够训练出对应不同偏好的策略。为了加速多策略探索，采用Warm-start技术，用已有策略初始化新策略的训练。</p>
<p>最后是实验验证与分析。在统一测试集上收集所有策略的4目标得分，绘制Pareto
Front图表来展示不同策略在多目标空间的分布情况。消融实验方面要通过移除或调整不同奖励项，可以分析各项对步态和稳定性的具体影响。将多目标方法与传统单目标方法进行对比，验证所提方法的优势。</p>
<p><img src="media/image1.png"
style="width:5.8in;height:2.04097in" /></p>
<p>图1　技术路线图</p>
<ol start="3" type="1">
<li><p><strong>预期完成目标</strong></p></li>
</ol>
<p>通过本课题研究，预期构建一套完整的多目标强化学习控制框架，实现4个关键奖励函数和权重采样机制。仿真平台方面，要搭建覆盖平地、坡地、台阶等典型场景的高保真环境。实验验证方面，要绘制出Pareto前沿图，并通过消融实验说明各组件的作用。最终形成可复现的实验流程，提交完整的源代码、训练脚本和预训练模型。</p></td>
</tr>
<tr>
<td><p>三、中外文参考文献（主要的文献10篇以上，其中外文文献至少2篇、学术期刊不少于4篇；且应与任务书的参考文件有所区别）</p>
<ol type="1">
<li><p>中商产业研究院. 2025-20
30年中国四足机器人行业前景与市场趋势洞察专题研究报告[R]. 北京:
中商产业研究院, 2024.</p></li>
<li><p>HWANGBO J, LEE J, DOSOVITSKIY A, et al. Learning agile and
dynamic motor skills for legged robots[J]. Science Robotics, 2019,
4(26): eaau5872.</p></li>
<li><p>HAYES C F, RĂDULESCU R, BARGIACCHI E, et al. A practical guide to
multi-objective reinforcement learning and planning[J]. Autonomous
Agents and Multi-Agent Systems, 2022, 36(1): 1-59.</p></li>
<li><p>ROIJERS D M, VAMPLEW P, WHITESON S, et al. A survey of
multi-objective sequential decision-making[J]. Journal of Artificial
Intelligence Research, 2013, 48: 67-113.</p></li>
<li><p>DI CARLO J, WENSING P M, KATZ B, et al. Dynamic locomotion in the
MIT cheetah 3 through convex model-predictive control[C]//2018 IEEE/RSJ
International Conference on Intelligent Robots and Systems (IROS).
Madrid: IEEE, 2018: 1-9.</p></li>
<li><p>KIM D, DI CARLO J, KATZ B, et al. Highly dynamic quadruped
locomotion via whole-body impulse control and model predictive
control[EB/OL]. (2019-09-14)[2025-01-20]. <a
href="https://arxiv.org/abs/1909.06586">https://arxiv.org/abs/1909.06586</a>.</p></li>
<li><p>RUDIN N, HOELLER D, REIST P, et al. Learning to walk in minutes
using massively parallel deep reinforcement learning[C]//Proceedings of
the 5th Conference on Robot Learning. London: PMLR, 2022:
91-100.</p></li>
<li><p>KUMAR A, FU Z, PATHAK D, et al. RMA: Rapid motor adaptation for
legged robots[C]//Proceedings of Robotics: Science and Systems XVII.
Virtual: RSS Foundation, 2021: 1-12.</p></li>
<li><p>TAN J, ZHANG T, COUMANS E, et al. Sim-to-real: Learning agile
locomotion for quadruped robots[C]//Proceedings of Robotics: Science and
Systems XIV. Pittsburgh: RSS Foundation, 2018: 1-10.</p></li>
<li><p>FU Z, KUMAR A, MALIK J, et al. Minimizing energy consumption
leads to the emergence of gaits in legged robots[C]//Proceedings of the
5th Conference on Robot Learning. London: PMLR, 2022: 928-937.</p></li>
<li><p>NGUYEN T T, NGUYEN N D, VAMPLEW P, et al. A multi-objective deep
reinforcement learning framework[J]. Engineering Applications of
Artificial Intelligence, 2020, 96: 103915.</p></li>
<li><p>MOSSALAM H, ASSAEL Y M, ROIJERS D M, et al. Multi-objective deep
reinforcement learning[EB/OL]. (2016-10-09)[2025-01-20]. <a
href="https://arxiv.org/abs/1610.02707">https://arxiv.org/abs/1610.02707</a>.</p></li>
<li><p>WANG J, HU C, ZHU Y. CPG-based hierarchical locomotion control
for modular quadrupedal robots using deep reinforcement learning[J].
IEEE Robotics and Automation Letters, 2021, 6(4): 7193-7200.</p></li>
<li><p>TAN W, FANG X, ZHANG W, et al. A hierarchical framework for
quadruped locomotion based on reinforcement learning[C]//2021 IEEE/RSJ
International Conference on Intelligent Robots and Systems (IROS).
Prague: IEEE, 2021: 8515-8520.</p></li>
<li><p>ZHANG L, QI Z, SHI Y. Multi-objective reinforcement
learning–concept, approaches and applications[J]. Procedia Computer
Science, 2023, 225: 684-693.</p></li>
<li><p>SCHULMAN J, WOLSKI F, DHARIWAL P, et al. Proximal policy
optimization algorithms[EB/OL]. (2017-07-20)[2025-01-20]. <a
href="https://arxiv.org/abs/1707.06347">https://arxiv.org/abs/1707.06347</a>.</p></li>
<li><p>ARACTINGI M, LÉZIART P A, FLAYOLS T, et al. Controlling the
solo12 quadruped robot with deep reinforcement learning[J]. Scientific
Reports, 2023, 13: 11945.</p></li>
</ol></td>
</tr>
</tbody>
</table>
