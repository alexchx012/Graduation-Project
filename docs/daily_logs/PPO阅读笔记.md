PPO：近端策略优化算法

**问题**：普通策略梯度用同一批数据多更新几步会“跑飞”（策略改得太猛）。

**TRPO 思路**：强行限制每次策略变化幅度（用 KL 距离），但实现复杂、不太友好。

**PPO 核心**：把“别改太猛”做进一个更简单的**替代目标**里（最经典的是 **clipping** 版本），从而允许同一批数据做多轮小批量更新。

PPO支持对小批量数据进行多个训练周期的更新：

指的是PPO算法的标准流程

先收集一批交互数据，机器人跑了几千步，把每一步的关键信息收集起来

然后这批数据打乱，切分成很多小批量（几百步），对其反复做多轮训练（为了提升样本利用率）

每条数据都包含：

**观测**：来自传感器/状态估计（关节角、角速度、末端位姿、距离传感器等）

**动作**：策略当时输出的控制命令（比如力矩、关节速度目标、位置增量等）

**奖励**：设计的目标函数对于这一步的反馈（比如到目标更近、能耗更小、避障成功、队形误差更小等）

**下一步观测**：执行动作后环境回到的新状态

机器人“运动函数”在 RL 里通常叫**策略** $π$（pi）。它把你的观测（关节角、速度、末端位姿等）映射成动作（力矩/速度指令/位置增量等）。

PPO规定了怎么用采样到的数据去更新策略参数，并通过 **clip** 机制让每次更新不要把策略改得太猛，从而训练更稳。



一阶优化：只用“梯度”这条一阶信息来做参数更新的优化方法。

最经典的更新公式：
$$
\theta \leftarrow \theta+\alpha \nabla_{\theta} J(\theta)
$$
$θ$（theta）：要学习/优化的参数（比如策略网络的权重）

$α$（alpha）：学习率（每一步更新走多大）

$∇$（nabla）：梯度算子，表示“对参数求导的方向”

$\nabla_{\theta}$（nabla theta）：对参数 θ 求梯度

$J$：我们要最大化的目标函数。在强化学习中，$J$通常表示**期望累计回报**：也就是“机器人按当前策略跑起来，长期平均能拿到多少奖励”。

$J(θ)$（ J of theta）：目标函数在参数 θ 下的取值

$\nabla_{\theta} J(\theta)$并非是目标函数对于$θ$的取值乘以$θ$的梯度，而是对$J$求梯度，是在当前参数 $θ$处，$J$往哪个方向增加最快。

**梯度方向** = 上山（最大增大 $J$）的最陡方向

**学习率 $α$** = 这一步走多大

经过多轮更新后，$θ$会无限接近理论最优吗？**不会**

因为在经典优化理论场景中，如果问题是凸的、梯度精确并且步长满足条件，那么一阶方法可以收敛到全局最优或满足收敛性质。

但是深度强化学习中，这些条件都不满足。策略网络是非凸的、梯度是采样估计（有噪声）、环境可能非平稳（尤其多机器人）、还用到了近似（值函数、优势估计等）。所以通常只能保证**收敛到某个局部最优/稳定点**，甚至有时会卡住或震荡。




$$
\hat{g}=\hat{\mathbb{E}}_{t}\left[\nabla_{\theta} \log \pi_{\theta}\left(a_{t} \mid s_{t}\right) \hat{A}_{t}\right]
$$
$\hat{g}$（ g hat）：策略梯度的采样估计值（用一批数据算出来的“更新方向”）

$\hat{\mathbb{E}}_{t}$（ E hat t）：对采样到的时间步 $t$做经验平均（batch 平均），不是理论精确期望

$∇$（nabla）：梯度算子

$\nabla_{\theta}$（nabla theta）：对参数 $θ$求导

$⁡\log$：自然对数（对数形式是策略梯度常用写法）

$\pi_{\theta}$（ pi theta）：参数为$θ$的策略（输出动作的概率分布）

$a_{t}$：时间步 $t$采取的动作（如力矩/速度目标/位置增量）

$∣$（ given / 条件于）：条件符号

$s_{t}$：时间步 $t$的状态/观测（来自传感器/状态估计）

$\hat{A}_{t}$（A hat t）：优势函数的采样估计，衡量该动作相对“平均水平”的好坏

$t$：时间步索引（也可理解为采样序号）



首先采样一段时间内的时间步$t$，比如1~1000

然后每一个$t$，都会有一项梯度贡献，也就是向量$\nabla_{\theta} \log \pi_{\theta}\left(a_{t} \mid s_{t}\right)$。这里 $\nabla$对 $\theta$求导得到的是**向量**，代表“把策略参数往哪个方向推，会更倾向于在状态$s_t$下输出动作 $a_t$”，同时并行计算标量$\hat{A}_{t}$，一般是用这一步的“实际好坏”减去“基线预测”算出来的，比预期好就正，比预期差就负。

如果$\hat{A}_{t}>0$，说明这一步动作“比基线预期好”，会让更新方向倾向于**提高**在 $s_{t}$下选 $a_{t}$的倾向（或概率密度）；如果小于0的话，那就会**降低**这个倾向。

然后将每个 $t$产生一个梯度贡献向量（被标量$\hat{A}_t$缩放过的向量），把这一批的贡献向量做平均，得到$\hat{g}$这个**总体梯度估计向量**，最后用这个$\hat{g}$去更新参数$θ$

**总结：对每个时间步 $t$，计算梯度向量$\nabla_{\theta} \log \pi_{\theta}\left(a_{t} \mid s_{t}\right)$并用优势标量$\hat{A}_{t}$加权，得到该步的梯度贡献向量；对一批样本做经验平均得到$\hat{g}$，作为策略参数$θ$的更新方向。**



在机器人运动策略的优化中，我们想要最大化的是“长期期望回报”，也就是在规定时间段内机器人按当前策略运动所拿到的奖励最多。但是这个“长期期望回报”难以对其进行优化，因为**对策略参数$\theta$求梯度时不仅环境是动态且未知的，回报也依赖整条轨迹分布**，所以需要构造一个函数来代替真实目标做优化。用采样数据构造一个**容易算、容易求梯度、并且在更新方向上等价**的目标函数，这个函数就叫代理目标函数$L^{P G}(\theta)$。在策略梯度里，最大化$L^{P G}(\theta)$的梯度所给出的更新方向与提高“长期期望回报”的方向采样近似，所以这个函数叫代理函数。
$$
L^{P G}(\theta)=\hat{\mathbb{E}}_{t}\left[\log \pi_{\theta}\left(a_{t} \mid s_{t}\right) \hat{A}_{t}\right]
$$

$$
\hat{g}=\hat{\mathbb{E}}_{t}\left[\nabla_{\theta} \log \pi_{\theta}\left(a_{t} \mid s_{t}\right) \hat{A}_{t}\right]
$$

$$
\hat{g}=\nabla_{\theta} L^{P G}(\theta)
$$

代理目标函数$L^{P G}(\theta)$对参数$θ$求梯度就能得出$\hat{g}$，这个$\hat{g}$（向量）就是我想要最大化$L^{P G}(\theta)$所需要前进的方向，后续真正的更新还需乘学习率$\alpha$，并且PPO算法还会加**clip/KL**去限制一次更新不要走太大。

代理目标函数的目标就是使$\pi_{\theta}$这个策略在优势$\hat{A}_{t}$为正的动作$a_{t}$上提高概率、为负的动作上降低概率。

- 当 $\hat{A}_t$> 0：在状态 $s_t$下，提高动作 $a_t$的概率$\pi_\theta(a_t\mid s_t)$
- 当 $\hat{A}_t$< 0：在状态 $s_t$下，降低动作 $a_t$的概率$\pi_\theta(a_t\mid s_t)$



信任域方法 TRPO

最大化一个代理目标函数，同时对策略更新的大小施加约束，与PPO不同的是TRPO需要对待优化目标进行线性近似以及对约束进行二次近似后，可以使用共轭梯度算法高效地近似求解该问题。（比PPO复杂）
$$
目标函数：\underset{\theta}{\operatorname{maximize}} \hat{\mathbb{E}}_{t}\left[\frac{\pi_{\theta}\left(a_{t} \mid s_{t}\right)}{\pi_{\theta_{\text {old }}}\left(a_{t} \mid s_{t}\right)} \hat{A}_{t}\right]
$$
$\underset{\theta}{\operatorname{maximize}}$（对$\theta$最大化）：当前要更新的策略参数（神经网络权重）

$\hat{\mathbb{E}}_{t}$（ E hat t）：对采样到的时间步 $t$做经验平均

$\pi$：策略（动作概率分布/密度）

$\pi_{\theta}$:新策略在状态$s_{t}$下选动作$a_{t}$的概率

$\pi_{\theta_{\text {old }}}\left(a_{t} \mid s_{t}\right)$：旧策略在同一状态下选该动作的概率

$\hat{A}_{t}$（A hat t）：优势估计（该动作相对“平均水平”的好/坏程度）

现在这批数据是用旧策略$\pi_{\theta_{\text {old }}}$采样出的（数据分布属于旧策略），但是我现在想更新到新策略$\pi_{\theta}$，所以需要一个“概率比值”$\frac{\pi_{\theta}\left(a_{t} \mid s_{t}\right)}{\pi_{\theta_{\text {old }}}\left(a_{t} \mid s_{t}\right)}$来衡量**新策略相对于旧策略，对这次的采样动作是更支持还是反对**

同时再乘上优势估计$\hat{A}_{t}$：

- 当 $\hat{A}_t$> 0：目标会鼓励你让**比值**变大 → 新策略更倾向于选这个动作
- 当 $\hat{A}_t$< 0：目标会鼓励你让**比值**变小 → 新策略减少选这个动作

**目标函数是在用旧数据推动新策略更偏向好动作，远离坏动作**


$$
约束函数：   \text { subject to } \quad \hat{\mathbb{E}}_{t}\left[\operatorname{KL}\left[\pi_{\theta_{\text {old }}}\left(\cdot \mid s_{t}\right), \pi_{\theta}\left(\cdot \mid s_{t}\right)\right]\right] \leq \delta \text {. }
$$
$\text { subject to }$（满足约束）：表示下面是约束条件

$\operatorname{KL}$：KL 散度，用来衡量两个策略分布的差异

$⋅$（dot）：占位符，表示“对所有动作的分布”（不是某个具体动作）

$\operatorname{KL}\left[\pi_{\theta_{\text {old }}}\left(\cdot \mid s_{t}\right), \pi_{\theta}\left(\cdot \mid s_{t}\right)\right]$：**在同一状态下，新旧策略整个位于“动作分布层面”的差异**

$\delta$（delta）：信赖域半径/阈值（允许策略变化的最大程度）

约束函数要求采样到的状态$s_{t}$上，新旧策略的分布差异（$\operatorname{KL}$）做平均后 **不能超过$\delta$**。

**信任域思想**：可以改策略，但是每次只能改一小步。因为用的是旧策略来估计更新方向，如果一次性改的太大，这种估计会失真、训练会不稳。



设$r_t(\theta)$为概率比，$r_t(\theta)=\frac{\pi_{\theta}\left(a_{t} \mid s_{t}\right)}{\pi_{\theta_{\text {old }}}\left(a_{t} \mid s_{t}\right)}$，当$\theta$为$\theta_\text{old}$时，分子分母就是一个东西，所以比值恒等于1，$r_t(\theta_\text{old})=1$
$$
L^{C P I}(\theta)=\hat{\mathbb{E}}_{t}\left[\frac{\pi_{\theta}\left(a_{t} \mid s_{t}\right)}{\pi_{\theta_{\text {old }}}\left(a_{t} \mid s_{t}\right)} \hat{A}_{t}\right]=\hat{\mathbb{E}}_{t}\left[r_{t}(\theta) \hat{A}_{t}\right] .
$$
$L^{C P I}(\theta)$：CPI是保守策略迭代的缩写，其每次更新策略都要“保守一点”

现有旧策略$\pi_{\theta_{\text {old }}}$采到一批数据（$s_{t}$,$a_t$），现在需要跟新到新策略$\pi_{\theta}$，但是由于成本原因不想重新采样，所以用$r_t(\theta)$做**重要性采样校正**：如果新策略比旧策略更喜欢这个动作（$r_t>1$），那就把这条样本的影响放大，反之缩小。

同时乘上优势$\hat{A}_{t}$：

- $\hat{A}_{t}>0$（这步动作比平均好）：最大化目标会倾向于让$r_t(\theta)$变大 → **提高这个动作在该状态下的概率**
- $\hat{A}_{t}<0$（比平均差）：会让$r_t(\theta)$变小 → **降低概率**



**PPO-Clip核心目标函数：**
$$
L^{C L I P}(\theta)=\hat{\mathbb{E}}_{t}\left[\min \left(r_{t}(\theta) \hat{A}_{t}, \operatorname{clip}\left(r_{t}(\theta), 1-\epsilon, 1+\epsilon\right) \hat{A}_{t}\right)\right]
$$
$\min(·,·)$：取两者中较小的

$\operatorname{clip}\left(·, 1-\epsilon, 1+\epsilon\right)$：把输入截断到区间$\left[1-\epsilon, 1+\epsilon\right]$内

$1-\epsilon, 1+\epsilon$：允许的变化范围上下界、

$\epsilon$：截断系数

$r_t(\theta)$是新策略改变幅度，当其大于1时新策略在该状态下更“喜欢”这个动作，小于1就是更不喜欢

$\operatorname{clip}\left(r_t(\theta), 1-\epsilon, 1+\epsilon\right)$的作用就是当新策略改变如果大于$1+\epsilon$时就当成$1+\epsilon$，反之亦然。将策略改变的最大幅度限制在截断系数内

PPO的目标不是直接最大化$r_t(\theta)\hat{A}_{t}$，而是把它和裁剪后的版本做对比，**取更小的那个**。

- 当更新在合理范围内时（变化小于截断系数），clip不起作用，目标就退化成普通的策略梯度代理目标。
- 当更细的太激进时（变化大于截断系数），$\min$会选择clip的版本，即变化到截断系数为止



假设$\epsilon=0.2$，clip区间为$[0.8,1.2]$

现有四种情况，当$\hat{A}_{t}=1$时，$r_t(\theta)=0.6或1.4$；当$\hat{A}_{t}=-1$时，$r_t(\theta)=0.6或1.4$

样本级PPO目标函数为：
$$
L_t(\theta)=\min \left(r_{t}(\theta) \hat{A}_{t}, \operatorname{clip}\left(r_{t}(\theta), 1-\epsilon, 1+\epsilon\right) \hat{A}_{t}\right)
$$
**Case 1：$\hat{A}_{t}=1；r_t(\theta)=1.4$**

$\because\epsilon=0.2\:\therefore\operatorname{clip}\left(r_{t}(\theta), 1-\epsilon, 1+\epsilon\right)=\operatorname{clip}\left(1.4, 0.8, 1.2\right)=1.2$

$\therefore L_t(\theta)=\min\left(1.4\times1,1.2\times1\right)=\min\left(1.4,1.2\right)=1.2$

因为$\hat{A}_{t}=1$，所以这是个好动作；但是我把概率提的太大，结果被PPO按在1.2处

**Case 2：$\hat{A}_{t}=-1；r_t(\theta)=0.6$**

$\because\epsilon=0.2\:\therefore\operatorname{clip}\left(r_{t}(\theta), 1-\epsilon, 1+\epsilon\right)=\operatorname{clip}\left(0.6, 0.8, 1.2\right)=0.8$

$\therefore L_t(\theta)=\min\left(0.6\times-1,0.8\times-1\right)=\min\left(-0.6,-0.8\right)=-0.8$

因为$\hat{A}_{t}=-1$，所以这是个坏动作；但是我把它概率降太小（$r_{t}(\theta) \hat{A}_{t}=-0.6$）,所以相对于被截断项的-0.8这是在变好。导致PPO在这会取更小、更保守的-0.8，**禁止超过阈值（截断项）的“额外变好”**

**Case 3：$\hat{A}_{t}=1；r_t(\theta)=0.6$**

$\because\epsilon=0.2\:\therefore\operatorname{clip}\left(r_{t}(\theta), 1-\epsilon, 1+\epsilon\right)=\operatorname{clip}\left(0.6, 0.8, 1.2\right)=0.8$

$\therefore L_t(\theta)=\min\left(0.6\times1,0.8\times1\right)=\min\left(0.6,0.8\right)=0.6$

因为$\hat{A}_{t}=1$，所以这是个好动作；但最后结果是0.6，这表明我把概率降太小了，这会让目标变差（对应论文page3. “而在其使目标变差时则予以保留”）

**Case 4：$\hat{A}_{t}=-1；r_t(\theta)=1.4$**

$\because\epsilon=0.2\:\therefore\operatorname{clip}\left(r_{t}(\theta), 1-\epsilon, 1+\epsilon\right)=\operatorname{clip}\left(1.4, 0.8, 1.2\right)=1.2$

$\therefore L_t(\theta)=\min\left(1.4\times-1,1.2\times-1\right)=\min\left(-1.4,-1.2\right)=-1.4$

因为$\hat{A}_{t}=-1$，所以这是个坏动作；但我把概率$r_t(\theta)$提的太大，这会让目标变得更差，此时PPO会保留更差的这一项

此时引出一个疑问，**为什么PPO算法会惩罚“变得更好”而不惩罚“变得更坏”**

其实PPO并没有惩罚“变得更好”，而是不允许越界后的额外变好；同时变坏是一定会被压回正常区间内，只不过并没有在目标函数中体现。

为什么不允许越界后的额外变好？原因是策略梯度用的是旧策略采样的数据，当新策略离旧策略太远时，用旧数据估出来的“这步更新能变好多少”，会变得很不可信。

那么$case3/4$是如何被压回正常区间的呢？

**体现在目标函数对参数$\theta$的梯度里**，我们得看$\nabla_{\theta}L^{C L I P}(\theta)$在当前case下会把策略往哪推

$Case\:3$：$\hat{A}_{t}=1；r_t(\theta)=0.6$

$L^{C L I P}(\theta)$在这条样本上起作用的是未被截断的$r_t(\theta)=0.6$在起作用，它并没有被clip按住，这就是会被压回正常区间的来源：**梯度并未被剪掉。**

此时样本级目标为$r_t(\theta)$，样本级目标函数为$L_t=r_t\hat{A}_{t}$

对$r_t$求导：$\frac{\partial L_{t}}{\partial r_{t}}=\hat{A}_{t}=1$，我的目标是让目标函数最大化，那么结果是正导数意味着**优化会推动$r_t(\theta)$变大**

$Case\:4$：$\hat{A}_{t}=-1；r_t(\theta)=1.4$

对$r_t$求导：$\frac{\partial L_{t}}{\partial r_{t}}=\hat{A}_{t}=-1$，导数为负意味着我在最大化目标函数时，会倾向于把$r_t$往小推，让其小于1.4，最好是小于1（减少坏动作的概率）



自适应KL惩罚系数
$$
L^{K L P E N}(\theta)=\hat{\mathbb{E}}_{t}\left[\frac{\pi_{\theta}\left(a_{t} \mid s_{t}\right)}{\pi_{\theta_{\text {old }}}\left(a_{t} \mid s_{t}\right)} \hat{A}_{t}-\beta \operatorname{KL}\left[\pi_{\theta_{\text {old }}}\left(\cdot \mid s_{t}\right), \pi_{\theta}\left(\cdot \mid s_{t}\right)\right]\right]
$$
$L^{K L P E N}(\theta)$：带KL惩罚的代理目标函数

$\hat{\mathbb{E}}_{t}$：对采样到的时间步 $t$做经验平均（batch 平均）

$\frac{\pi_{\theta}\left(a_{t} \mid s_{t}\right)}{\pi_{\theta_{\text {old }}}\left(a_{t} \mid s_{t}\right)}$：概率比$r_t(\theta)$

$\pi_{\theta}$：新策略

$\pi_{\theta_{\text {old }}}$：旧策略

$\beta$：KL惩罚系数

$\operatorname{KL}\left[ \;， \right]$：KL散度，衡量两个分布差异

$\pi_{\theta_{\text {old }}}\left(\cdot \mid s_{t}\right)$：旧策略在状态$s_t$下对所有动作的分布

$\pi_{\theta}\left(\cdot \mid s_{t}\right)$：新策略在状态$s_t$下对所有动作的分布

这里把PPO里的$r_{t}(\theta) \hat{A}_{t}$提到前面来，把clip和min发挥的作用放到后面这个$-\beta \operatorname{KL}\left[\pi_{\theta_{\text {old }}}\left(\cdot \mid s_{t}\right), \pi_{\theta}\left(\cdot \mid s_{t}\right)\right]$中

在自适应KL版PPO中$\beta$是实时改变的，所以称之为自适应。如何自适应呢？

先计算实际KL：$d$
$$
d=\hat{\mathbb{E}}_{t}\left[\operatorname{KL}\left[\pi_{\theta_{\text {old }}}\left(\cdot \mid s_{t}\right), \pi_{\theta}\left(\cdot \mid s_{t}\right)\right]\right]
$$
如果$d<d_{\text {targ}}/1.5$，那么就把$\beta/2$将其减半；这会让KL惩罚变弱，下一轮更新更大胆（$d_{\text {targ}}$为目标KL，是超参数）

如果$d>d_{\text {targ}}\times1.5$，同理将$\beta\times2$，使其加倍；这会让KL惩罚更强，下一轮更新会更保守

至于为什么乘除都是1.5，这是经验上的“容忍带”，只有当偏离超过大约50%时，才会对$\beta$做大幅调整



**PPO实际训练时真正优化的总目标：**
$$
L_{t}^{C L I P+V F+S}(\theta)=\hat{\mathbb{E}}_{t}\left[\underbrace{L_{t}^{C L I P}(\theta)}_{\text{Actor/策略优化}}-c_{1} \underbrace{L_{t}^{V F}(\theta)}_{\text{Critic/价值拟合}}+c_{2} \underbrace{S\left[\pi_{\theta}\right]\left(s_{t}\right)}_{\text{熵正则化}}\right]
$$
$L_{t}^{C L I P+V F+S}(\theta)$：PPO的总目标（策略+价值函数+熵）

$L_{t}^{C L I P}(\theta)$：PPO的截断策略代理目标

$c_1$：值函数损失权重系数

$L_{t}^{V F}(\theta)$：值函数损失

$VF$：价值函数（value function）

$c_2$：熵奖励权重系数

$S\left[\pi_{\theta}\right]\left(s_{t}\right)$：策略在状态$s_t$下的熵

$S$：熵奖励

$\pi_\theta$：参数为$\theta$的策略分布

$s_t$：时间步为$t$的状态

**Actor（执行者）$\equiv$  策略（Policy，$\pi_\theta$）。**Actor是策略的参数化（神经网络）表示。其输入为状态$s_t$，输出为动作$a_t$

**Critic（评论家）$\equiv$  价值函数（Value Function）。**Critic是价值函数的参数化表示。其输入为状态$s_t$，输出为该状态的标量价值$V(s_t)$，即预期未来回报。

这个函数把三件事结合在一起学：

**1.学策略**：用$L_{t}^{C L I P}(\theta)$做稳定的策略改进

**2.学价值函数**：利用$L_{t}^{V F}(\theta)$将$V_\theta(s_t)$尽可能靠近$V_{t}^{targ}$

**3.保持探索**：用熵奖励$S\left[\pi_{\theta}\right]\left(s_{t}\right)$防止策略过早变得“过于确定”

第一项$L^{CLIP}$让策略变好的同时别进步太猛，第二项$L^{VF}$把价值函数误差当做惩罚项（所以前面是减号），第三项$S$奖励更高的熵，鼓励算法继续探索以免过早收敛到局部最优

> 实现里常见写法是“最小化 loss”，那就会把整体取负号，变成最小化$-L^{CLIP}+c_1L^{VF}-c_2S$。论文这里写的是“最大化”的形式。



$$
L_{t}^{V F}(\theta)=\left(V_{\theta}\left(s_{t}\right)-V_{t}^{\text {targ }}\right)^{2}
$$
$V_\theta(s_t)$：价值函数。是Critic的预测：在当前状态$s_t$下，如果接下来继续按当前策略$\pi_\theta$走，未来长期总奖励。大概有多少

$V_{t}^{\text {targ }}$：价值目标或者叫监督标签（target），是根据采样轨迹算出来的更接近真实回报的目标值

**$V_\theta(s_t),V_{t}^{\text {targ }},\hat{A}_{t}$三者的关系：**
$$
V_{t}^{\operatorname{targ}}=\hat{A}_{t}+V_{\theta}\left(s_{t}\right)
$$
$\hat{A}_{t}$上面说过叫优势估计，用于衡量该动作比价值函数预测的“平均水平”好多少。如果大于0，新策略更倾向于选这个动作；如果小于0，那么新策略更倾向于不选这个动作。可以看做$\hat{A}_{t}\approx 实际回报-基线预测$，如果实际回报大于基线预测，那就说明这是个好动作（大于0），反之亦然。

所以$实际回报\approx\hat{A}_{t}+基线预测$，所以$V_{t}^{\operatorname{targ}}=实际回报,V_\theta(s_t)=基线预测$

上面说过我们要最大化$L_{t}^{C L I P+V F+S}(\theta)$，那么既然是$-c_1L_{t}^{V F}(\theta)$，那就需要最小化$L_{t}^{V F}(\theta)$。又因为$L_{t}^{V F}(\theta)$是$V_\theta(s_t)$与$V_{t}^{\text {targ }}$的平方误差，所以我们的**目标就是让这两个数的差值趋近于0**，等同于实际回报$\approx$基线预测



**如何计算$\hat{A}_{t}$**：利用TD（Temporal Difference）残差+截断版GAE（Generalized Advantage Estimation）

TD：时间差分     GAE：广义优势估计
$$
\delta_{t}=r_{t}+\gamma V\left(s_{t+1}\right)-V\left(s_{t}\right)
$$
$\delta_t$：TD残差

$r_t$：第t步的即时奖励

$\gamma$：折扣因子（未来奖励打几折）

$V(s_t)$：第$t$步的价值函数

$V\left(s_{t+1}\right)$：第$t+1$步的价值函数

价值函数$V$在状态$s_t$时给一个预测：从现在开始未来总奖励大概是多少

走了一步之后（$t+1$）得到两样东西：

- 这一步的即时奖励$r_t$

- 到了下一步$t+1$的状态，又可以用价值函数进行预测未来总奖励$V(s_{t+1})$

那么现在我就得到了一个“走一步之后的更现实的估计”：现在立刻拿到的分$r_t$+下一步开始的未来分（打折后）$\gamma V\left(s_{t+1}\right)$。TD残差就是把“走了一步之后的更现实的估计”与“上一步的未来估计”做差，也就是预测误差。

假设：

$V(s_t)=10$：原预测未来总奖励

$r_t=1$：这一步拿到了1分

$V\left(s_{t+1}\right)=12$：走了一步后再预测未来总奖励

$\gamma=0.9$：折扣因子
$$
\delta_{t}=r_{t}+\gamma V\left(s_{t+1}\right)-V\left(s_{t}\right)=1+0.9\times12-10=11.8-10=1.8
$$
也就是说我一开始预计未来总奖励为10，结果走了一步之后再算（这一步的奖励+从这一步开始未来总奖励）发现比上一步（10）还要好，那么上一步的偏差就是1.8，比原来预计的要好1.8分。



计算$\hat{A}_{t}$只用TD残差还不够，因为它只看了一步，对于长期的好坏完全没算进去，所以需要截断版GAE来看得更远：用一串TD残差做加权求和，得到更平滑、更稳的优势估计$\hat{A}_{t}$
$$
\hat{A}_{t}=\delta_{t}+(\gamma \lambda) \delta_{t+1}+\cdots+(\gamma \lambda)^{T-t+1} \delta_{T-1}
$$
$\delta_t$：TD残差

$\gamma$（gamma）：折扣因子

$\lambda$（lambda）：GAE参数

$T$：轨迹片段长度

$t$：当前时间步索引

什么叫截断版：由于PPO是一个策略跑$T$步再更新，所以优势估计不能看超过第$T$步，GAE的求和必须在片段末尾被截停

需要注意的是$(\gamma\lambda)$是有幂次的，$\hat{A}_{t}$可以写成：$\hat{A}_{t}=(\gamma \lambda)^0\delta_{t}+(\gamma \lambda)^1 \delta_{t+1}+\cdots+(\gamma \lambda)^{T-t+1} \delta_{T-1}$

幂次=从当前步$t$到那一项$\delta$所在的时间步的距离

- 当前项$\delta_{t}$的距离是0，所以系数是$(\gamma \lambda)^0=1$
- 下一项$\delta_{t+1}$的距离是1，所以系数是$(\gamma \lambda)^1$
- 再下一项的距离是2，所以系数是$(\gamma \lambda)^2$

在PPO中，$\gamma$和$\lambda$是常数超参数。**$\gamma$负责“未来奖励打折”：离现在越远，影响越小；$\lambda$负责“GAE看多远”，离现在越远，被信任程度越低**



PPO算法一次迭代顺序：

1.采样：用旧策略$\pi_{\theta_{\text {old }}}$跑一段数据（$s_t$，$a_t$，$r_t$...）

2.计算价值预测$V_\theta(s_t)$：因为接下来算$\delta_t$和$\hat{A}_{t}$都要用到$V\left(s_{t}\right)$和$V\left(s_{t+1}\right)$

3.计算优势$\hat{A}_{t}$：Actor更新的权重就是$\hat{A}_{t}$，而它依赖价值函数提供“基线”。（计算$\hat{A}_{t}$前先计算$\delta_t$）

4.构造价值目标$V_{t}^{\text {targ }}$：基线$V_\theta(s_t)$+优势$\hat{A}_{t}$

5.最后一起优化：策略项用$\hat{A}_{t}$去推策略（PPO-clip）；价值项用$V_{t}^{\text {targ }}$去回归$V_\theta$

需要注意的是在工程实现时：**在多轮epoch更新时，$\hat{A}_{t}$和$V_{t}^{\text {targ }}$通常会被当做固定标签，不能被梯度穿透，否则目标会变得不稳定**



PPO算法伪代码：

1**for** interation=1,2,... **do**

2    **for** actor=1,2,...,*N* **do**

3        用旧策略$\pi$，参数为$\theta_{old}$的版本，在环境里跑T步

4        对刚刚这段长度为T的轨迹，计算优势$\hat{A}_{t}$

5    **end for**

6    在这NT条样本上，构造代理函数$L_\theta$，然后做K个epoch的小批量优化，每个mini-batch的大小是M，并且$M\le NT$

7    $\theta_{old}\gets\theta$

8**end for**

第一行：这是PPO的迭代次数，每次iteration都包含用当前策略采样的一批轨迹、用这批轨迹把策略/价值网络更新若干步

第二行：这是并行环境进程，N越大，每一次iteration采样的数据就越多

第三行：用旧策略的版本在环境里跑T步，每一一个时间步会存一条transition，至少包括$s_t,a_t,r_t,V(s_t)$等

第四行：用TD残差+GAE计算优势$\hat{A}_{t}$

第六行：执行PPO-clip算法（$\theta_{old}$在一次iteration的K轮优化里保持不变）

第七行：把$\theta_{old}$更新成当前最新的$\theta$
