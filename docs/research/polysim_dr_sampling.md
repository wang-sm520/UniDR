# PolySim 训练与 DR 采样：论文与源码证据核对

状态：论文与固定提交源码研究笔记，未运行 PolySim；不是物理效果验收或后端支持承诺。
核对日期：**2026-09-12，UTC+08:00**。
本轮只新增本文，不修改既有研究笔记、运行时、配置或依赖。

## 结论摘要

**源码结论：已核对的公开训练路径不保证同一次 PPO 更新内跨仿真器 DR 实际值相等。**
启用对应开关时，KP/KD 等在各服务器的局部环境 reset 中自行抽样；PPO 的
24-step rollout 会包含不同环境、不同 episode 的参数组合，step RPC 只传动作，
没有集中抽样后广播 DR 张量的路径。不能将“本地抽样”升级成严格统计独立性保证。
另外，README 的并行训练示例明确选择 `NO_domain_rand`，并未启用参数级 DR。
详细代码证据见“固定提交源码核对”一节。[README][code-readme] [reset DR][code-reset-dr]

**仅凭论文，不能断言“同一次 PPO update 内，所有仿真器的 DR 实际采样值完全相同”。**
论文明确描述单策略、每个训练迭代内的多引擎并行 rollout、统一初始化规格及接口时序；
但没有给出跨引擎逐环境配对抽样、随机种子共享、DR 重采样时机或 reset 同步协议。
因此也不能反过来声称论文证明了各引擎独立抽样。[方法][method] [路由器][router]

尤其需要区分：**共享配置/参数范围、共享实际抽样值、共享采集/更新时间、共享 reset 时间**
是四个不同命题。论文中的统一物理规格，甚至还不足以核实逐字段 DR 分布和范围完全相同；
“synchronized clocks”也不是“所有环境同时 reset 并重抽同一组参数”。[路由器][router]

## 来源与证据边界

- 论文：*PolySim: Bridging the Sim-to-Real Gap for Humanoid Control via Multi-Simulator Dynamics Randomization*。
- 附件首页标注 **arXiv:2510.01708v3，2025-10-14**；`pdfinfo` 确认共 **8 页**。
  以下页码均为 PDF 文件页序，从首页开始计数。[官方论文][paper] [官方 PDF][pdf]
- 已用 `pdftotext -layout` 将[本地附件][attachment]全文输出到 stdout，并分段复核第 1 至 8 页；
  官方 v3 HTML 的方法、实验段落及表格标题与附件交叉核对。
- 论文核对与源码核对分别进行。`git ls-remote` 确认官方仓库当前 HEAD 为
  `0fdb349785479b5c0c30d3c676fae67025b8d34a`，源码使用该固定提交的 raw 文件。
  这个公开提交不是论文实验所用提交的证明，不能从仓库默认配置还原所有论文实验。
- 既有 `adaptive_simulator_mixture.md` 仅用于匹配文档风格，不作为本轮论文结论的证据。
  本轮未另外阅读 ASAP 论文、补充视频或其他未提供附件，不能把它们的细节移植为 PolySim 事实。

证据标签：**论文表述**是作者明确写出的内容；**概念区分**是解释其逻辑边界；
**未说明**限定于已核对的 v3 PDF/HTML，不等于实际代码不存在该行为。

## 共享范围、配对抽样与同步

| 待核对命题 | 论文中的直接证据 | 能否据此确认 |
| --- | --- | --- |
| 多引擎共同训练同一策略 | 第 3 页，III 节：“optimizing a single policy against a mixture of heterogeneous dynamics within each training iteration”。[原文][method] | **论文表述：是。** 是每个训练迭代内的联合训练，不只是按阶段换引擎。 |
| 初始化尽量使用一致的物理规格 | 第 3 页，III-B：“a unified physical specification”；场景要求是“as consistent as possible”。[原文][harmonization] | **论文表述：是。** 这是统一规格及尽量对齐，不是所有物理行为完全相同。 |
| 所有引擎的每个 DR 字段、分布及范围完全相同 | III-B 列出物理映射类别，但全文没有逐字段 DR 范围或分布表。[原文][harmonization] [DR 表说明][dr-results] | **未说明。** 不能把统一初始化规格升级成逐字段 DR 分布已经核实相等。 |
| 对应环境跨引擎拿到完全相同的 DR 抽样值 | 未给出环境配对、共同随机数、集中抽样后广播或参数向量比较协议。[方法][method] [路由器][router] | **未说明。** 相同范围即使成立，也不意味着相同抽样值；独立抽样同样未被证明。 |
| 同一次 PPO update 对应唯一、固定的一组 DR 值 | 没有 DR 抽样循环、PPO rollout 长度或 DR 生命周期说明。[方法][method] [实验设置][sim-experiments] | **未说明。** 没有证据表明 DR 以 update 为重采样单位，或在整段 rollout 中不变。 |
| 多引擎具有一致的接口时序 | 第 3 页，III-B：“synchronized clocks”；段末还有“identical tensor shapes, semantics, and timing”。[原文][router] | **论文表述：是，限于接口设计。** 未说明具体 step 屏障、超时或更新屏障实现。 |
| 对应环境同步终止/reset/重采样 | 没有跨引擎共享 done、同步 reset 或统一 episode 边界的流程。[方法][method] [路由器][router] | **未说明。** 同步时钟不推出同步 episode 生命周期。 |

**概念区分：** 设 `D_i` 是引擎 `i` 的参数分布，`xi[i,e,k]` 是它的环境 `e`
在第 `k` 次抽样事件取得的参数向量。`D_i = D_j` 只是同分布；
逐环境配对要求 `xi[i,e,k] = xi[j,e,k]`，还需要明确环境映射和抽样事件对齐。
这里的 `k` 不预设等于 PPO update、episode 或 reset 次数。论文未提供这个映射。
同一 PPO 更新使用多个来源的数据，在逻辑上也不要求全部 transition 共享同一组物理参数。
以上是概念解释，不是论文给出的算法或对实际实现的判断。

## 论文公开的训练细节

### 1. 训练组织与算法边界

- **论文表述：** 第 3 页 III-A 将训练和仿真分为 `TrainClient` 与多个 `SimServer`。
  每个服务器推进本地物理并返回 `(o, r, d, info)`；客户端
  “aggregates trajectories and updates the policy”。这是聚合轨迹后更新单策略的描述，
  不是多个独立策略训练完成后平均权重。[III-A][isolation]
- **论文表述：** 第 4 页 Fig. 3 给出统一 scene/agent/task 配置、观测和奖励计算、
  网络产生动作、优化器更新的系统示意；router 把统一初始化配置映射到引擎设置。[Fig. 3][overview]
- **未说明：** v3 没有编号的训练算法/DR 伪代码，也没有 PPO 超参数表。
  第 8 页参考文献 [6] 是 PPO 论文，但它在第 2 页 II-A 的相关工作中被引用，
  不能仅凭这条引用推导 PolySim 实验的具体 PPO 配方。[II-A][related-control] [参考文献 6][ppo-reference]
- **未说明：** rollout 步数、各引擎环境数量与配比、minibatch 数、每轮优化 epoch 数、
  actor/critic 架构、学习率、GAE、clip、entropy、优势归一化及种子均没有在该论文中列出。
  本文不从已有笔记或 ASAP 设置补写这些数值。[方法][method] [实验设置][sim-experiments]
- **概念区分：** 第 4 页 IV 节 Definition 2 写出混合 transition kernel
  `T_Q(. | s,a) = sum_i w_i * T_i_tilde(. | s,a)`。
  这个分布层面的理论定义不是 DR 抽样器，也没有给出引擎配额、权重更新或参数配对机制；
  不能据此断言实现每步重新选择引擎或每次 PPO 更新统一重抽参数。[IV 节][theory]

### 2. 任务、训练时长和对照设置

| 项目 | 论文公开内容及限定 |
| --- | --- |
| 任务 | 第 5 页 V-A.1：从 ASAP 数据集中选出 **14 个** easy/medium/hard 动作进行 motion tracking；未逐一列出动作名称或各难度数量。[设置][sim-experiments] |
| 训练迭代 | 第 5 页 V-A.1：“10k, 15k, and 20k iterations, respectively”，分别对应 easy、medium、hard；没有给出每个 iteration 的环境步数或梯度步数。[设置][sim-experiments] |
| 训练引擎 | 第 5 页 V-A.1：IsaacGym、IsaacSim、Genesis 的并行训练组合；第 6 页 Table I 包含三个单引擎、三个双引擎和一个三引擎组合。[设置][sim-experiments] [Table I][main-results] |
| 未见域评估 | 第 5 页 V-A.1：MuJoCo 是 entirely unseen simulator，进行 zero-shot transfer。Fig. 3 虽画出 MuJoCo，不能据此说这些实验把它用于训练。[设置][sim-experiments] [Fig. 3][overview] |
| 机器人 | 第 5 页 V-A.2 和第 7 页 V-F：真实机器人是 **Unitree G1**，部署挑战性全身动作；没有给出精确关节/动作维数。[实机设置][real-experiments] [部署][deployment] |
| sequential 对照 | 第 5 页 V-B：依次使用两或三个引擎，比较不同顺序，使用文中所述相同迭代数；下一阶段继承上一阶段的 policy 和 curriculum learning parameters。[基线][baselines] |
| parallel/sequential 公平性 | 第 6 页 V-D：比较 **5 个不同动作任务**，声明“identical RL algorithms and training settings across simulators”。该声明不能推广成抽样 realization 或 reset 时机相等。[对照][parallel-sequential] |
| 成功率定义 | 第 5 页 V-A.3：任意时刻 mean body position error 超过 **0.5 m** 判为模仿失败。这个评估指标不等于论文已经说明训练环境何时 done/reset。[指标][metrics] |

### 3. 控制与物理对齐

**论文表述：** 第 3 页 III-B 的初始化物理映射覆盖摩擦与接触设置、执行器模型和控制器增益、
重力与积分步长、刚体属性及机器人描述。接触求解器或积分器导致的残余差异被作者称为
“fixed engine meta-parameters”。这些是允许保留的引擎动力学差异，不是逐环境随机参数相等的证据。
[物理对齐][harmonization]

**论文表述：** 同节的 Numerical Normalization 举例说明角度转为弧度、
将归一化 `[-1, 1]` 动作映射到各引擎关节限位或位置目标范围，并进行符合执行器/控制器限制的裁剪。
措辞包含 “such as”，因此这是接口处理示例，不能直接当作所有实验已核实的动作配置。
[数值接口][normalization]

**未说明：** 具体控制频率、物理步长数值、decimation、PD 的 `Kp/Kd` 数值、
动作缩放/延迟、观测维数与历史长度、完整奖励及训练终止条件都没有列出。
共享这些接口语义也不意味着不同引擎在相同数值参数下产生完全相同的动力学。[III-B][router]

## 参数 DR 的实验范围与表格

### 1. 不要混淆 sim-to-sim 与 sim-to-real

- **论文表述：** 第 5 页 V-A.1 在主仿真实验说明中写道，策略零样本迁移到未见 MuJoCo，
  “without applying any parameter-level domain randomization”。应保留这一实验范围；
  该段未给出训练/评估逐阶段 DR 开关表，不能据此恢复所有代码路径的默认状态。[设置][sim-experiments]
- **论文表述：** 第 7 页 V-F 则明确描述实机部署策略来自多个引擎组合，
  “augmented with parameter-based DR”。因此不能说 PolySim 从不使用参数 DR，
  也不能把实机协议的加 DR 描述推广成所有主仿真实验均加了 DR。[部署][deployment]
- **论文表述：** 第 7 页 Table III 单独比较 Kobe 动作的参数 DR 基线，caption 明确
  “DR denotes parameter-based Domain Randomization in ASAP”。这只是 DR 方法来源说明，
  没有复印 ASAP 的参数范围，也没有说明跨引擎是否共享抽样值。[Table III][dr-results]

### 2. 实际存在的“DR 表”是结果表

附件共三张表：第 6 页 Table I 是跨引擎跟踪结果，Table II 是迭代耗时；
第 7 页 Table III 是 Kobe 动作的 DR 对照结果。**没有质量、摩擦、增益、延迟等逐字段的 DR 范围表。**
[Table I][main-results] [Table II][timings] [Table III][dr-results]

以下按 Table III 转录；`Eg-mpjpe` 的单位按 V-A.3 为 mm，`-` 保留原表缺项。
原表只有前两行带 DR 标记；此处不替未标记的组合行补上 DR 状态。[Table III][dr-results] [指标][metrics]

| Training Env. | Genesis Succ | Genesis Eg-mpjpe | MuJoCo Succ | MuJoCo Eg-mpjpe |
| --- | --- | --- | --- | --- |
| IsaacGym DR | 1.000 | 163.135 | 0.100 | 295.877 |
| IsaacSim DR | 1.000 | 130.936 | 0.100 | 272.610 |
| IsaacSim+IsaacGym | 1.000 | 103.941 | 0.100 | 178.190 |
| IsaacSim+IsaacGym+Genesis | - | - | 1.000 | 199.166 |

**证据边界：** Table III 没有比较“独立抽样”与“跨引擎配对抽样”，
不能用它的性能差异支持任一抽样协议。[Table III][dr-results]

## 由论文提出的实现核对问题

以下是论文留下的问题，不是已观察到的代码行为：

1. 每个训练入口实际启用了哪些 DR 项；各引擎的有效范围、分布、单位、基准值和支持项是否一致？
2. DR 参数由哪个进程抽样，是否有集中生成并广播到各引擎的参数向量或跨引擎环境映射？
3. 随机数生成器、种子、抽样顺序和消耗次数如何管理；是否真的建立共同随机数协议？
4. 各 DR 项在初始化、单环境 reset、时间间隔、每步或 PPO 迭代边界中的哪个时机更新？
5. done/reset 是逐环境独立处理还是跨引擎配对处理；一个 rollout 是否可能覆盖多次 DR 事件？
6. PPO 的 rollout、更新及服务器等待点是什么；同一更新是否混合多个参数 realization？

即使发现所有引擎配置使用同一个 seed，也不能仅凭 seed 值推导跨引擎实际抽样张量相等；
还需要核实生成器、调用顺序及广播/配对路径。这是核对方法，不是对当前实现的结论。

## 复核命令与限制

实际使用的主要命令如下；PDF 文本仅输出到 stdout，没有生成额外提取文件：

```bash
pdfinfo /home/wsm/.codex/attachments/762df366-534d-4381-9aee-4ffe24b63107/2510.01708v3.pdf
pdftotext -layout /home/wsm/.codex/attachments/762df366-534d-4381-9aee-4ffe24b63107/2510.01708v3.pdf -
pdftotext -layout -f 1 -l 3 /home/wsm/.codex/attachments/762df366-534d-4381-9aee-4ffe24b63107/2510.01708v3.pdf -
pdftotext -layout -f 4 -l 5 /home/wsm/.codex/attachments/762df366-534d-4381-9aee-4ffe24b63107/2510.01708v3.pdf -
pdftotext -layout -f 6 -l 8 /home/wsm/.codex/attachments/762df366-534d-4381-9aee-4ffe24b63107/2510.01708v3.pdf -
curl -fsSL --connect-timeout 10 --max-time 40 https://arxiv.org/html/2510.01708v3
date -Iseconds
```

- PDF 读取及 HTML 请求均成功；全文检索覆盖 DR/randomization、PPO/proximal、
  reset、synchronization、seed、algorithm、频率/控制及 appendix 等词，并结合逐页阅读核对。
- 已检查根目录与文档路径的适用指令，并保留工作区原有修改。
- 按本次授权边界，不运行 tests、训练、GPU 工作或依赖安装；未创建 PR。
- 本节仅记录论文核对方法；实现结论使用下节独立的固定提交代码证据。

## 固定提交源码核对

### 1. README 示例实际上关闭参数 DR

README 的并行 motion-tracking 示例使用 IsaacGym、IsaacSim、Genesis，各 2048
环境，同时传入 `+domain_rand=NO_domain_rand`。[示例][code-readme]
该 YAML 虽继承 `domain_rand_base`，但将 push、friction、COM、link mass、
PD gain、torque RFI、RFI limit 和 control delay 的随机化开关都设为 false。
不能只读取继承文件的范围就说示例已经启用了那些 DR。[关闭配置][code-no-dr]

`NO_domain_rand` 也不表示整个实验完全没有随机性：动作探索、观测噪声和参考运动
初始状态采样属于其他配置或代码路径。这里仅说明所列参数级 DR 的关闭状态。

公共 `domain_rand_base` 声明 link mass 倍率 `[0.8,1.2]`、KP/KD 倍率各
`[0.75,1.25]`、friction 范围 `[0.5,1.25]`、delay 整数范围 `[0,2]` 等。
这是文件声明，不是论文参数表，也不是已验证可在所有引擎同义生效的运行配方。
[范围配置][code-dr-base]

### 2. 参数由局部环境生命周期采样

调用链是 `LeggedRobotBase._post_physics_step()` 根据本地 `reset_buf` 取得
`env_ids`，调用 `reset_envs_idx(env_ids)`，再经 `_reset_tasks_callback` 调用
`_episodic_domain_randomization(env_ids)`。MotionTracking 继承这条路径。
[逐步 reset][code-post-step] [reset callback][code-reset] [motion 继承][code-motion-reset]

启用 PD DR 时的关键代码是：

```python
self._kp_scale[env_ids] = torch_rand_float(
    self.config.domain_rand.kp_range[0],
    self.config.domain_rand.kp_range[1],
    (len(env_ids), self.num_dofs), device=self.device)
```

KD 另抽一份同形状数组。抽样单位是**本次需要 reset 的每个环境、每个关节**，
不是每个仿真器一个标量，更不是一次 PPO 更新共享一个标量。这里赋值覆盖倍率，
PD 计算使用 `_kp_scale * p_gains` 与 `_kd_scale * d_gains`。
[reset 抽样][code-reset-dr] [PD 计算][code-torque]

| DR 项 | 所读路径的采样/应用时机 | 更新内含义 |
| --- | --- | --- |
| KP/KD 倍率 | 本地环境 reset，形状为 `reset_envs x joints`。 | 未 reset 的行保持；reset 的行重抽，可在 rollout 中途变化。 |
| RFI 幅度倍率 | 同一个 episodic reset 函数中重抽。 | 不按 PPO 更新边界统一变更。 |
| control delay | 本地 reset 时抽每环境整数 delay，并清空对应 action queue；初始化也创建随机索引。 | 各环境拥有自己的 delay 索引。 |
| RFI 瞬时扰动 | `_compute_torques` 每次调用执行 `torch.rand_like(torques)`；物理 decimation 循环内调用。 | 不要求一个 episode 或 rollout 内保持固定。 |
| push | 每环境自己的 interval counter 到期时重抽速度及下一间隔。 | 不是全来源同一时刻施加同一推力。 |
| IsaacGym mass/COM/shape friction | 创建环境的 rigid-body/shape callbacks。 | 这些代码不是每轮 PPO 或每个 reset 重抽。 |
| IsaacSim mass/COM/joint friction | EventManager 的 `startup` 事件。 | 这些代码不是每轮 PPO 或每个 reset 重抽。 |

表中每项都以对应开关开启为前提，不是 README 无参数 DR 示例的运行状态。
[episodic][code-reset-dr] [物理子步][code-physics] [RFI][code-torque]
[push][code-push] [Gym 创建][code-gym-create] [Sim startup][code-sim-events]

### 3. 同步的是 step 完成，不是参数 realization

`EnvClient.step` 按固定来源 slice 切动作，通过线程池并发 RPC，再等待全部结果；
传入每台服务器的是 `{"actions": act}`，没有 DR 参数表、配对环境编号或新抽样指令。
`reset_all()` 是另外的调用，不在 PPO 的每次迭代开头执行。[RPC step][code-client]

`PPO.learn()` 在循环外 `reset_all()`，循环内部是 `_rollout_step(obs_dict)` 再
`_training_step()`。rollout 内逐步收集数据并允许环境 autoreset。默认 H=24，
更新配置为 5 epochs、4 minibatches；训练后清空 storage 写入位置。
[训练循环][code-learn] [rollout][code-rollout] [更新][code-update] [PPO 配置][code-ppo-config]

因此同一轮数据可以同时包含：同一引擎中不同参数的环境、不同引擎的参数组合、
同一环境 reset 前后的不同参数。在一轮数据上做多个 PPO epoch，不会为旧样本
重新抽 DR 或重新模拟；它们的奖励已经由采样时的动力学产生。

“共用 seed”也不足以产生配对保证：还需要相同 RNG、调用顺序、形状、调用次数
及环境 reset 事件。已读服务器中的显式 seeding 调用还是注释状态；这不能支持
跨后端参数相等的说法，也不能据此声明所有后端都没有内部 seed。
[服务器][code-server]

### 4. 配置同名不一定是相同物理量

公开实现里有一个可直接核对的差异：

- IsaacGym 的 `randomize_friction` 抽取桶化系数并写入 rigid shape 的
  `props[s].friction`，即接触相关的形状摩擦。[Gym friction][code-gym-friction]
- IsaacSim 的同名开关创建 `randomize_joint_parameters` startup 事件，
  指定 `friction_distribution_params` 和 `operation="scale"`，改的是关节摩擦。
  这不是与 Gym 相同的接触摩擦随机化。[Sim friction][code-sim-events]

所以不能说这份公开代码已严格保证所有引擎的有效物理 DR 集合及分布一致。
Genesis 的已读 adapter 中也未找到对应的 mass/COM/friction 随机化配置消费，
但这不等于 Genesis 没有参数 DR：共享环境层仍有 PD/RFI/delay 等路径。
此处是源码核对，不是运行效果测试或对所有分支的穷尽式否定。[Genesis][code-genesis]

### 5. 对 UniLab 实验设计的含义

**建议，不是 PolySim 已有机制：** 联合鲁棒训练优先统一物理语义、支持集合、
范围/分布、标称基准和 reset 生命周期，再允许每来源每环境自行抽样。
不同来源样本数相同，不等于每项参数数值、episode 长度或梯度贡献相同。

若目的是隔离引擎差异或排查 IsaacSim 异常，更适合另设配对诊断：集中产生一个
含初始状态、命令、参考轨迹及物理参数的 case，映射并读回到各引擎，从共同起点
比较。相同 seed 不能替代 case；相同倍率也不保证不同标称资产下的绝对参数相同。
开放环诊断可以固定动作序列；闭环比较可固定策略，但因观测不同而产生的动作
差异是闭环响应的一部分。不要为了“每个 PPO 更新一致”在 episode 中途强改参数。

本轮没有改变 UniLab 的四源配额、DR 配置或训练生命周期，也没有重启当前训练。

### 6. 源码复核方法与限制

- 官方 Git 远端 HEAD 查询成功；Git clone 连接超时，未使用未完成 checkout 作为证据。
- GitHub tree API 和固定提交 `raw.githubusercontent.com` 文件读取成功；个别请求
  超时后重读，未把网络失败作为代码不存在的证据。
- 只读缓存位于 `.tmp/polysim-code/`，使用 `rg`、`sed`、`wc` 核对调用关系；
  没有安装或运行 PolySim，未导入其 Python 文件，也没有 GPU 活动。
- 本次新增笔记做本地引用/锚点与 whitespace 检查；无运行时代码修改，不运行训练测试。

## 一手来源

[paper]: https://arxiv.org/html/2510.01708v3
[pdf]: https://arxiv.org/pdf/2510.01708v3
[attachment]: /home/wsm/.codex/attachments/762df366-534d-4381-9aee-4ffe24b63107/2510.01708v3.pdf
[method]: https://arxiv.org/html/2510.01708v3#S3
[isolation]: https://arxiv.org/html/2510.01708v3#S3.SS1
[router]: https://arxiv.org/html/2510.01708v3#S3.SS2
[harmonization]: https://arxiv.org/html/2510.01708v3#S3.SS2.p2
[normalization]: https://arxiv.org/html/2510.01708v3#S3.SS2.p4
[overview]: https://arxiv.org/html/2510.01708v3#S3.F3
[theory]: https://arxiv.org/html/2510.01708v3#S4
[related-control]: https://arxiv.org/html/2510.01708v3#S2.SS1
[ppo-reference]: https://arxiv.org/html/2510.01708v3#bib.bib6
[sim-experiments]: https://arxiv.org/html/2510.01708v3#S5.SS1.SSS1
[real-experiments]: https://arxiv.org/html/2510.01708v3#S5.SS1.SSS2
[metrics]: https://arxiv.org/html/2510.01708v3#S5.SS1.SSS3
[baselines]: https://arxiv.org/html/2510.01708v3#S5.SS2
[parallel-sequential]: https://arxiv.org/html/2510.01708v3#S5.SS4
[deployment]: https://arxiv.org/html/2510.01708v3#S5.SS6
[main-results]: https://arxiv.org/html/2510.01708v3#S4.T1
[timings]: https://arxiv.org/html/2510.01708v3#S5.T2
[dr-results]: https://arxiv.org/html/2510.01708v3#S5.T3
[code-readme]: https://github.com/EmboMaster/PolySim/blob/0fdb349785479b5c0c30d3c676fae67025b8d34a/README.md#L229-L258
[code-no-dr]: https://github.com/EmboMaster/PolySim/blob/0fdb349785479b5c0c30d3c676fae67025b8d34a/humanoidverse/config/domain_rand/NO_domain_rand.yaml
[code-dr-base]: https://github.com/EmboMaster/PolySim/blob/0fdb349785479b5c0c30d3c676fae67025b8d34a/humanoidverse/config/domain_rand/domain_rand_base.yaml
[code-post-step]: https://github.com/EmboMaster/PolySim/blob/0fdb349785479b5c0c30d3c676fae67025b8d34a/humanoidverse/envs/legged_base_task/legged_robot_base.py#L248-L262
[code-reset]: https://github.com/EmboMaster/PolySim/blob/0fdb349785479b5c0c30d3c676fae67025b8d34a/humanoidverse/envs/legged_base_task/legged_robot_base.py#L384-L422
[code-motion-reset]: https://github.com/EmboMaster/PolySim/blob/0fdb349785479b5c0c30d3c676fae67025b8d34a/humanoidverse/envs/motion_tracking/motion_tracking.py#L160-L165
[code-reset-dr]: https://github.com/EmboMaster/PolySim/blob/0fdb349785479b5c0c30d3c676fae67025b8d34a/humanoidverse/envs/legged_base_task/legged_robot_base.py#L816-L832
[code-physics]: https://github.com/EmboMaster/PolySim/blob/0fdb349785479b5c0c30d3c676fae67025b8d34a/humanoidverse/envs/legged_base_task/legged_robot_base.py#L233-L244
[code-torque]: https://github.com/EmboMaster/PolySim/blob/0fdb349785479b5c0c30d3c676fae67025b8d34a/humanoidverse/envs/legged_base_task/legged_robot_base.py#L585-L609
[code-push]: https://github.com/EmboMaster/PolySim/blob/0fdb349785479b5c0c30d3c676fae67025b8d34a/humanoidverse/envs/legged_base_task/legged_robot_base.py#L316-L322
[code-gym-create]: https://github.com/EmboMaster/PolySim/blob/0fdb349785479b5c0c30d3c676fae67025b8d34a/humanoidverse/simulator/isaacgym/isaacgym.py#L227-L250
[code-gym-friction]: https://github.com/EmboMaster/PolySim/blob/0fdb349785479b5c0c30d3c676fae67025b8d34a/humanoidverse/simulator/isaacgym/isaacgym.py#L267-L290
[code-sim-events]: https://github.com/EmboMaster/PolySim/blob/0fdb349785479b5c0c30d3c676fae67025b8d34a/humanoidverse/simulator/isaacsim/isaacsim.py#L118-L167
[code-client]: https://github.com/EmboMaster/PolySim/blob/0fdb349785479b5c0c30d3c676fae67025b8d34a/humanoidverse/env_client.py#L281-L315
[code-learn]: https://github.com/EmboMaster/PolySim/blob/0fdb349785479b5c0c30d3c676fae67025b8d34a/humanoidverse/agents/ppo/ppo.py#L168-L194
[code-rollout]: https://github.com/EmboMaster/PolySim/blob/0fdb349785479b5c0c30d3c676fae67025b8d34a/humanoidverse/agents/ppo/ppo.py#L236-L301
[code-update]: https://github.com/EmboMaster/PolySim/blob/0fdb349785479b5c0c30d3c676fae67025b8d34a/humanoidverse/agents/ppo/ppo.py#L362-L377
[code-ppo-config]: https://github.com/EmboMaster/PolySim/blob/0fdb349785479b5c0c30d3c676fae67025b8d34a/humanoidverse/config/algo/ppo.yaml#L3-L29
[code-server]: https://github.com/EmboMaster/PolySim/blob/0fdb349785479b5c0c30d3c676fae67025b8d34a/humanoidverse/hydra_server.py#L162-L169
[code-genesis]: https://github.com/EmboMaster/PolySim/blob/0fdb349785479b5c0c30d3c676fae67025b8d34a/humanoidverse/simulator/genesis/genesis.py
