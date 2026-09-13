# UniLab 自适应多仿真器 DR：证据与设计提案

状态：研究提案，未实现、未训练验证；不是已接受的 ADR 或后端支持承诺。
UniLab 检查基线：`db1a6e5b3dbe4dd096b16a517837f2d46dba8164`。
本轮只新增本文，不修改运行时、任务默认配置、依赖、分支或工作流政策。

## 建议摘要

可以把仿真器视为参数 DR 之外的一个离散域，在线调整域权重，替代对所有静态配方的穷举。
推荐先实现“**常驻多来源采集 + 单 learner + 有覆盖约束的自适应组权重**”，
验证鲁棒性收益后，再优化实际采集配额；不要首先实现热切换引擎或动态扩缩容。

第一版使用固定参数 DR、统一任务语义及平衡评估，以正则化最差域风险驱动权重。
它追求已知域的鲁棒折中，不估计真实世界属于哪个仿真器，也不保证 sim-to-real 最优。
“样本来自哪里”“优化偏重哪里”“为每个引擎分配多少资源”是三个不同问题。

## PolySim 一手资料核对

### 1. 来源、版本与证据边界

- 访问日期：本地时钟 **2026-09-11，UTC+08:00**；对应 UTC 日期为 **2026-09-10**。
- 论文：*PolySim: Bridging the Sim-to-Real Gap for Humanoid Control via Multi-Simulator Dynamics Randomization*。
- 附件首页标注 arXiv:2510.01708v3，**2025-10-14**；首页明确给出 EmboMaster/PolySim。
- [arXiv v3 官方 HTML][paper] 也链接到该仓库；不是按仓库名称猜测官方性。
- 附件全文使用 `pdftotext -layout` 阅读，并与官方 HTML、源码分别核对。
- 官方仓库：[EmboMaster/PolySim，固定版本树][tree]。
- `git ls-remote --symref` 返回默认分支 `main`，HEAD 为 **`0fdb349785479b5c0c30d3c676fae67025b8d34a`**。
- [该提交][commit] 的补丁头日期为 **2025-10-27 15:10:04 +0800**，提交说明为 `Update LICENSE`。
- 以下源码链接全部固定到此提交；这不是对未公开代码、其他分支或未来版本的判断。

证据标签：**论文表述**指附件 v3；**README 描述**指维护者文档；
**源码已证实**指实际读取的调用链；**推断**指从该代码结构推出的结论；
**未证实**不等于不存在，也不等于已通过运行验证。

### 2. 先给结论

1. **源码已证实：主要配比入口是静态的各引擎环境数量 `num_envs_list`。**
   启动时给各服务器传入对应 `num_envs`，客户端固定分片并令总数为列表之和。[配置][overall] [启动][launch] [分片][distribution]
2. **源码已证实：跨引擎聚合的是 transition 样本，不是多个 learner 的梯度。**
   EnvClient 拼接观测、奖励和 done；一个 PPO 实例对合并后的向量环境采样、更新。[拼接][concat] [learner][learner]
3. **源码已证实：PPO 没有独立的引擎目标 loss 权重。**
   actor、value、entropy 项使用 batch 均值；引擎出现得更多，就在经验 batch 中占更多样本。[损失][loss]
4. **限定性否定：在已核对的 README 对应训练路径中，未发现自适应引擎配比机制。**
   没有看到按引擎回报调整环境数、采样概率或 loss 权重的控制器；`schedule: adaptive` 调的是学习率。[调度配置][ppo-config] [学习率实现][adaptive-lr]

### 3. 论文、README、实现分别说了什么

| 层次 | 可以记录的事实 | 不能升级成的结论 |
| --- | --- | --- |
| 论文首页 | 宣称一次训练可并行使用异构引擎，并报告迁移收益。[论文][paper] | 不等于任意混合比例都有收益；理论限制见后文。 |
| README | 示例选择 IsaacGym、IsaacSim、Genesis，并用 `[2048,2048,2048]` 指定各自环境数。[README][readme-training] | “可选任意环境数”不是训练中自动调整比例。 |
| README | 描述独立终端/RPC；用 MuJoCo 做可视化和评估。[README][readme-training] [评估][readme-eval] | MuJoCo 评估不等于该并行训练入口支持 MuJoCo。 |
| 源码 | 主脚本自动启动后台进程；支持映射只列三种训练引擎。[启动][launch] | 不是多个独立训练器训练后平均权重。 |
| 文档一致性 | README 摘要仍说接收后放代码，但 TODO 已勾选 backbone/parallel pipeline 发布。[状态][readme-status] | 不应只依据旧摘要断言仓库尚无实现。 |

### 4. 静态来源配置与并行边界

- [总体配置][overall] 默认 `simulator_list=['isaacsim','isaacgym']`、`num_envs_list=[4096,4096]`。
  README 三引擎等分示例是另一组显式配置，不是算法推导出的比例。
- [启动代码][launch] 令 RPC world size 为引擎数加一；每个引擎一个服务器进程，额外一个客户端/learner。
  三个引擎分别在 `hvgym`、`hvlab`、`hvgen` Conda 环境中启动，进程通过 `subprocess.Popen` 创建。
- [客户端创建][client-create] 根据列表计算总环境数、服务器名称及 GPU 映射，并初始化 TensorPipe RPC。
  地址和接口设置指向本机；本轮未证明跨机器部署能力或实际 GPU 通信吞吐。
- [环境服务器][server] 实例化环境并注册 RPC 服务；[EnvService.step][server-step] 调用环境 step。
  learner 侧只实例化一次算法并调用 `setup()`、`learn()`。[learner][learner]
- [EnvClient 初始化][client-init] 只在初始化阶段计算服务器分片、预分配 tensor 和日志权重。
  分片采用每个服务器固定的连续环境区间，而非每个 episode 随机选择引擎。

### 5. 三种“比例”必须分开

| 名称 | 该实现中的精确定义 | 证据/限定 |
| --- | --- | --- |
| 环境分配比例 | `n_i / sum(n)`，其中 `n_i` 来自启动配置。 | 总量与服务器分片见 [客户端创建][client-create]、[分配][distribution]。 |
| 完整 rollout 样本份额 | 每引擎 `H*n_i` 条 transition，总计 `H*sum(n)`。 | 每个 rollout 对合并环境统一执行 `H` 次 step。[采样][rollout] |
| 实际 minibatch 份额 | 合并 rollout 展平后随机打散；单个 minibatch 不保证保持环境比例。 | 没有按引擎分层采样。[存储采样器][minibatch] |
| 目标 loss 权重 | 没有单独的引擎权重参数；实现对样本损失求均值。 | 相同计数不保证相同梯度贡献，优势大小等仍会影响更新。[损失][loss] |
| 日志聚合权重 | `info_weights = num_envs_list / total_envs`。 | 用于 `episode` / `to_log` 合并，不是训练 loss 权重。[初始化][client-init] [日志合并][info-merge] |

- **源码已证实：** PPO 配置的 rollout 长度是 `num_steps_per_env=24`。[PPO 配置][ppo-config]
  正常完整采集时，各引擎每个环境贡献同样步数；episode 长短不是这里的混合比例。
- **源码已证实：** 观测按 key 拼接，reward/done 沿环境维拼接，顺序遵循 `server_names`。
  `step` 的并发完成顺序没有被用来决定谁多贡献样本。[拼接][concat] [step][client-step]
- **源码细节：** 采样器用 `batch_size // num_mini_batches` 计算 batch 大小，
  随机排列的范围是 `num_mini_batches * mini_batch_size`，而不是始终整个 rollout。
  不能整除时，展平后的尾部样本不进入更新；因此“所有有效训练样本精确按环境比例”有整除前提。[采样器][minibatch]
- **推断：** 能整除且无异常时，每个训练 epoch 遍历完整 rollout，整体样本份额等于环境份额；
  但这只是样本计数意义的隐式权重，不是独立可控的目标权重，也不等于等量梯度贡献。

### 6. 有没有自适应比例；是否使用 replay

- 核对范围：README 示例 → 总体配置 → launcher → EnvClient → PPO → RolloutStorage。
  这些文件中，环境列表/分片在初始化时确定，没有发现训练中重建或重新分配引擎配额的调用。
- PPO 的 `adaptive` 分支根据 batch KL 改 actor/critic 学习率，不修改来源环境数量。
  把该配置称为“自适应仿真器配比”会误读实现。[学习率实现][adaptive-lr]
- **源码已证实：** 每次迭代先 `_rollout_step`，再 `_training_step`；
  更新后 `storage.clear()` 将写入位置归零，随后覆盖本轮存储。[训练循环][train-loop] [更新][update] [存储][storage]
- **结论：** 这是同步收集、集中更新的 PPO 路径；没有在这条路径发现跨迭代 replay buffer 或引擎 replay 配额。
  同一轮 rollout 做多个 epoch 的 PPO 更新，不应表述成 off-policy replay。[PPO 配置][ppo-config] [采样器][minibatch]
- **未证实：** 仓库其他算法/实验分支是否存在另一套配比机制；本轮没有做全仓穷尽式否定。

### 7. engine id、critic 与 normalization

- **源码已证实：** 客户端知道服务器名称和分片区间，但聚合代码没有附加 engine-id 观测；
  PPO 存储键也没有注册独立的 `engine_id`。[拼接][concat] [PPO 存储键][ppo-storage]
- **源码已证实：** README 使用的观测配置区分 actor/critic；critic 额外包含 base linear velocity、
  刚体参考误差/参考位置等信息，两者均有历史输入，字段列表没有 simulator id。[观测配置][obs]
  因而可称该配置采用不对称/privileged critic，不能称“已实现 engine-conditioned critic”。
- **源码已证实：** 该 YAML 列出固定 `obs_scales`、`noise_scales`；
  PPO 对整份合并 rollout 的 advantages 求全局均值/标准差，而非逐引擎归一化。[观测配置][obs] [GAE/优势][advantage]
- **未证实：** 所有环境/网络层是否还有运行均值、观测裁剪或奖励归一化；本轮未完整追踪这些层，
  因而不作“全仓不存在 running normalization”的断言，也未运行验证跨后端数值一致性。

### 8. 同步点与 straggler

- **源码已证实：** EnvClient 先向各服务器并发提交 step，随后等待全部结果，再返回合并 batch。
  `_rpc_call_with_retry` 虽调用 `rpc_async`，却立即 `future.wait()`；外层也等待每个任务完成。[step][client-step] [RPC 等待][rpc-wait]
- **推断：** 这是每个环境步都有屏障的同步采样；快引擎不会因先完成而占更高样本份额，慢引擎会拖慢整个 step。
  并发 RPC 不等于异步 actor–learner，也不能据此声称消除了 straggler。
- **源码已证实：** RPC 有超时和重试；step 最终失败会抛异常，未看到跳过慢源后继续训练的分支。[RPC 等待][rpc-wait] [step][client-step]
- **未证实：** 各引擎实际耗时、跨 GPU 开销、吞吐瓶颈占比；本轮未运行仿真、profiling 或 benchmark。

### 9. 复核方法与交付限制

- 已实际读取官方固定提交的 README、启动/客户端/服务器、PPO、存储及所引配置；源码只下载到内存，没有 clone/checkout。
- 命令包括 `date -Iseconds`、`git ls-remote --symref`、`curl -fsSL`、`nl -ba`、`rg`、首页 `pdftotext -f 1 -l 2`。
- 源码核对过程中 GitHub API 遇到 403，改用官方 Git 远端、GitHub 页面及固定提交 raw 内容交叉核实。
  官方 arXiv 页面和关键 raw 源码另经 web open 核对。
- 个别网络读取超时后改读同一提交成功；未把连接失败当成源码不存在。
- 未安装依赖，未执行 PolySim/UniLab 训练；本地 PATH 无 `uv`，没有直接调用 Python。
- 后续方案应以“**静态环境配额 → 同步样本拼接 → PPO 样本均值损失**”作为已证实基线；
  不把自适应比例、独立目标权重或逐引擎 normalization 描述成 PolySim 已提供的能力。

## UniLab 的现状与前置条件

以下是本地代码事实，不代表已验证外部依赖内部的所有能力。路径链接指向本仓库；行号对应本文基线。

| 已核实位置 | 现状 | 对设计的约束 |
| --- | --- | --- |
| [依赖声明](../../pyproject.toml)，第 43–50 行；[导入边界测试](../../tests/test_library_import_boundary.py)，第 53 行 | 算法、collector、IPC、训练日志已属于独立 `uni_rl`；物理属于 `unisim`。 | 不能在 UniLab 重建 runner/IPC，也不能让 `uni_rl` import UniLab。 |
| [EnvFactory](../../src/unilab/base/env_factory.py)，第 78 行 | 一个可 pickle 的 factory 绑定一个 task/backend，子进程重新 bootstrap registry。 | 复用为每个来源的 factory，而不是在 env.reset 中换引擎。 |
| [PPO 入口](../../src/unilab/scripts/train_rsl_rl.py)，第 617、656 行；[APPO 入口](../../src/unilab/scripts/train_appo.py)，第 79、379 行 | 当前入口构造单 env 或传入单 factory。 | 本地未找到单 learner 多后端的现成装配入口。 |
| [off-policy 入口](../../src/unilab/scripts/train_offpolicy.py)，第 131、233 行 | 向独立 runtime 注入单 factory、override 和设备绑定。 | 没有从本地调用链证实多来源 quota 接口；不能据此断言外部库绝不支持。 |
| [Manager reset](../../src/unilab/envs/manager_based_rl_env.py)，第 557 行；[DR events](../../src/unilab/envs/mdp/events.py)，第 586 行 | 当前 manager 任务经 reset event 和 reset transaction 改物理参数。 | 参数 DR 保留在现有 task/event owner；配比控制不放进 DR event。 |
| [DR manager](../../src/unilab/dr/manager.py)，第 35 行；[测试](../../tests/dr/test_manager.py)，第 197 行 | 另一路 reset payload 会过滤不支持项并告警；interval 不支持项报错。 | 混合实验必须显式记录有效 DR，不能把“配置相同”当成“生效相同”。 |
| [MotionSampler](../../src/unilab/tasks/motion_tracking/common/motion_loader.py)，第 502、547 行 | 已有失败统计驱动的动作片段采样；含均匀质量与平滑。 | 可借鉴思想，但它属于 motion task，不是引擎权重控制器。 |
| [Sim2Sim](../../src/unilab/utils/sim2sim.py)，第 36、81 行 | 严格项含观测、动作、网络维度、normalization、motion sampling mode。 | 复用快照及校验语义，不能绕过 DENYLIST 来拼接不兼容来源。 |

**不能直接拼现有 G1 默认配置。**
[MuJoCo G1WalkFlat](../../src/unilab/conf/ppo/task/g1_walk_flat/mujoco.yaml) 第 14–17 行使用 `actor` 且关闭 empirical normalization；
[Motrix owner](../../src/unilab/conf/ppo/task/g1_walk_flat/motrix.yaml) 第 2–4 行明确标为 intentionally non-transferable，
第 15–40 行还改变 observation groups、action scale、command 分布和 PD DR。
[G1MotionTracking Motrix owner](../../src/unilab/conf/ppo/task/g1_motion_tracking/motrix.yaml) 第 11–32 行覆盖 reward。
这些差异会让控制器同时适应任务定义和物理引擎，无法归因于 simulator DR。

混合训练的 cold-path preflight 应比已有 train-to-play 检查更严格：

- 统一任务、机器人/动作顺序、坐标系、单位、控制周期、动作映射、观测历史及 actor/critic spec。
- 统一 reward 定义与尺度、termination/truncation、命令/动作片段分布；不能只检查 tensor shape。
- 使用公共参数 DR 语义及各 backend 支持能力的显式交集；交集外的受控差异单列为实验因素。
  未授权或未实现的必需项在构造前失败，不在新混合路径中静默降级。
- XML/asset 映射、能力绑定和 nominal 参数检查仅在初始化/缓存进行；env 不访问 backend 私有实现。
- 不给 actor 增加 simulator id。来源 ID 仅作为采集/学习统计 metadata；若以后扩展 privileged critic，
  要一起更新 policy-I/O、checkpoint dimension、Sim2Sim contract 及测试。
- normalization 必须有统一、可复现、部署时一致的策略；第一版优先用共享固定尺度。
  不能在部署时要求识别引擎后再选独立 normalizer。

## 自适应机制：先改变优化权重

### 1. 分布与目标

令 `domain_id` 表示一个经过 preflight 的仿真域；引擎不是质量、摩擦等连续参数的替代品。
参数抽样仍由该任务的 DR owner 完成，联合域可写为：

\[
P_t(\mathrm{domain}=i,\xi)=p_{t,i}D_i(\xi),\qquad
\sum_i p_{t,i}=1.
\]

第一版实际并不按这个式子热切换单个 env；各引擎池常驻，由组损失权重实现目标域混合。
每条物理轨迹始终留在同一引擎，reset 仍遵循该 env 的已有生命周期。
固定引擎的 episode/trajectory mixture，一般不等于每一步重抽引擎的 transition-kernel mixture。
本文不要求在不同引擎之间移植接触求解器或隐含状态。

令 `p` 为希望优化的组权重，`q_i=N_i/B` 为本轮实际接纳的有效 transition 份额，
另记资源分配为 `a_i`。三者不应混用；episode 数量也不是 `N_i`。
对相同 policy 版本、各域独立形成的 PPO rollout，定义：

\[
L(\theta;p)=\sum_i p_i\widehat L_i^{\mathrm{PPO}}(\theta),\qquad
\widehat L_i^{\mathrm{PPO}}=\frac{1}{N_i}\sum_{k\in i}L_k^{\mathrm{PPO}}.
\]

每样本实现等价于在完整 batch 均值内乘 `p_i/q_i`，不是乘 `p_i` 后再按样本数求均值。
分组求均值和逐样本校正二选一，不重复加权。这里校正的是来源份额，不能替代 PPO 自身的 policy ratio。
actor/value/entropy 各项采用什么组权重须一致定义并单独记录；第一版使用同一 `p`，保留原有 loss 系数。

第一版固定 `q` 和 worker 数量。为简化归因，先接同步 PPO，而不是同时引入 APPO policy lag 或 off-policy replay。
“固定 `q`”仍需真正接通多来源采集，不是只在当前单 backend 的 CLI 后面新增一个权重参数。

### 2. 用可比较的风险反馈，而不是原始训练回报

每隔 `M` 个更新窗口，在全部训练域上评估同一冻结 policy：

- 评估的任务/命令/动作片段和 DR **分布固定**，各域 episode 预算平衡，终止也按 episode 计入失败率。
- 在各域使用匹配的语义场景及种子表；不能假设不同引擎的内部 RNG 会产生相同扰动。
- 覆盖固定难度桶，按事先确定的桶权重汇总；不直接读取随采集份额和 motion curriculum 改变的训练失败计数。
- 保留从不反馈到控制器的独立 audit/test 场景。反馈评估可以从冻结的场景分布轮换抽样，降低对固定样本的适应。

例如 WBC 的反馈风险可用：

\[
\widehat r_i=w_f(1-\widehat{\mathrm{success}}_i)
+w_e\operatorname{clip}(\widehat{\mathrm{tracking\ error}}_i/e_{\mathrm{ref}},0,1),
\quad w_f+w_e=1.
\]

`e_ref`、权重和成功阈值由共同任务定义，训练前固定，不用各引擎自己的移动 min/max 做归一化。
若选择与 actor reward 不同的失败率/跟踪误差作为反馈，本机制是 **DRO-inspired 代理控制**，
不能声称 PPO 精确优化了该风险的 min-max 目标。

### 3. 有覆盖约束的更新

以固定正 prior `p0`（第一版均匀）为锚，先对每域风险做 EMA，得到 `bar_r`，再求：

\[
p^*=\arg\max_{p\in\mathcal P}
\left\{\sum_i p_i\bar r_i-\tau\,D_{\mathrm{KL}}(p\Vert p_0)\right\},
\qquad p_{t+1}=(1-\alpha)p_t+\alpha p^*.
\]

这里 `tau > 0` 控制偏向困难域的程度，`0 < alpha <= 1` 控制外环响应速度。
这是参考 [group DRO][group-dro] 的组风险思想提出的控制器，不是 PolySim 已实现的算法。
约束集合包含 `sum(p)=1`、每域覆盖下限和上限；可行性要求 `K*p_min <= 1 <= K*p_max`。
单域直接退化为权重 1。

没有上下限时，解为 `p*_i ∝ p0_i * exp(bar_r_i/tau)`；
有上下限时，对这个解做有界单纯形上的 **KL 投影**，不能简单 clip 后 normalize 破坏下限。
实现可在 log-space 稳定计算，并求使 `sum(clip(c*u_i, lower_i, upper_i))=1` 的标量 `c`。
前一权重与候选都可行时，平滑组合仍可行。

直觉：表现已好的域逐渐让出份额，但不会消失；较差域得到更多训练压力，但不能独占全部更新。
相同风险时最优候选回到 prior；没有必要为每一组离散比例独立重跑完整训练。
这不消除所有调参：`M`、温度、平滑强度、风险定义仍需小规模敏感性验证。

### 4. 不能省略的稳定性条件

- **有效样本量：** 约束 `p_i/q_i`，并要求每域足够多的独立轨迹。
  忽略相关性时权重 ESS 约为 `B / sum_i(p_i^2/q_i)`；实际还受轨迹相关性影响。
  超预算时把候选向可支持的基线回缩、增加新鲜采样或暂停调整，不靠反复复用稀少 PPO 轨迹补量。
- **置信度：** 只在所有域达到预先规定的完整 episode/有效样本门槛后更新；
  风险差异小于不确定度时保留原权重。EMA 混合历史 policy，应同时记录风险的 policy 版本与时效。
- **外环节奏：** uniform warm-up；`p` 在完整 rollout 和其所有 PPO epochs 内冻结。
  每个域分别计算 GAE，保留 recurrent sequence、terminated/truncated、terminal actor/critic obs 和 bootstrap mask。
- **振荡：** 记录权重总变差、长期贴上限及跨域退化；持续交替恶化时冻结自适应或回退预定义静态基线，
  而不是持续增大对抗强度。回退是显式运行事件，要保留 checkpoint 和触发原因。
- **健康与任务失败分离：** NaN、资产/坐标错误、超时、进程崩溃不是“高价值困难样本”。
  必需域故障默认终止本轮；不能静默删除来源后仍报告原混合实验。
- **不可约冲突：** 高失败率未必可通过加权修复；同一 actor 信息下可能无法兼顾相反控制需求。
  长期饱和需要做单域能力、可观测性和任务一致性诊断，不能仅据低回报自动剔除域。
- **两级课程：** 第一版冻结参数 DR 范围，并让 motion sampling 采用共同固定分布；
  后续才分别评估 adaptive motion、参数 ADR 与引擎权重的交互，不能把三种非平稳性一次打开。

### 5. 第二阶段才控制真实采集量

第一版只是改变“用已有样本如何学习”，不承诺节省仿真开销。
第二阶段保持进程和 backend 常驻，调整下一窗口的完整 unroll 配额，记录实际 `q`，仍按 `p` 定义组目标。
若 `q` 接近 `p`，权重方差通常更小；若为吞吐主动偏离 `p`，必须保留来源校正和 ESS 门槛。

按完整序列做整数配额，保证总有效 batch budget 和各域最小覆盖，不能靠提前完成的引擎自然填满队列。
慢引擎成本进入预算分配/资源约束，而不是简单把目标风险乘以 FPS 或除以耗时。
阶段一允许明确的 rollout 屏障；动态配额也不会自动消除慢源或 GPU 争用，需要测量后再决定是否扩展协议。

若改做 SAC/TD3，需要另行核实 source-aware replay、每域样本年龄和有效覆盖；
本轮的 on-policy `p/q` 推导不是对 stale/off-policy 数据的完整校正。

## Module 落点与最小交付路线

| Module / owner | 负责什么 | 不负责什么 |
| --- | --- | --- |
| UniLab task/config 与训练 adapter | 共同任务契约、每来源 factory/owner 配置、backend 映射、preflight、物理指标到风险的映射、run snapshot。 | 不实现 PPO、collector IPC 或引擎热切换。 |
| `uni_rl` mixture-controller Module | 根据中立的 domain 统计和预算，维护风险 EMA、组权重、稳定性状态；给出下一窗口计划。 | 不认识 UniLab、资产或 backend 私有类型。 |
| `uni_rl` runner/learner/collector | 常驻来源生命周期、policy 版本、source metadata、独立 GAE、组损失、后续 quota、训练日志。 | 不解析任务 XML、猜测物理参数语义。 |
| `unisim` backend Adapter | 公开物理能力、DR payload 的真实应用、engine-native 生命周期。 | 不决定哪种训练域应该多学。 |

建议 controller 保持一个小 Interface：`advance(state, report, budget) -> (next_state, plan)`。
输入是已验证的数值统计和来源 ID，输出是不可变窗口计划；调度及采样作为依赖留给 runner。
这是提议中的 Interface，当前仓库不存在该公开入口。纯状态变换便于不用启动仿真器就验证整个控制行为。

来源 metadata 必须从 collector 到 learner 保真，不是只往 `env.info` 塞字符串。
至少需要审定 `domain_id`、`policy_version`、窗口/DR 配置版本、有效样本量和健康状态的归属与生命周期。
run snapshot 保存所有 owner 的 resolved 配置、共同 Sim2Sim contract、engine/asset 版本、prior 与控制器超参；
checkpoint 保存当前权重、EMA、窗口计数和 RNG 状态，并明确 env 状态未持久化时不能声称 bitwise 续训。

本地没有 `uni_rl`/`unisim` sibling 源码，未安装依赖，故它们内部是否已有可复用多源扩展点仍待核实。
实现前应以当前依赖的源码复核来缩小补丁，不能把未核实的缺口直接变成新公共协议。
相关约束见 [ADR-0001](../sphinx/source/adr/ADR-0001-runtime-model-and-layer-boundaries.md)、
[ADR-0005](../sphinx/source/adr/ADR-0005-unified-obs-critic-env-and-ipc-contract.md)、
[ADR-0007](../sphinx/source/adr/ADR-0007-unisim-extraction-boundary.md)。
ADR-0007 中 runner/learner 的旧归属已与当前依赖/边界测试不同，本文遵循当前代码和根目录指南。

按[贡献流程](../sphinx/source/en/4-developer_guide/5-contributing_workflow.md)，
进入实现前须确定 driving issue、PR base 和跨仓 owner，并为确实新增的运行协议/结构决定记录 ADR；
本文不等于这些 scope 决策已经获批。

建议分为三个可判定结果，而不是一开始承诺全引擎生产化：

1. **固定联合训练成立：** 两个来源、共同任务/DR、固定 50:50、单 learner；
   先用确定性 fake factories 验证聚合，再使用通过 parity/capability 检查的实际 backend。
   MuJoCo/Motrix 是候选，不直接复用上述不兼容默认配置，也不声称本轮已跑通。
2. **自适应组权重有效：** 固定采集 `q`，加入平衡反馈、控制器、checkpoint 和可审计权重；
   首先回答相比 uniform/fixed 是否提高鲁棒性，而不是吞吐。
3. **采集预算值得自适应：** 只有第二步显示稳定收益，才调整 quota，验证真实时间/资源收益。
   更多引擎、参数 ADR、engine-conditioned critic 和异步训练分别作为新增 scope 评估。

## 实验、可证伪性与理论限制

**建议实验，不是已运行结果：**

- 基线至少包括各单引擎 + 相同参数 DR、均匀混合、少量固定配方、自适应组权重；
  固定配方只用 validation 选优，不能报告用 test 挑出的“最优静态”。
- 比较固定 valid-transition budget 和固定 wall-clock budget 两种视角，记录相同资源配置及全部评估开销；
  不能用更多 GPU、样本或额外评估掩盖控制器成本。优先至少 3 个训练种子并报告不确定性，资源许可再扩充。
- 报告训练域 macro-average、最差域、逐域成功/跟踪误差、独立场景测试、权重轨迹/熵、ESS、实际 `q`、
  policy lag、吞吐、采样和评估耗时；改善训练加权回报本身不足以证明收益。
- 明确留一个从不向 controller 提供反馈的 backend/域作为 unseen test，且先确认该任务实际可评估；
  一旦用它调权重、超参或选 checkpoint，就不能继续称它为未见测试域。
- 做去掉平衡评估、覆盖下限/温度、慢更新、动态 quota 的消融；
  不可达域、重复/高度相关引擎、域间负迁移、快慢差异应作为失败案例，而不是只呈现成功配方。
- 验收目标是自适应方案在预先选定的独立指标/预算下优于 uniform，并不超出预先规定的退化容限；
  没有增益或只是样本预算增加，就保留固定混合，拒绝扩展调度复杂度。

可以复用 [DR manager tests](../../tests/dr/test_manager.py)、[event tests](../../tests/envs/mdp/test_events.py)、
[backend conformance](../../tests/base/test_backend_conformance.py)、[Sim2Sim tests](../../tests/training/test_sim2sim_resolver.py)、
[autoreset tests](../../tests/base/test_np_env.py)、[runner 接线测试](../../tests/scripts/test_train_scripts.py)、
[APPO spawn/close tests](../../tests/algos/test_appo_runner.py) 和
[collector failure tests](../../tests/algos/test_offpolicy_double_buffer_runner.py)。
新增算法测试应落在 `uni_rl`：同风险回到 prior、单域退化、边界/NaN/缺失统计、`p=q` 退化为普通均值、
不等样本数时手算组权重、ESS 回退、配额取整、terminal/critic/source 保真、断点恢复和域故障。

**论文不能证明任意比例都好。** [PolySim v3][paper] 第 IV 节的 proof 选择真实动力学距离下的最优凸混合，
并依赖凸包内存在更接近真实动力学的假设；实际并未提供由真实 `T0` 求配比的在线算法。
更小的某个误差上界不能直接推出任意实际配方/策略的迁移表现都优于所有单引擎。
本提案尤其不把 trajectory mixture 自动当成该逐状态 transition-kernel 凸混合。

论文 Table I 的 MuJoCo 成功率为：IsaacGym 单训 `0.500`，IsaacSim+IsaacGym `0.429`，
三引擎 `0.564`；因此至少在该实验里，“多加一个引擎”并不单调改善迁移。
这支持研究配比的动机，不证明本文反馈控制器会找到更好的配方。
论文仿真对比注明没有参数 DR，而实机部署使用了参数 DR；不能把两种实验口径混为一谈。[论文][paper]

没有真实反馈或独立目标域证据，只能声称优化已知仿真域的鲁棒性代理，不能声称学到了真实世界的最优比例。
如果用户已有真实轨迹，可另行设计受控的目标域校准；不能自动要求在线真机试错。
“自适应 DR”本身也不是新的研究概念，已有 [Active Domain Randomization][active-dr] 等工作。
若希望形成研究贡献，后续可检验“哪个域的额外更新改善其他域/独立验证指标”的迁移收益分配，
但这需要额外对照或梯度/反事实估计，不能把最高当前失败率直接等同于最高跨域训练价值。

## 本轮验证记录

- 仓库和论文/源码只读检查已完成；没有运行训练、物理仿真或 benchmark。
- 使用 Perl 检查本文全部 27 个本地 Markdown 链接：目标全部存在；独立空白与代码围栏检查无错误。
- `git diff --no-index --check` 对新增文件与 `/dev/null` 的比较返回 1，未输出空白错误；
  该结果不作为通过证据。另用临时 Git index 纳入新增文件后执行 `git diff --cached --check`，通过（退出 0）；
  未改变用户的实际暂存区。
- `uv run pytest tests/scripts/test_check_docs.py -q`：退出 127，`uv` 未找到，测试未执行。
- `make check`：退出 2，在 `uv run ruff format` 处因缺少 `uv` 失败，未完成格式/类型检查。
- `make test`：退出 2，在 `uv run pytest -m "not slow"` 处因缺少 `uv` 失败，未执行测试。
- 本文不进入 Sphinx 发布树；未构建 Sphinx，也未为研究提案安装依赖或运行 `make test-all`。
  未创建/更新 PR；正式仓库 gate 仍未通过，不能把文档检查当成训练方案验证。

## 一手来源索引

所有源码行号对应同一固定提交；下列引用可直接打开到证据所在行。

[paper]: https://arxiv.org/html/2510.01708v3
[tree]: https://github.com/EmboMaster/PolySim/tree/0fdb349785479b5c0c30d3c676fae67025b8d34a
[commit]: https://github.com/EmboMaster/PolySim/commit/0fdb349785479b5c0c30d3c676fae67025b8d34a
[readme-training]: https://github.com/EmboMaster/PolySim/blob/0fdb349785479b5c0c30d3c676fae67025b8d34a/README.md#L229-L259
[readme-eval]: https://github.com/EmboMaster/PolySim/blob/0fdb349785479b5c0c30d3c676fae67025b8d34a/README.md#L262-L282
[readme-status]: https://github.com/EmboMaster/PolySim/blob/0fdb349785479b5c0c30d3c676fae67025b8d34a/README.md#L13-L49
[overall]: https://github.com/EmboMaster/PolySim/blob/0fdb349785479b5c0c30d3c676fae67025b8d34a/humanoidverse/config/base_overall.yaml#L13-L34
[launch]: https://github.com/EmboMaster/PolySim/blob/0fdb349785479b5c0c30d3c676fae67025b8d34a/humanoidverse/polysim_train_agent.py#L161-L230
[client-create]: https://github.com/EmboMaster/PolySim/blob/0fdb349785479b5c0c30d3c676fae67025b8d34a/humanoidverse/polysim_train_agent.py#L62-L95
[learner]: https://github.com/EmboMaster/PolySim/blob/0fdb349785479b5c0c30d3c676fae67025b8d34a/humanoidverse/polysim_train_agent.py#L249-L282
[server]: https://github.com/EmboMaster/PolySim/blob/0fdb349785479b5c0c30d3c676fae67025b8d34a/humanoidverse/hydra_server.py#L167-L192
[server-step]: https://github.com/EmboMaster/PolySim/blob/0fdb349785479b5c0c30d3c676fae67025b8d34a/humanoidverse/hydra_server.py#L22-L45
[client-init]: https://github.com/EmboMaster/PolySim/blob/0fdb349785479b5c0c30d3c676fae67025b8d34a/humanoidverse/env_client.py#L29-L75
[distribution]: https://github.com/EmboMaster/PolySim/blob/0fdb349785479b5c0c30d3c676fae67025b8d34a/humanoidverse/env_client.py#L145-L165
[concat]: https://github.com/EmboMaster/PolySim/blob/0fdb349785479b5c0c30d3c676fae67025b8d34a/humanoidverse/env_client.py#L187-L239
[info-merge]: https://github.com/EmboMaster/PolySim/blob/0fdb349785479b5c0c30d3c676fae67025b8d34a/humanoidverse/env_client.py#L241-L279
[client-step]: https://github.com/EmboMaster/PolySim/blob/0fdb349785479b5c0c30d3c676fae67025b8d34a/humanoidverse/env_client.py#L281-L311
[rpc-wait]: https://github.com/EmboMaster/PolySim/blob/0fdb349785479b5c0c30d3c676fae67025b8d34a/humanoidverse/env_client.py#L126-L143
[ppo-config]: https://github.com/EmboMaster/PolySim/blob/0fdb349785479b5c0c30d3c676fae67025b8d34a/humanoidverse/config/algo/ppo.yaml#L3-L29
[ppo-storage]: https://github.com/EmboMaster/PolySim/blob/0fdb349785479b5c0c30d3c676fae67025b8d34a/humanoidverse/agents/ppo/ppo.py#L99-L128
[train-loop]: https://github.com/EmboMaster/PolySim/blob/0fdb349785479b5c0c30d3c676fae67025b8d34a/humanoidverse/agents/ppo/ppo.py#L168-L194
[rollout]: https://github.com/EmboMaster/PolySim/blob/0fdb349785479b5c0c30d3c676fae67025b8d34a/humanoidverse/agents/ppo/ppo.py#L236-L308
[advantage]: https://github.com/EmboMaster/PolySim/blob/0fdb349785479b5c0c30d3c676fae67025b8d34a/humanoidverse/agents/ppo/ppo.py#L331-L360
[update]: https://github.com/EmboMaster/PolySim/blob/0fdb349785479b5c0c30d3c676fae67025b8d34a/humanoidverse/agents/ppo/ppo.py#L362-L377
[adaptive-lr]: https://github.com/EmboMaster/PolySim/blob/0fdb349785479b5c0c30d3c676fae67025b8d34a/humanoidverse/agents/ppo/ppo.py#L412-L429
[loss]: https://github.com/EmboMaster/PolySim/blob/0fdb349785479b5c0c30d3c676fae67025b8d34a/humanoidverse/agents/ppo/ppo.py#L431-L465
[storage]: https://github.com/EmboMaster/PolySim/blob/0fdb349785479b5c0c30d3c676fae67025b8d34a/humanoidverse/agents/modules/data_utils.py#L22-L83
[minibatch]: https://github.com/EmboMaster/PolySim/blob/0fdb349785479b5c0c30d3c676fae67025b8d34a/humanoidverse/agents/modules/data_utils.py#L99-L114
[obs]: https://github.com/EmboMaster/PolySim/blob/0fdb349785479b5c0c30d3c676fae67025b8d34a/humanoidverse/config/obs/motion_tracking/deepmimic_a2c_nolinvel_LARGEnoise_history.yaml#L4-L79
[group-dro]: https://arxiv.org/abs/1911.08731
[active-dr]: https://arxiv.org/abs/1904.04762
