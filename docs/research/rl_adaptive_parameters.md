# RL 自适应参数：Isaac Lab、RSL-RL 与当前 G1 训练

状态：只读源码研究，未启动仿真、训练或 GPU，未运行测试；不是运行效果验收。
核对日期：2026-09-12，UTC+08:00。
范围：环境侧 curriculum、learner 侧自适应，以及当前 UniLab G1 多仿真器训练的启用状态。

## 来源与边界

- 本地目录：`/home/wsm/IsaacLab-2.3.2`。
- `git -C /home/wsm/IsaacLab-2.3.2 log -1 --format='%H%n%D%n%s%n%cI'`：
  提交 `37ddf626871758333d6ed89cf64ad702aef127d0`，标签 `v2.3.2`，
  提交时间 `2026-01-29T10:16:56-08:00`，显示 `grafted`，历史不完整。
- `git -C /home/wsm/IsaacLab-2.3.2 status --short --branch`：分离 HEAD，无变更条目。
  因此以下是这个固定版本的源码事实，不声称是通用默认或最新版本行为。
- 来源为本地 Isaac Lab 官方项目源码；没有进行网络扩展研究。
  本文讨论 Isaac Lab 环境/任务层，不能把这些机制归为 Isaac Sim 物理引擎自动调参。

## 1. 管理器负责调度，不内置统一自适应算法

`CurriculumManager` 从配置收集 `CurriculumTermCfg`，跳过 `None`；
`compute(env_ids)` 按配置项调用 `func(env, env_ids, **params)` 并保存返回值。
函数既可按训练时间修改参数，也可按表现反馈修改参数；“注册为 curriculum”本身不等于闭环自适应。[管理器][manager]

标准 `ManagerBasedRLEnv._reset_idx` 在场景 reset、reset 随机化事件、各 manager reset **之前**调用课程 compute。
因此课程可以读取尚未清空的状态；触发单位是待重置的环境子集，不是 PPO 梯度更新或每个物理 substep。
该调用也位于初始/手动 reset 所经过的路径，不能理解为只在成功 episode 结束时运行。[重置顺序][reset]
`reset()` 主要输出 `Curriculum/...` 日志并重置有状态课程项，不是执行课程更新的入口。[管理器][manager]

## 2. 地形课程是真正的表现反馈

`terrain_levels_vel` 在 reset 前计算二维位移 `distance = ||root_xy - env_origin_xy||`：

- 升级：`distance > terrain_generator.size[0] / 2`。
- 降级：`distance < ||command_xy|| * max_episode_length_s * 0.5`，且本次不升级。
- 否则保持等级。[判定实现][terrain-term]

这是**位移代理指标**，不是轨迹累计路程、平均奖励或速度误差积分；
使用 reset 时的速度命令和最大 episode 时长，而非实际存活时长。
若中途命令变化，不能把门限解释为整段 episode 的精确目标距离。[判定实现][terrain-term]

`update_env_origins` 对相应环境的等级加一/减一，下界截到零；超过最高等级后随机分配合法等级，
然后按等级与 terrain type 选择新的环境原点。不是每次 reset 都重新生成一套地形。
课程返回的日志值是**全部环境**的平均等级，虽然函数文档写的是给定环境集合。[等级更新][terrain-update] [返回值][terrain-term]

## 3. 地形课程是否启用由任务配置决定

`LocomotionVelocityRoughEnvCfg` 挂载 `CurriculumCfg()`，其中 `terrain_levels` 指向上述函数；
其 `__post_init__` 根据该项是否为 `None` 设置地形生成器的 `curriculum` 开关。[Rough 配置][rough]
相反，`G1FlatEnvCfg` 改用 plane、移除 generator，并明确设置 `self.curriculum.terrain_levels = None`。[G1 Flat][flat]
所以“框架提供地形课程”和“某个训练任务启用地形课程”必须分开陈述。

## 4. 奖励权重示例是时间调度，不是性能自适应

通用 `modify_reward_weight` 仅判断 `env.common_step_counter > num_steps`；超过后，
通过 reward manager 把指定项权重直接设为目标值，并返回当前权重。[权重机制][weight]
它没有读取成功率、奖励趋势或 KL，也没有在两端之间连续插值。
`common_step_counter` 每次环境 `step()` 加一，是并行环境共享的环境步数，
不是乘以环境数的 transition 总数，也不是 learner iteration。[计数器][counter]
按标准 reset 调用路径，权重变化发生在越过门限后的下一次课程 compute，不保证恰好门限那一步修改。

具体例子：Lift 基类配置将 `action_rate` 和 `joint_vel` 初始惩罚权重设为 `-1e-4`，
课程配置在 `num_steps=10000` 后把两者设为 `-1e-1`，并挂载该课程配置。[Lift 配置][lift]
**设计解释：** 可以理解为先降低动作平滑约束、之后增强约束；但这种固定时间安排不会判断策略是否真的学会任务。

## 5. DexSuite ADR 是任务族实例，不是框架通用默认

DexSuite 的 `DifficultyScheduler` 在待重置环境上检查物体目标位置误差，配置旋转容差时也检查姿态误差。
满足容差升一级，否则降一级；`promotion_only=True` 时失败不降级；结果截断在配置上下界。
它使用 reset 时的误差判定，不是成功率滑动窗口。[难度控制器][adr-impl]

控制器计算 `difficulty_frac = mean(all_env_difficulties) / max(max_difficulty, 1)`；
`initial_final_interpolate_fn` 根据该全局比例对初值与终值线性插值，比例小于 `0.1` 时返回 `NO_CHANGE`。
因此这里是局部环境表现更新难度、全体平均难度驱动共享配置变化，不是每个环境拥有一套独立配置范围。[插值与更新][adr-impl]

该任务族的 ADR 配置显式选择初始/最低难度零、最高难度十，
并配置关节观测噪声边界、重力事件分布参数等修改项；`DexsuiteReorientEnvCfg` 挂载该配置。
这些是该任务配置的具体选择，不代表所有 Isaac Lab locomotion/DR 任务默认启用 ADR。[ADR 配置][adr-cfg] [任务挂载][dex-cfg]

## 6. 概念与设计原则

**概念解释，非额外源码承诺：** 时间调度是 `parameter = f(step)`；闭环自适应是
`parameter_next = f(parameter_now, measured_performance)`；固定分布中不断重抽参数则只是随机化，
只有分布边界/难度也随反馈变化时，才构成这里所说的自适应随机化。
这些都是环境侧规则，不要求通过梯度学习课程参数，也不等价于策略网络学习或优化器自适应。

设计课程时应明确六件事：被调整量、反馈指标、测量窗口、触发时机、作用范围、上下界/回退规则。
本版本例子分别展示了按环境的等级控制、全局权重时间切换和局部表现聚合后的全局参数插值。
其共同动机是改变训练数据/任务难度；效果仍需对应任务实验确认，源码不能证明提高收敛或泛化。

## 7. Learner 侧：不同类型的自适应

以下依据当前隔离环境安装的 RSL-RL 源码及 `uni_rl` 配置转换，不推广为所有 PPO 实现的统一规则。

- **KL 学习率反馈。** 每个 minibatch 比较 rollout 旧策略与当前策略的平均 `KL(old || new)`。
  当 `schedule=adaptive`，KL 大于目标的两倍时 `lr=max(1e-5, lr/1.5)`；
  在零与目标一半之间时 `lr=min(1e-2, lr*1.5)`；否则保持不变。
  这是优化步长的反馈控制，不会调 PPO clip 或 DR，也不是严格信赖域保证。[PPO][ppo]
- **有效配置需要追踪适配器。** `normalize_ppo_train_cfg` 删除旧的
  `adaptive_kl_beta`、`adaptive_lr_growth/decay/update_interval`、`target_kl_stop` 等字段。
  当前 `enable_compile=false` 路径使用上述 RSL-RL 规则，不能把 YAML 中的 1.1/1.2 当作有效倍率。[配置转换][adapter] [调用路径][ppo-wrapper]
- **探索标准差是梯度学习参数。** `GaussianDistribution(std_type=scalar)` 创建长度为动作维数的
  `nn.Parameter`；不是全动作共用一个标量。均值依赖观测，标准差按动作维度学习、在环境间共享，
  通过 PPO likelihood 和 entropy 的梯度更新，不保证单调下降。[动作分布][distribution]
- **Adam 是逐网络参数步长缩放。** 默认优化器使用梯度一阶/二阶矩，更新近似为
  `-lr * m_hat / (sqrt(v_hat) + epsilon)`；这与全局 KL 学习率调节可以同时存在。[优化器选择][optimizer]
- **统计归一化。** `EmpiricalNormalization` 用累计样本更新均值、方差和计数，输出
  `(observation-mean)/(std+epsilon)`；这是数据统计更新，不是奖励反馈。评估应冻结统计量。
  关闭观察归一化不等于关闭优势归一化；当前 PPO 仍在整个 rollout 上标准化 advantage。[观察统计][normalization] [优势归一化][advantage]

## 8. 当前 10000 轮 G1 多来源训练

依据保存的实际运行配置，不是从单后端示例推断。[运行快照][run]

| 机制/参数 | 当前状态 |
| --- | --- |
| KL 自适应学习率 | 开启；初始 0.001，目标 KL 0.01；联合 minibatch 的全局反馈，非每引擎分别控制 |
| Adam、探索标准差 | 开启；29 个共享的可学习标准差，初值均为 1.0 |
| entropy 系数、PPO clip | 固定为 0.01、0.2；没有自动 entropy-temperature 调节 |
| 观察统计归一化 | 关闭；rollout advantage 归一化仍开启 |
| 地形/速度/奖励课程 | 当前 flat 多来源配置未注册 curriculum |
| 物理 DR | pelvis mass [0.8,1.2]，KP/KD [0.9,1.1]；reset 相对标称值重抽，范围固定 |
| 速度命令范围 | vx [0.4,0.7]，vy/wz 为零，范围不随表现变化 |
| 仿真器占比 | 四源各 2000 环境，25% 固定；不是自适应来源采样 |

UniLab 还提供未在本次 flat 训练启用的 `G1PenaltyCurriculum`：缓存原始负奖励权重，
episode 长度指标过低则缩小 penalty scale、过高则放大，并截断到允许范围。[惩罚课程][penalty]
Motion tracking 的自适应采样则累计失败计数，做 EMA、加均匀底数、可选平滑后归一化抽样。
这里是失败**计数**而非按曝光次数校正的失败概率；均匀加数不等于固定比例的探索下限。[困难片段采样][motion]

## 9. 对自适应仿真器占比的设计启示

**以下是设计建议，不是 Isaac Lab 默认功能，也未实施：** 用每引擎固定评估分布下的成功率、
速度误差或学习进展作为反馈，做平滑与慢速、有上下限的比例更新，保留所有引擎的最低覆盖。
原始 reward 容易受任务尺度、reward curriculum 和不同 episode 长度影响，不能直接作为难度排名。
更改采样占比会改变默认优化的混合目标；必须先明确要优化固定均匀平均、最差引擎还是其他目标。
来源 loss 权重与实际 rollout 样本占比是不同控制量；固定四源环境数和 rollout 长度时，
只改 loss 权重不会改变物理采样比例或提升采样吞吐。当前阶段仍保持固定 25%，未改变实验定义。

## 核对记录

执行了 `git status`、`git log`、`rg`、`sed`、`ls`，并经 `uv run --no-sync python` 只读解析运行配置，读取适用指引并核实文件路径；
缺失指引/目标文件的 `ls` 返回非零仅表示路径不存在。仅新增本文，保留 UniLab 既有脏树。
未改动任何运行时、配置、SDK、依赖或其他研究笔记；按本次研究约束不运行测试。

[manager]: /home/wsm/IsaacLab-2.3.2/source/isaaclab/isaaclab/managers/curriculum_manager.py:23
[reset]: /home/wsm/IsaacLab-2.3.2/source/isaaclab/isaaclab/envs/manager_based_rl_env.py:349
[terrain-term]: /home/wsm/IsaacLab-2.3.2/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/curriculums.py:27
[terrain-update]: /home/wsm/IsaacLab-2.3.2/source/isaaclab/isaaclab/terrains/terrain_importer.py:314
[rough]: /home/wsm/IsaacLab-2.3.2/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py:278
[flat]: /home/wsm/IsaacLab-2.3.2/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/flat_env_cfg.py:13
[weight]: /home/wsm/IsaacLab-2.3.2/source/isaaclab/isaaclab/envs/mdp/curriculums.py:24
[counter]: /home/wsm/IsaacLab-2.3.2/source/isaaclab/isaaclab/envs/manager_based_rl_env.py:202
[lift]: /home/wsm/IsaacLab-2.3.2/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/lift_env_cfg.py:155
[adr-impl]: /home/wsm/IsaacLab-2.3.2/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/dexsuite/mdp/curriculums.py:22
[adr-cfg]: /home/wsm/IsaacLab-2.3.2/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/dexsuite/adr_curriculum.py:13
[dex-cfg]: /home/wsm/IsaacLab-2.3.2/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/dexsuite/dexsuite_env_cfg.py:405
[ppo]: /home/wsm/wang-sm/UniLab/.tmp/multisim-gpu/lib/python3.11/site-packages/rsl_rl/algorithms/ppo.py:269
[adapter]: /home/wsm/wang-sm/unilab_rl/src/uni_rl/algos/rsl_rl.py:119
[ppo-wrapper]: /home/wsm/wang-sm/unilab_rl/src/uni_rl/algos/rsl_rl_ppo.py:149
[distribution]: /home/wsm/wang-sm/UniLab/.tmp/multisim-gpu/lib/python3.11/site-packages/rsl_rl/modules/distribution.py:140
[optimizer]: /home/wsm/wang-sm/UniLab/.tmp/multisim-gpu/lib/python3.11/site-packages/rsl_rl/algorithms/ppo.py:49
[normalization]: /home/wsm/wang-sm/UniLab/.tmp/multisim-gpu/lib/python3.11/site-packages/rsl_rl/modules/normalization.py:15
[advantage]: /home/wsm/wang-sm/UniLab/.tmp/multisim-gpu/lib/python3.11/site-packages/rsl_rl/algorithms/ppo.py:207
[run]: /home/wsm/wang-sm/UniLab/logs/rsl_rl_ppo/G1WalkFlat/20260912_205814_multisim_10000/run_config.json:3350
[penalty]: /home/wsm/wang-sm/UniLab/src/unilab/tasks/locomotion/g1/manager_terms.py:671
[motion]: /home/wsm/wang-sm/UniLab/src/unilab/tasks/motion_tracking/common/motion_loader.py:502
