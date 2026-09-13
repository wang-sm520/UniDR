# G1 自适应仿真器占比：反馈控制器与实现约束

**状态：设计，尚未实现；不改变正在运行的 fixed25% 实验。**
核查日期：2026-09-12；依据当前脏工作树，不把未提交代码视为已发布能力。
UniLab HEAD `db1a6e5b3dbe4dd096b16a517837f2d46dba8164`；uni_rl HEAD `79418e0cbff7b95fe6e49709b194454701ba20fa`。
`.venv` 的 [editable 路径][editable] 指向 sibling uni_rl；rollout 依据本地 RSL-RL 5.0.1 源码，未检查运行进程的导入快照。
[旧提案][prior] 是 2026-09-11 的历史设计；其中“没有 sibling 源码”的限制不再代表本次核查条件。
本文同时给出反馈控制器提案和配额、存储、生命周期、成本、评估隔离约束；所有新参数均为待验证起点。

## 为什么实际采样仍是 25%

- [组合 owner][owner] 固定四源各 2000；第 124 行校验总数 8000、第 128 行校验 H=24，第 194 行逐源拒绝其他数量。
- [runner][runner] 每次迭代对整个联合 env 调用 24 次 step；[IPC step][ipc-step] 总是发送所有固定分片，[worker][worker] 每次累计该源完整 num_envs 个 transition。
- 因此每源 B_i=2000*24=48000，总 B=192000，实际生成占比 q_i=B_i/B=25%；速度、episode 长短和失败率不改变这个 transition 比例。
- [屏障][barrier] 等待所有源 ACK：慢源拖长墙钟时间，不会自动少采。完成 episode 的来源比例可以不同，不能把它当作 q。

## 真实配额的可选路径

以下均为尚未实现的设计路径，不是可直接设置的现有开关。

| 路径 | 可改变什么 | 当前约束与代价 |
| --- | --- | --- |
| 固定 H，改变活跃 n_i | q_i=n_i/sum(n_j)；保持总活跃数可保持 learner 的 [H,N] 形状 | 需新活跃槽位与来源映射协议；物理容量、IPC 分片不随 learner 的固定总形状自动变化。 |
| 预分配每源最大容量 | 避免每次调配额都重建引擎 | 各容量仍为 2000 且总活跃数仍为 8000 时没有调整空间；需额外容量或降低总有效 batch，并承担常驻内存。 |
| 所有源统一改变 H | 改变总 batch 和更新频率 | n_i 不变时 H 在比例中抵消，仍是 25%；且当前 owner 拒绝 H!=24。 |
| 各源不同 H_i / 不同数量完整 unroll | q_i=n_i*H_i/sum(n_j*H_j) | 当前同步 step 接口无法独立推进来源；需分源采集调度、独立 bootstrap 与有效长度存储。 |

[IPC 初始化][ipc-init] 一次确定总数、分片和共享内存；[wrapper][wrapper] 一次分配 episode 统计。
[storage][storage] 分配固定 [H,N,...] 张量，第 170 行按整行写入，第 204 行 clear 只重置游标；
[展平采样][batches] 遍历配置的 H*N，不是“已经填了多少有效数据”，现有 Batch 也没有来源配额字段。
因此仅改 num_envs/H 属性或只填一部分 buffer 会造成形状不匹配或旧值参与训练；需要显式映射、有效性和存储生命周期。
不同 H_i 还改变 GAE 截断与 bootstrap 依赖，不能解释成纯配比消融；[当前返回计算][returns] 使用统一 H 反向递推。

**活跃不等于停算。** [SimBackend.step][backend-step] 接收完整 (num_envs,nu)，没有活跃索引参数；
下采样已生成轨迹只能改变“进入训练的 q”，不能改变物理生成 q 或承诺节省仿真成本；零动作也不是暂停。
真正跳过槽位必须先扩展公开 backend 能力，再协调 env 的计时、事件、DR、终止与统计，不可调用私有接口。
[IPC reset][ipc-reset] 只重置既有索引，不扩容、不迁移来源；[env reset][env-reset] 要求首次全量 reset，并会执行 curriculum/reset 事件。
容量重建或槽位换源需在 rollout/update 边界处理：保留未变槽位的轨迹身份，定义停用/恢复及新槽位 reset 语义；
切换时不能把旧源最后状态与新源首状态拼成一条 GAE 链，也不能把暂停观测重复登记为新 transition。
采集使用明确的 policy 版本与新鲜轨迹；等待旧 episode 结束或强制 reset 都会改变调度/状态分布，必须记录。

## 采样与损失权重分开验证

用户最终目标是**实际采样比例自适应**；loss 权重自适应只是先行可行性验证，不能冒充已经实现该目标。
固定 q=25% 时改变各源目标权重 p，只改变优化压力；分组均值或按实际 q 校正的 p/q 必须保持来源标识随 minibatch 洗牌一致。
不能用少数轨迹反复重放来伪造新增 on-policy 配额；应分别记录目标 p、物理生成 q、有效训练 q 和独立轨迹覆盖。
建议先在独立实验验证 loss-weight adaptation 的收益和稳定性，再单独审定真实采样协议；固定25% 对照保持不动。
协议 owner 属于 uni_rl，任务选择/快照属于 UniLab，物理活跃能力属于 unisim；参照 [ADR-0010][adr]，本文不批准新增公共协议。

## 时间成本与评估风险

- [现有计时][timing] 记录各源 step 时长、p95、transitions、collect/learn 秒数；第 150 行明确：step 含 autoreset、不含外层 IPC、不额外同步 CUDA，计时项可重叠。
- 各步最大源时长之和不是实测 barrier 时长；collect 减去它还含推理、复制、调度和日志，不能全部记为 IPC，也不是独立 GPU kernel 时间。
- [当前拓扑][topology] 把 learner 与三个 GPU 来源放在 GPU0。成本风险：CPU 线程饱和/内存带宽、GPU launch 开销/批量利用率/显存与跨进程争用都可能非线性；不能按当前秒数/n_i 线性外推缩容收益。
- 后续仅在独立验证中测稳态 n_i 变化下的端到端墙钟、有效 transitions/s、尾延迟与 reset 成本，并分清 CPU 墙钟与 GPU 执行；本轮不测量，也不把高耗时等同高学习价值。
- [现有评估][evaluation] 使用固定 seeds 101--104、每源 100 个首 episode，四源等 episode 权重汇总。若这些结果反馈控制、选择超参数/检查点，它们就是验证集，不再是独立测试证据。
- 控制反馈应使用独立、平衡的留出 episodes，与审计 seeds 101--104 分离；最终指标固定域权重，报告逐源、宏平均与最差源，不能按自适应 p/q 加权掩盖退化。
- [评估覆盖][eval-overrides] 关闭 actor 噪声和 base_mass/pd_gains reset DR；反馈需另建保留固定训练 DR 的 profile，不能复用该覆盖后声称测到训练分布风险。在线完成 episode 还有短失败轨迹先到的偏差，必须固定统计口径与窗口。
- 仅四个已知 simulator 的留出 seeds 不证明未见动力学或 sim-to-real 泛化；本次没有训练、评估或性能实测结论。

## 建议控制器：有覆盖约束的慢速风险反馈

### 目标与反馈定义

第一版假定目标是改善四个已知引擎中的最差表现，并监测宏平均退化，不是最大化 FPS。
只改变来源训练压力，保持奖励、观测、质量/KP/KD 的固定 DR 范围以及速度命令范围不变。
这借鉴 KL 学习率控制的慢反馈、课程学习的表现信号及有界调节；不代表 PPO 原生提供该功能。

设 p 为目标训练权重、q 为实际进入 learner 的新鲜 transition 份额。
反馈必须用同一个冻结 policy 版本、每域等量完整 episodes，另设独立于审计 seeds 101--104 的反馈种子表。
可从每域 100 episodes 起步；这只是最小预算，不保证统计精度。只统计指定 cohort 的首个完整 episode，
不得只取某段墙钟时间内率先完成的 episodes。按 episode 内有效步先平均误差，再对 episodes 等权汇总。
建议使用确定性动作、关闭观察噪声但保留训练物理 DR 范围的独立反馈 profile；这测量的是该验证分布风险，
不等于训练时随机动作/噪声下的完整分布，也不是现有关闭物理 DR 的评估入口已具备的功能。

一个适用于当前 flat 任务的简洁风险例子：

```text
F_e = 1 if episode has a non-timeout termination, else 0
D_e = 1 - min(episode_time_seconds / 20, 1)
E_e = min(episode_vx_mae / 0.3, 1)
r_i = mean_episodes(0.5 * F_e + 0.2 * D_e + 0.3 * E_e)
```

上述 r 在 [0,1]；0.3 的单位为 m/s，各系数和尺度在实验开始前固定，不用逐引擎移动 min/max。
F 优先惩罚失败，D 区分同样失败但存活时长不同的策略，E 区分站着不倒和真正跟踪速度。
这是便于解释的代理指标，不是已经验证的最优组合；它没有覆盖全部行为质量。
必须另记横向漂移、角速度、动作质量及逐源失败类型，且宏平均/最差源退化 guard 使用反馈验证集，
不能拿最终审计集参与在线回退。当前 vx/vy 来自 pelvis IMU 局部速度，wz 来自 torso gyro，
不可把后者描述成 pelvis/root yaw rate。[指标语义][metric-provenance]

### 平滑与比例更新

```text
initial p = [0.25, 0.25, 0.25, 0.25]
smoothed_r = 0.8 * previous_r + 0.2 * measured_r
target_p_i = clip(c * exp(smoothed_r_i / 0.2), 0.10, 0.40)
choose c > 0 so sum(target_p) = 1
delta = max_i(abs(target_p_i - p_i))
alpha = min(0.2, 0.02 / delta) if delta > 0 else 0
next_p = (1 - alpha) * p + alpha * target_p
```

风险 EMA 首次用首个合格窗口的实际风险初始化，不用虚构的零风险。
exp 实现减去最大 logit，再对 c 单调二分求解；不能先 clip 再 normalize，因为那会破坏覆盖上下限。
候选是均匀 prior 下、带 KL 正则与上下限的风险倾斜分布；温度 0.2 决定偏向困难域的力度。
平滑和每次最多 2 个百分点限制响应速度。可行的旧 p 与候选的凸组合仍在 [0.10,0.40] 内。
无风险差异时候选回到均匀分布，不会因累积乘法不断放大旧偏好。

新实验可先均匀训练 500 次 PPO update，此后每 200 次完整 update 才检查一次反馈。
每域不足 100 完整 episodes、反馈过旧或 bootstrap 置信区间不足以支持风险差异时保持旧 p，
不把缺失数据当作零风险。置信度不足而保持不变与候选回到 prior 是两个不同的控制阶段。
这些预算与门限都不是 Isaac Lab 默认值；要计入反馈评估的显存、启动与墙钟成本后再定案。
单卡现有训练常驻时能否容纳反馈环境尚未验收，不能假定额外评估免费或随时可构造。

手算例子：按 Gym/Sim/Motrix/Genesis 顺序，若平滑风险为 [0.15,0.55,0.30,0.45]，
候选约 [0.1000,0.4000,0.1604,0.3396]；从均匀分布出发，alpha=0.13333，
下一窗口 p 约为 [0.2300,0.2700,0.2381,0.2619]。这是公式数值演示，不是实测引擎排名。

### 接入与验收

第一阶段仍生成等量物理样本，优化 `L = sum_i p_i * mean(loss_i)`；
等价于给每个样本乘窗口级 `p_i/q_i`，当前 q_i=0.25 时就是 `4*p_i`。
这里是来源校正，不能替代 PPO 新旧策略概率比。actor/value/entropy 第一版采用同一来源权重，
保留各自原有系数；minibatch 洗牌必须携带来源 ID，不按随机 minibatch 的偶然组成重定义目标权重。
保持现有全局 advantage 归一化作为对照条件，单独记录逐源及目标加权 KL，不让全局 KL 掩盖某源漂移。
p 必须在下一 rollout 开始前固定，并贯穿其所有 PPO epochs；checkpoint 保存 EMA、p、窗口与反馈 policy 版本。

第二阶段才把目标 p 映射为总活跃数 8000 下的整数 n_i，保持 H=24，并记录取整后的实际 q。
份额高于 25% 的来源需要超过 2000 的容量；若允许上限 40%，每源最大可达 3200。
全源预留最大容量会产生最多 12800 个物理槽位的常驻成本，不能假定与原 8000 容量等价；
只改变活跃标记也不保证后端停止计算，必须按前述公开接口与生命周期验收。
若 p 因容量限制不能实现，不静默降低总 batch、伪造采样比例或改变 H 来掩盖；独立报告受限结果。

长期贴上限却无进步时诊断可学习性、跨域冲突和任务一致性；NaN、crash、超时等是健康失败，
不是应该增加或删除来源的课程信号。最终是否冻结/回退按预先定义的验证容忍度和持续窗口规则决定。
第一版不加入按 FPS 或风险/耗时的比例奖励；那会改变目标。后续资源调度需在固定目标 p 下另行研究。
实验至少比较同预算 uniform、自适应 p、以及随后真实 q 自适应；统一种子/初始化和固定物理 DR，
报告逐源、宏平均、最差源、比例轨迹、权重 ESS、样本数及含反馈评估的总耗时，建议至少 3 个训练种子。
可先消融风险反馈/比例响应速度，不同时开启参数 ADR、多 GPU 或修改奖励。
改善反馈代理不证明最优 sim-to-real 比例，也不保证高失败率域有更高的训练边际收益。

## 本轮核对

用 `rg`、`sed`、`cat` 阅读当前 owner、IPC、learner、评估与计时实现；仅新增本文。
用工具内 JavaScript 算术核对上述有界分配例子的和、上下限与单次变化，没有执行训练或仿真。
未运行测试或 GPU 工作，未更改配置、协议、依赖、训练进程或已有研究笔记。

## 当前源码定位

[editable]: /home/wsm/wang-sm/UniLab/.venv/lib/python3.11/site-packages/unilab_rl.pth:1
[prior]: /home/wsm/wang-sm/UniLab/docs/research/adaptive_simulator_mixture.md:294
[owner]: /home/wsm/wang-sm/UniLab/src/unilab/training/multi_source.py:26
[runner]: /home/wsm/wang-sm/UniLab/.venv/lib/python3.11/site-packages/rsl_rl/runners/on_policy_runner.py:79
[ipc-step]: /home/wsm/wang-sm/unilab_rl/src/uni_rl/ipc/multi_source_env.py:452
[worker]: /home/wsm/wang-sm/unilab_rl/src/uni_rl/ipc/_multi_source_worker.py:149
[barrier]: /home/wsm/wang-sm/unilab_rl/src/uni_rl/ipc/multi_source_env.py:243
[ipc-init]: /home/wsm/wang-sm/unilab_rl/src/uni_rl/ipc/multi_source_env.py:102
[wrapper]: /home/wsm/wang-sm/unilab_rl/src/uni_rl/algos/rsl_rl.py:199
[storage]: /home/wsm/wang-sm/UniLab/.venv/lib/python3.11/site-packages/rsl_rl/storage/rollout_storage.py:125
[batches]: /home/wsm/wang-sm/UniLab/.venv/lib/python3.11/site-packages/rsl_rl/storage/rollout_storage.py:222
[returns]: /home/wsm/wang-sm/UniLab/.venv/lib/python3.11/site-packages/rsl_rl/algorithms/ppo.py:187
[backend-step]: /home/wsm/wang-sm/unisim/src/unisim/backend/base.py:872
[ipc-reset]: /home/wsm/wang-sm/unilab_rl/src/uni_rl/ipc/multi_source_env.py:470
[env-reset]: /home/wsm/wang-sm/UniLab/src/unilab/envs/manager_based_rl_env.py:580
[adr]: /home/wsm/wang-sm/UniLab/docs/sphinx/source/adr/ADR-0010-synchronous-multi-source-training.md:1
[timing]: /home/wsm/wang-sm/unilab_rl/src/uni_rl/algos/rsl_rl_source_timing.py:72
[topology]: /home/wsm/wang-sm/UniLab/src/unilab/conf/ppo/task/g1_walk_flat/multisim.yaml:8
[evaluation]: /home/wsm/wang-sm/UniLab/src/unilab/training/evaluation.py:542
[eval-overrides]: /home/wsm/wang-sm/UniLab/src/unilab/tasks/locomotion/g1/evaluation.py:150
[metric-provenance]: /home/wsm/wang-sm/UniLab/src/unilab/tasks/locomotion/g1/evaluation.py:37
