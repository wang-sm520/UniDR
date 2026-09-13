---
orphan: true
---

# ADR-0010 同步多来源训练环境

- Status: Accepted
- Date: 2026-09-12
- Owners: UniLab training / uni_rl IPC / unisim backend maintainers
- Supersedes: None
- Superseded by: None

## Context

同一策略需要从多种物理后端同时采样，才能把仿真器选择作为固定比例的
domain randomization。单后端 task owner 可能包含不同的奖励、动作缩放和
终止条件，不能直接拼接并声称在训练同一个任务。独立发布的 `uni_rl` 只消费
注入的环境，`unisim` 只拥有物理能力，UniLab 则负责任务配置与装配。

## Decision

在 `uni_rl.ipc` 实现对外满足现有 `EnvProtocol` 的同步多来源 module。
UniLab 注入可 pickle 的 `EnvFactory` 和来源配置，PPO 继续使用一个 wrapper、
一个 rollout storage 和共享 Actor/Critic。任务选择不进入 IPC implementation，
物理参数写入不进入 UniLab 或 learner。

`multisim` 是训练 task owner 的身份，不是新的 `SimBackend`。真实后端只从
该 owner 的来源列表选择。各来源的任务语义必须一致；source ID 只用于诊断
和日志，不增加到策略或价值网络的观测。

## Stable Contracts

- 来源顺序和环境配额在构造后固定。每步先提交全部来源，再等待同步屏障，
  最后按配置顺序发布完整结果。共享内存传输大数组，pipe 传输控制消息。
- reset 仍返回选中行的 `(obs_dict, info_dict)`，并保持调用者的行顺序。
  拼接必须保留 terminal observation、mask 与 timeout bootstrap 语义。
- step/reset 非幂等，失败后不重试、不降级、不返回部分结果。失败会终止全部
  自有来源进程树；关闭操作幂等。`uni_rl` 不 import `unilab` 或 `unisim`。
- 来源进程使用独立的 CPython resource tracker，清理 native crash 后的
  Python-tracked 嵌套共享内存；不扫描或删除其他任务的共享内存。
- DR 在 UniLab 的 reset transaction 中采样，经现有
  `SimBackend.set_state(randomization=...)` 写入；未经过效果验证的能力
  不得宣称支持。资产解析与索引映射仅在冷路径执行。
- 训练保存共同策略 contract 和各来源配置。独立评估在环境构造前校验
  Sim2Sim contract 和 checkpoint 维度，不为多来源放宽 DENYLIST。
- 固定比例约束作用于完整 rollout 和完整 PPO epoch，不要求随机 minibatch
  内严格等额。单 GPU 不保证不同来源的物理计算同时执行。
- 有界验收复用 stock PPO learning loop，在完整更新后的日志边界结束并
  保存；预热及其资源采样不计入 60 分钟正常训练计时。

## Alternatives Considered

- 新建多来源 PPO runner：可以显式追踪 rollout provenance，但会重复或侵入
  已有 PPO 采样循环；固定比例同步实验不需要这种额外耦合。
- UniLab 内实现 RPC collector：把 IPC、生命周期和日志放到错误的 owner。
- 组合 `SimBackend`：需要转发大量物理方法，且无法隔离完整环境与 runtime
  的进程级状态。

## Consequences

三个仓库需要联调并记录版本。来源最慢路径和单 GPU 竞争决定实际吞吐，
必须用真实全规模测试验证，不能用小规模 fake 测试代替容量证据。
首次交付限定于 G1 四来源同步 PPO 的训练就绪闭环；不承诺收敛、正式
24 小时训练、动态比例或生产级多机支持。

## Evidence In Repo

- `src/unilab/base/env_factory.py`
- `src/unilab/base/config_adapter.py`
- `src/unilab/scripts/train_rsl_rl.py`
- `src/unilab/conf/ppo/task/g1_walk_flat/base.yaml`
- `src/unilab/utils/sim2sim.py`

## Related Documents

- {doc}`ADR 索引 </adr/ADR-0000-index>`
- {doc}`运行时分层 </adr/ADR-0001-runtime-model-and-layer-boundaries>`
- {doc}`Backend 能力 </adr/ADR-0002-backend-capability-boundary-for-play-and-snapshot>`
- {doc}`Task owner </adr/ADR-0003-task-owner-and-config-compose-contract>`
- {doc}`观测与 IPC </adr/ADR-0005-unified-obs-critic-env-and-ipc-contract>`
- {doc}`协作流程 </en/4-developer_guide/5-contributing_workflow>`
