# PolySim 是否每个仿真器独占一张 GPU？

核对日期：2026-09-12。源码固定于官方提交
`0fdb349785479b5c0c30d3c676fae67025b8d34a`；仅静态阅读，未运行 PolySim。

## 直接结论

**官方 README 的配置意图是三个仿真器使用三个不同 GPU 编号，但训练器与 Genesis 共用编号 7；不是四进程各独占一张卡。**
独立 conda 环境、独立进程和独立物理 GPU 是不同概念；已核对的启动器没有 GPU 编号唯一性检查。
这不能升级成“所有后端已验证支持同卡运行”，也不能仅凭配置证明实际物理引擎落卡。[README][readme] [启动器][launch]

## 官方配置实际指定什么

README 将 `simulator_list=[isaacgym,isaacsim,genesis]`、`num_envs_list=[2048,2048,2048]`
与 `device_list=[5,6,7,7]` 按顺序对应，末项给 client。[示例][readme]

| 角色 | 环境数 | 指定设备编号 | 进程/依赖环境 |
| --- | ---: | ---: | --- |
| IsaacGym server | 2048 | 5 | 子进程，`hvgym` |
| IsaacSim server | 2048 | 6 | 子进程，`hvlab` |
| Genesis server | 2048 | 7 | 子进程，`hvgen` |
| PPO trainer / EnvClient | 聚合 6144 | 7 | 启动器主进程 |

- `device_list[i]` 生成 worker 的 `device=cuda:N` 和 `rpc.server_gpu=N`；
  `num_envs_list[i]` 生成其 `num_envs`。启动器用 `conda run` + `Popen` 拉起服务器，不要求手动开三个终端。[启动器][launch]
- client 的 rank 为服务器数，读取对应设备项；`EnvClient.num_envs=sum(num_envs_list)`，
  learner 实例化时使用 `env.device`。因此末项是训练设备，不是第四个仿真器。[client 构建][client-init] [learner][learner]
- **基础 YAML 与 README 示例不同：** 默认顺序是 `[isaacsim,isaacgym]`，环境数 `[4096,4096]`，
  设备 `[0,1,0]`，即两个 worker 编号 0/1，trainer 编号 0，总计 8192 环境。
  同文件的 `device: cuda:1` 和 `num_envs: 4096` 不能取代上述 client 构建逻辑。[基础 YAML][base] [client 构建][client-init]
- 在预期的 N 个仿真器、N+1 个设备项配置下，启动器按索引取值，未要求 GPU ID 互异；
  默认配置和 README 都已重复 trainer 的设备编号。**未据此验证所有 worker 共卡的可运行性或性能。**[启动器][launch] [基础 YAML][base]

## 设备编号与后端陷阱

- 启动器继承父进程环境，未按 worker 设置 `CUDA_VISIBLE_DEVICES`；只设置 `device=cuda:N` 和 RPC 映射。
  CUDA 可见设备会按可见列表重新编号，例如外部设置 `CUDA_VISIBLE_DEVICES=5,6,7` 后，
  应理解为进程内 0/1/2，不能继续把 `cuda:5/6/7` 当成有效局部编号。[启动器][launch] [NVIDIA 定义][cuda-visible]
- 若改为各 worker 只看一张物理卡，该卡通常是各自的局部 `cuda:0`，RPC 映射也必须对应各进程的局部编号。
  当前启动器直接复用 `device_list` 构造双向映射，没有物理 UUID 到各进程局部编号的转换层。[client 映射][client-init] [server 映射][server-rpc]
- **Genesis 尤其不能只看 `device=cuda:7`：** PolySim 适配器保存此字符串，却只调用
  `gs.init(backend=gs.gpu)`，没有传入编号或设置可见设备。[Genesis 适配器][genesis-adapter]
  README 要求的 Genesis 0.2.1 初始化进一步调用 `get_gpu_device()`，使用无索引 `torch.device("cuda")`、
  查询设备 0，并以无设备编号的 `ti.init(arch=...)` 初始化 Taichi。
  因而配置中的 7 **不足以证明 Genesis 物理计算实际在 7 上**；须另行核对后端初始化和运行证据。[Genesis 初始化][genesis-init] [设备选择][genesis-device]

## RPC 路径与同步边界

- client/server 均创建 `TensorPipeRpcBackendOptions`，以 `set_device_map` 配置双向 CUDA 映射；
  **这段公开源码不是显式 NCCL 通信配置，也没有证明实际使用 NVLink、零拷贝或完全不经 host staging。**[client 映射][client-init] [server 映射][server-rpc]
- client 按环境配额切分动作，经 RPC 调用服务器 `env.step`；服务器返回观测、奖励、done、info 等。
  client 用 `.to(self.device, non_blocking=True)` 和 `torch.cat` 聚合张量。
  已读 step 路径未显式先把所有张量转成 NumPy/CPU，但 `.to` 和设备映射不能证明底层实际传输通道。[client][client] [服务端 step][server-step]
- 每步线程池并发提交各 server 请求；内部 `rpc_async(...).wait()`，外层逐一 `future.result()`，
  **收齐所有服务器结果才聚合并返回**。这是并发执行、同步等待，不是快仿真器无限领先的异步采集。[client][client]
- 启动器还给 worker 设置 `CUDA_LAUNCH_BLOCKING=1`。本次未测其耗时影响。
  本地 `Popen`、client 固定 `MASTER_ADDR=127.0.0.1`、默认 loopback 网络设置，也不能作为多节点已支持的证据。[启动器][launch] [client 构建][client-init]

## 论文与当前 UniLab 的证据边界

- **论文表述，主任务已核对附件 v3，本文未重复阅读：** III-A（PDF 第 3 页）将隔离的 TrainClient/SimServers
  与多 GPU 执行、工具链独立和资源争用联系起来；III-C 声称 RPC over NCCL，借 NVLink/高速 PCIe 避免 host staging。
  后者须标为论文主张，不能覆盖公开源码使用 TensorPipe 的事实。[论文 v3][paper]
- Table II（第 6 页）及效率讨论（第 7 页）报告并行耗时接近最慢单引擎；
  主任务全文硬件词检索未发现 GPU 型号、确切卡数或完整分配。**不能由 README 编号还原论文实验硬件，亦不能外推单卡性能。**[论文 v3][paper]
- **当前本地配置，不是新 profiling：** learner 指定 `cuda:0`；IsaacGym、IsaacSim、Genesis 的 GPU 设备编号均为 0。
  Motrix 使用 CPU 物理，来源配置中继承的 `training.device` 不能作为其物理求解设备的证据。
  当前校验要求单 learner/GPU 0，并拒绝非零 `*_device_id`。[运行快照][run] [来源配置][source-config] [Motrix 设备][motrix-device] [单卡约束][single-gpu] [来源设备约束][source-gpu]
  IsaacGym 当前另有 CPU state tensor pipeline，但仍启用 GPU PhysX 求解；张量位置与物理计算位置不能混为一谈。[Gym 求解与数据设备][gym-device]
  UniRL 当前多来源 IPC 是 NumPy + `multiprocessing.shared_memory` 的 host transport，不能称为 GPU 直传；
  这是源码架构差别，**不是已经测得的瓶颈归因**。[IPC owner][ipc]

## 本次操作

只新增本文。使用 `git status --short`、`rg`、`nl -ba`、`sed`、`cat` 阅读本地文件，
使用 `curl -fsSL --max-time 15/20/30` 将官方 raw 源码及 NVIDIA 说明输出到 stdout。
未安装、导入或执行 PolySim，未启动 GPU 作业、发送信号、运行测试或修改配置；未验证实际落卡和传输性能。

[readme]: https://github.com/EmboMaster/PolySim/blob/0fdb349785479b5c0c30d3c676fae67025b8d34a/README.md#L229-L260
[launch]: https://github.com/EmboMaster/PolySim/blob/0fdb349785479b5c0c30d3c676fae67025b8d34a/humanoidverse/polysim_train_agent.py#L161-L230
[client-init]: https://github.com/EmboMaster/PolySim/blob/0fdb349785479b5c0c30d3c676fae67025b8d34a/humanoidverse/polysim_train_agent.py#L62-L102
[learner]: https://github.com/EmboMaster/PolySim/blob/0fdb349785479b5c0c30d3c676fae67025b8d34a/humanoidverse/polysim_train_agent.py#L248-L282
[base]: https://github.com/EmboMaster/PolySim/blob/0fdb349785479b5c0c30d3c676fae67025b8d34a/humanoidverse/config/base_overall.yaml#L13-L34
[server-rpc]: https://github.com/EmboMaster/PolySim/blob/0fdb349785479b5c0c30d3c676fae67025b8d34a/humanoidverse/hydra_server.py#L145-L197
[server-step]: https://github.com/EmboMaster/PolySim/blob/0fdb349785479b5c0c30d3c676fae67025b8d34a/humanoidverse/hydra_server.py#L22-L47
[client]: https://github.com/EmboMaster/PolySim/blob/0fdb349785479b5c0c30d3c676fae67025b8d34a/humanoidverse/env_client.py#L126-L310
[genesis-adapter]: https://github.com/EmboMaster/PolySim/blob/0fdb349785479b5c0c30d3c676fae67025b8d34a/humanoidverse/simulator/genesis/genesis.py#L22-L36
[genesis-init]: https://github.com/Genesis-Embodied-AI/Genesis/blob/v0.2.1/genesis/__init__.py#L35-L97
[genesis-device]: https://github.com/Genesis-Embodied-AI/Genesis/blob/v0.2.1/genesis/utils/misc.py#L106-L124
[cuda-visible]: https://docs.nvidia.com/deploy/topics/topic_5_2_1.html
[paper]: https://arxiv.org/pdf/2510.01708v3
[run]: /home/wsm/wang-sm/UniLab/logs/rsl_rl_ppo/G1WalkFlat/20260912_205814_multisim_10000/run_config.json:6
[single-gpu]: /home/wsm/wang-sm/UniLab/src/unilab/training/multi_source.py:112
[source-gpu]: /home/wsm/wang-sm/UniLab/src/unilab/training/multi_source.py:208
[ipc]: /home/wsm/wang-sm/unilab_rl/src/uni_rl/ipc/_multi_source_shared.py:1
[source-config]: /home/wsm/wang-sm/UniLab/src/unilab/conf/ppo/task/g1_walk_flat/multisim.yaml:16
[motrix-device]: /home/wsm/wang-sm/UniLab/docs/sphinx/source/zh_CN/2-user_guide/3-backends/0-index.md:30
[gym-device]: /home/wsm/wang-sm/unisim/src/unisim/backend/isaacgym/worker.py:88
