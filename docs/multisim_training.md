# G1 四仿真器 PPO 联调与验收

此实现面向训练就绪闭环，不代表已经通过真实物理、容量、60 分钟稳定性或
收敛验收。只有 G1–G7 必需 gate 全部通过，才能称为“训练就绪”。正式
24 小时训练不属于本阶段。

架构决定见 `docs/sphinx/source/adr/ADR-0010-synchronous-multi-source-training.md`。
已有 `docs/research/` 不属于本次交付的修改范围。

UniDR 单仓快照将两个修改版依赖放在 `vendor/`，通过根目录 `uv.lock` 与
editable source 配置安装；无需另行克隆兄弟仓库。见
[依赖来源与安装](../vendor/README.md)和
[最终 checkpoint 与评估命令](../checkpoints/g1_walk_flat_multisim_10000/README.md)。
下文三仓路径保留为原始开发记录；单仓的验收入口优先检查 `vendor/` 路径。

ManagerBasedRlEnv 通过既有 `algo_capabilities` 协议提供原始动作空间边界和
按 ActionManager 执行顺序拼接的关节名称。名称来自已解析 Joint action 的
`target_names`，不是从场景配置猜测；非关节 action 或重复名称不声明该字段。
多来源仍严格验证实际名称与公共 owner 一致。无界动作空间可用 `±inf` 描述
边界，但实际 action/obs/reward/terminal observation 仍要求数值有限。
training owner 直接使用与来源 worker 相同的 uni_rl NaN guard，避免将
UniLab env-layer 的同名旧类型传给 IPC。utils 不增加对 runtime 的反向依赖，
没有关闭 NaN guard。

## 固定实验

| 项目 | 设置 |
| --- | --- |
| 任务 | G1WalkFlat，29 个动作，Actor 98 维，Critic 101 维 |
| 来源顺序 | IsaacGym → IsaacSim → Motrix → Genesis |
| 配额 | 每源 2000，总计 8000，不自动缩容 |
| PPO | 一个共享 Actor/Critic，24-step rollout，共 192,000 个 transitions |
| 比例 | 每源每轮 48,000；完整 PPO epoch 等额，不限制每个随机 minibatch |
| DR | pelvis mass × `[0.8,1.2]`，全部关节 KP/KD 各 × `[0.9,1.1]` |
| DR 基准 | 每次 reset 相对缓存标称值，不累乘；不更新惯量 |
| 命令 | 机体坐标 vx `[0.4,0.7]`，vy/wz 为零 |
| 控制 | action scale 0.25，ctrl_dt 0.02 s，episode 最长 20 s |
| 公共语义 | G1 PPO base 的奖励、终止、初始状态和训练观测噪声；关闭 empirical normalization |

不包含 friction、COM、push、delay、RFI、自适应比例、episode 跨引擎迁移、
多 GPU 或实机部署。IsaacSim 接触按 link 净接触力判断，不代表足底四个角点的
精确接触重建。保留原有足部接触奖励，不能用关闭奖励来绕过物理测试。

## 三仓开发环境

2026-09-12 建立的开发基线：

| 仓库 | 基线 commit |
| --- | --- |
| UniLab | `db1a6e5b3dbe4dd096b16a517837f2d46dba8164` |
| unilab_rl | `79418e0cbff7b95fe6e49709b194454701ba20fa` |
| unisim | `1ef3bb6b04a2cd76f20be72c49cd4d15f528e51b` |

三个实际开发仓库放在同一父目录。不要编辑发布版的 `site-packages` 或临时
解包源码。保留 UniLab 的 `uv.lock`，用 Python 3.11 建立独立 `.venv`：

```bash
uv sync --python 3.11 --locked --extra mujoco --extra motrix --extra genesis
uv pip install --python .venv/bin/python --no-deps -e ../unilab_rl -e ../unisim
export UV_NO_SYNC=1
export UNILAB_LOCAL_UNISIM="$(realpath ../unisim)"
uv pip freeze --python .venv/bin/python
uv run --no-sync python -c 'import uni_rl, unisim; print(uni_rl.__file__); print(unisim.__file__)'
```

联调后使用 `uv run --no-sync` 或 `UV_NO_SYNC=1`，否则自动同步会把 editable
依赖换回 lock 中的发布版。不要为了联调原地升级现有 conda 环境。

首次 CUDA 下载未完成时另建了 `.venv-cpu` 进行 CPU gate，使用同一 lock 的
依赖，但将 torch 明确换为 `2.8.0+cpu`。这个 CPU 副本仍保留，不能执行 GPU
验收；未完成的旧环境和下载缓存也未删除。

2026-09-12 驱动恢复后，已在 `.tmp/multisim-gpu` 完成独立 GPU 环境的 locked
sync 和三仓 editable 安装。实际版本为 Python 3.11.16、PyTorch
`2.8.0+cu128`、CUDA 12.8、cuDNN 9.10.2；GPU 矩阵乘法、卷积、反向传播和
Adam 更新均通过，`torch.cuda.is_available()` 为 true。当前 `.venv` 指向这个
GPU 副本。没有修改 `uv.lock` 或已有 conda/SDK 的依赖。

本机联调命令可以先加载本地环境配置，避免自动同步覆盖 editable 依赖：

```bash
source /home/wsm/wang-sm/UniLab/.tmp/multisim-gpu.env
uv run --no-sync python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

在新机器上建立独立 GPU 环境仍使用同一锁定安装流程：

```bash
export UV_PROJECT_ENVIRONMENT="$(pwd)/.tmp/multisim-gpu"
uv sync --python 3.11 --locked --extra mujoco --extra motrix --extra genesis
uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" --no-deps -e ../unilab_rl -e ../unisim
export UV_NO_SYNC=1
```

Pyright 1.1.408 需要支持现代 JavaScript 的 Node。本机系统 Node 12 不兼容；
本次只在验证命令的 PATH 中选用已有 VS Code Server 的 Node 24.18.1，没有
全局升级 Node。Pyright 的仓库配置读取 `.venv`，因此会跟随该本地链接使用
当前 GPU editable 环境；CPU 副本可通过显式指定环境路径继续使用。

IsaacGym 使用独立 Python 3.8 runtime，IsaacSim 使用独立 Python 3.11 runtime。
可使用各 backend dependency 模块支持的环境变量或 cache 布局配置 runtime。
本机仅创建了以下新符号链接，没有修改已有 SDK/conda 文件：

| Cache 路径（相对 `~/.cache/unisim`） | 指向现有目录 |
| --- | --- |
| `isaacgym/miniconda3/envs/hsgym` | `/home/wsm/miniconda3/envs/homierl` |
| `isaacgym/isaacgym` | `/home/wsm/isaacgym` |
| `isaacsim/venv` | `/home/wsm/miniconda3/envs/atec_isaac51` |
| `isaacsim/IsaacLab` | `/home/wsm/IsaacLab-2.3.2` |

对应发现结果是 Python 3.8.20/IsaacGym 和 Python 3.11/IsaacSim 5.1.0.0/
IsaacLab 0.54.2（本地 SDK 目录版本 2.3.2）。这些只是 runtime 发现，
不是实际 DR、接触或训练成功的证据。

## 模块边界和失败语义

- `unisim`：既有 `SimBackend.set_state(randomization=...)`、DR capabilities
  和 nominal getters；Isaac 共享协议版本/能力协商、mass/gains SHM reset
  数组、初始化时 body/joint 映射以及真实 IsaacSim ContactSensor。
- `uni_rl.ipc.multi_source_env`：公开 `EnvSourceSpec`、`MultiSourceOptions`、
  `make_multi_source_env`、`MultiSourceEnvError`。每来源一个非 daemon spawn
  进程；默认启动/操作/清理 timeout 为 600/120/10 s；不 import UniLab/unisim。
- `unilab.training.multi_source`：解析 owner，校验固定实验及各来源语义，注入
  pickleable factories，验证实际观测/动作维度和顺序，生成四份配置 manifest。
- `uni_rl.algos.rsl_rl_validation`：运行 stock PPO，在完整更新后的 logger
  边界结束有界验收并保存 checkpoint。不复制 PPO 训练循环，不打断 step/reset。
- `unilab.training.evaluation`：严格 Sim2Sim/维度校验和无渲染指标评估。
- `unilab.training.validation`：薄验收编排、资源预检、版本关联和结果记录。

各来源 action/obs/reward/done/terminal observation 使用共享内存；控制 pipe
不传大数组。每步先向全部来源提交，再等全部完成，按配置顺序发布双缓冲结果。
发生超时、崩溃、协议序号错误、非法 shape 或非有限核心输出时整体失败，
不返回部分 rollout、不重试非幂等操作，并清理来源进程组及嵌套 Isaac worker。

`source/<backend>/...` 保留来源指标；`source_statistics` 提供累计 transition、
step/reset 延迟和来源 PID/RSS。并发提交不保证单 GPU 上物理计算完全重叠。
环境已有的数字型 `info["timing"]` 会保留到 `timing/<字段>/last` 和 `/total`；
其中 `reset_done_reset_call_ms`、`reset_done_count` 记录 PPO step 内部的自动
reset，不能用仅统计显式 RPC reset 的计数替代。字段单位沿用环境原始名称。

### 逐来源采样计时

多来源 owner 默认开启 `training.source_timing.enabled=true`；单后端默认关闭。
计时由 `uni_rl.algos.rsl_rl_source_timing` 观察现有 PPO logger hook 和公开
`source_statistics`，不修改 step/reset 协议、采样顺序或 PPO 更新。
在每次完整更新后刷新运行目录中的 `source_timing.jsonl`，不覆盖已有文件；
中途失败的半轮 rollout 不会作为完整记录发布。运行中的旧进程不会自动加载
新代码，需经确认从 checkpoint 续训后才有这些记录，不能回填历史来源耗时。

每条 JSONL 记录包含四个来源各自的 24 个 `step_seconds`、总和、均值、
P50/P95/最大值、transition 数、PID/RSS 和 `environment_timing_totals`。
P95 使用线性插值。一次 step 是该来源全部 2000 个环境前进一个控制步，
不是单个环境的耗时；一轮每源 48,000 transitions。

终端每轮增加 `Source sampling [iteration]` 行，显示每源 ms/step、P95 和
s/rollout。TensorBoard 路径为 `source/<name>/sampling/step_seconds_mean`、
`step_seconds_p95`、`step_seconds_sum` 等，值的单位仍是秒；环境分项位于
`source/<name>/timing/<原始字段>/rollout_sum`。其中 `_ms` 是该轮毫秒总和，
`reset_done_count` 是该轮自动 reset 的环境总数，不是 reset RPC 次数。

`step_seconds` 是来源 worker 的 `env.step()` 墙钟耗时，包含物理推进、状态
更新、奖励和内部自动 reset，但不含外层共享内存发布及 learner 的策略推理。
backend 子计时可能互相嵌套、包含等待或不含 GPU 后续同步，不能直接相加，
也不能视作隔离测得的纯 GPU kernel 时间。此功能不额外调用 CUDA synchronize。

四来源并发提交，所以四个来源的 rollout 耗时不能相加当作总采样时间。
`max_source_step_seconds_sum` 是每步最长 worker 耗时的总和，不是实测屏障
等待时间；`collection_minus_max_source_step_seconds` 是与总采样时间的差，
包括推理、传输、拷贝、调度、日志等未归因部分，不能直接称作 IPC 耗时。
`slowest_steps` 表示该来源在本轮多少步具有最长 worker 耗时，并列时均计数。

## 入口和运行记录

```bash
uv run --no-sync train --algo ppo --task g1_walk_flat --sim multisim
```

此命令只选择训练 owner，不把 `multisim` 注册成物理 backend。首次验收优先用
下一节的有界入口，不直接启动不限时间的训练。多来源只允许单 learner、
GPU 0、无渲染，拒绝 torchrun、多 rank、改配额及共享任务语义的漂移。

`run_config.json` 中保留顶层严格 Sim2Sim snapshot，并在 `run.multi_source`
记录来源顺序、固定 slice、配额、seed、版本、四份解析后配置和各自 snapshot。
基准 learner seed 为 1；默认来源 seed 依次为 2、3、4、5。

独立评估同一个 checkpoint：

```bash
uv run --no-sync eval --algo ppo --task g1_walk_flat --sim isaacgym \
  --profile multisim --metrics algo.load_run=/absolute/run/validation_final.pt
```

将 `--sim` 分别换成 `isaacsim`、`motrix`、`genesis`。`--metrics` 即使
render mode 为 none 也实际执行策略和环境 step；不走 viewer 的“跳过播放”分支。
评估先严格检查训练配置与 checkpoint，再关闭评估观测噪声、物理 DR；
保留公共任务 reset 分布，不改 Sim2Sim DENYLIST。

每后端 seeds `[101,102,103,104]`，每 seed 25 envs，每个环境仅记录首个完整
episode；每后端 100 个。指标按 episode 有效步先聚合，再报告分布：episode
length、非 timeout 终止率、20 秒存活率、vx MAE、|vy|、|wz|。未收敛 checkpoint
指标差并不代表评估链路实现失败。

指标沿用公共 G1 传感器定义：vx/vy 来自 pelvis IMU site，wz 来自
`torso_gyro` 的 torso-local 坐标系，**不是 pelvis/root 的角速度**。报告逐项
记录该来源和坐标系；不能把两者混称为 root-frame 速度。

### MuJoCo Sim2Sim 与录像

`mujoco_multisim` 评估 owner 继承同一公共任务配置；MuJoCo 不是训练的四个
来源之一，此入口用于独立的未见引擎迁移评估，不改变四后端 gate 的汇总集合。
`--load-run` 只接受运行目录名或 `-1`；绝对 checkpoint 路径使用上例的
`algo.load_run`。指定运行名与轮次的本机示例：

```bash
source .tmp/multisim-gpu.env
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 uv run --no-sync eval \
  --algo ppo --task g1_walk_flat --sim mujoco --profile multisim --metrics \
  --load-run 20260912_205814_multisim_10000 algo.checkpoint=9999 \
  training.device=cpu training.evaluation.record_video=true \
  training.cam_distance=3.5 training.cam_elevation=-15
```

默认仍评估 4 seeds × 25 envs；策略与 MuJoCo 仿真在 CPU，EGL 渲染可使用 GPU。
录像只缓存第一个 seed 的环境 0 的首个 episode，直到首次终止/超时，
不重试或重新选片。每个控制步执行前缓存一次，避免混入自动 reset 后的画面。
全部指标采样后才离线渲染 1280×720 跟随镜头 MP4；0.02 s 控制周期对应
50 FPS，完整 episode 为 20 秒。报告的 `metadata.video` 记录轨迹、镜头、
帧数、FPS 和文件路径；`render_mode=none` 表示采样时不渲染。
已有同名视频会报错，不会覆盖；不可渲染或没有生成非空视频也会报错。
默认自动创建唯一时间戳目录，位于 checkpoint 的 `evaluation/` 下；可用
`training.evaluation.output_dir` 指定新目录。未开启录像时不会创建 renderer。

## Gate 驱动

使用一个新的绝对输出目录。每个 stage 开始前都检查 GPU driver、现有计算
任务和 editable 来源。不终止别人的任务，不通过缩容/删源/关闭 DR 降级。
失败或跳过的真实后端测试不算通过；依赖 gate 必须对应当前三仓 source hash。

```bash
OUT="$(pwd)/artifacts/g1-multisim-acceptance"
uv run --no-sync -m unilab.training.validation --gate preflight --output-root "$OUT"
uv run --no-sync -m unilab.training.validation --gate physics --output-root "$OUT"
uv run --no-sync -m unilab.training.validation --gate single --output-root "$OUT"
uv run --no-sync -m unilab.training.validation --gate capacity --output-root "$OUT"
uv run --no-sync -m unilab.training.validation --gate soak --output-root "$OUT"
uv run --no-sync -m unilab.training.validation --gate evaluate --output-root "$OUT"
```

| Gate | 实际操作与证据 |
| --- | --- |
| G1 | 四个真实 G1 effect pytest 子进程，以及四后端 native 参数/惯量读回；两类 JUnit 均必须有执行用例且零 skip/failure/error |
| G2/G3 | CPU spawn、故障注入和 stock PPO 集成测试，由三仓测试 gate 执行 |
| G4 | 四后端分别 2000 env，DR reset 后 step 1000 个控制步，再正常 close |
| G5 | 四源同时各 2000 + learner，24-step rollout 和完整 PPO update |
| G6 | 与 G5 相同规模，预热 2 次完整更新后持续正常 PPO 至至少 3600 s |
| G7 | G6 的同一个 checkpoint，四后端依次执行各 100 episodes，并汇总总体报告 |

训练四源采样永远是并发提交、同步屏障；只有独立 smoke/评估按 backend
先后运行来释放显存。G6 在完整更新后安全结束、正常保存 `validation_final.pt`，
不用外部 kill 充当完成。`updates.jsonl` 保存每轮样本数、每源配额、采样/更新
耗时、来源 step/reset 累计延迟、RSS、可获取的 GPU 内存及吞吐；不从估算中
生成容量结论。

输出目录的 `preflight-<stage>.json`、`G1.json`/`G4.json`/`G5.json`/`G6.json`/
`G7.json` 和各子目录日志是验收证据。已有 gate 结果不覆盖；修改源码后不能
复用旧 gate 冒充当前 revision 通过。G1 或容量失败时停在该 gate 修复。

## 本地代码 gate

```bash
UV_NO_SYNC=1 make check
UV_NO_SYNC=1 make test
```

两个 sibling 仓库分别按自身 AGENTS 运行 lint/format/type/tests；共享 `.venv`
时用 `UV_PROJECT_ENVIRONMENT` 指向 UniLab `.venv`，仍保持 `UV_NO_SYNC=1`。
创建或更新 PR 前还需要 `make test-all`；本次不会自动创建 PR、commit、push
或发布包。默认测试中的可选 GPU skip 不替代上述显式 opt-in G1/G4–G7。

## 当前验收限制

首次预检的 NVIDIA 内核/NVML 版本不匹配已在用户修复后解除。2026-09-12
复验时两者均为 580.178.04，RTX 3090 24 GB 的 CUDA 实际运算通过。此次
仅配置独立 Python 环境，没有修改驱动、重启机器或终止用户的计算任务。
GPU 环境恢复不代表 G1–G7 已通过；物理 gate 失败时仍不得推进容量或训练。

本次四后端的真实 G1 mass/KP/KD 与接触效果复测均通过。IsaacGym 修复使用
GPU PhysX + CPU 状态张量，避开 Preview 4 在 GPU tensor pipeline 上运行时
mass 写入破坏 root reset 的问题；没有改成 CPU 物理、关闭 DR 或额外 step。
该传输方式对全部 IsaacGym 入口生效，2000-env 的容量与吞吐影响尚未测量。
Genesis 修复 mass 写入与 forward 缓存更新顺序；IsaacSim 修复关闭时 STOP
callback 的渲染循环。后续复验记录见
`docs/validation/g1_multisim_cuda_2026-09-12.md`。

此外，Motrix 的四环境真实 G1 DR/接触效果测试已经通过，但强制 native
读回测试无法取得当前 Motrix 0.8.2 的 live per-environment inertia。
读取 model-only `LowSceneModel.link_inertias` 不能证明 solver 的惯量不变。
这是**缺少验证能力，不是观测到惯量变化**；测试明确失败，不能用 nominal
缓存替代读回，也不能用通过的效果测试抵消它。需要可验证的 live inertia
读取能力及测试绑定；即使 GPU 驱动恢复，G1 也必须重新完整通过后才能继续。
