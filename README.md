# UniDR：9-16-resume 历史基线与地面修复

本分支恢复 **2026 年 9 月 16 日 16:56:31（北京时间）** 保存的 G1 flip
训练源码，随后按实验要求调整单源规模并修复 Isaac Sim 地面。
纯历史恢复点为 `72b76f255e4a37ea7388895bb806a921f0758178`；恢复工作在
9 月 21 日执行，提交时间保留真实重建时间。
原始实验修改没有单独的 Git commit，因此本分支由三仓基线、当时的工作树补丁和
源码 ZIP 重建，不是伪造的历史提交，也不加载旧模型续训。

**这是尚未实现仿真器比例自适应的代码版本。** 来源调度器、动态配额采集器、独立
probe、自适应配置与对应 runner/IPC 扩展都不在此版本。PPO 原有的 adaptive KL
学习率调节仍然存在。当前分支已移植 Isaac Sim 共享地面修复及物理错误检测。

## 当前分支的训练设置

| 项目 | 四个单后端：各自独立训练 | 四后端联合：保持历史配置 |
| --- | --- | --- |
| 环境数量 | 每次 4096 | 每源 1024，总计 4096 |
| 默认轮数 | 5000 | 10000；下方比较实验命令仍显式指定 5000 |
| 每轮采集 | 24 步，98,304 transitions | 同样 98,304 transitions，各源固定 25% |
| PPO 更新 | 5 epochs × 4 minibatches，20 次/轮 | 相同 |
| 5000 轮总预算 | 每个模型 491,520,000 transitions、100,000 次优化 | 唯一模型相同总预算 |
| 5000 轮最终 checkpoint | `model_4999.pt` | `model_4999.pt` |

单源 owner 的 `*_comparison.yaml`、准备入口和队列默认值统一为 4096/5000。
奖励、动作缩放、归一化、PPO 超参数、终止和 bootstrap 语义保持历史基线；
联合 owner、runtime 和设备映射没有修改。共享地面修复同时作用于单源与联合中的
Isaac Sim，避免其复制环境时重复复制无限碰撞平面。
本次改动没有启动新的训练，也没有重启或修改正在运行的实验。

## 源码与完整性

为保留当时的相对路径和 owner 边界，三个仓库按历史运行方式相邻放置：

```text
UniLab/                  # task、Hydra、环境适配、sim2sim
unilab_rl/               # PPO、rollout、GAE、learner、同步 IPC
unisim-unidr-backends/   # 四个仿真器的物理适配
history/2026-09-16/      # 原始归档、补丁、哈希及本次验证记录
```

| 原仓 | 基线 commit |
| --- | --- |
| UniLab | `b4e6b58fe0861a435fd19c0f0206bd84f4427a9c` |
| uni_rl | `79418e0cbff7b95fe6e49709b194454701ba20fa` |
| UniSim | `4270aa81d868744980db90dac6dd959d3f542f50` |

纯历史恢复提交包含 1228 个文件；当时 ZIP 中的 644 个源码/配置文件逐字节匹配。
当前改动位于该提交之后；`history/2026-09-16/` 中的归档与证据仍原样保留。
所有当时未追踪的 `src/`、`scripts/`、`examples/` 文件均在 ZIP 中。
有 7 个当时未追踪的文档/测试文件未被归档，缺失清单见
[重建清单](history/2026-09-16/reconstruction.json)，不能宣称完整工作树无遗漏。
三个 owner 原有 Apache-2.0 LICENSE 均保留。

原 ZIP SHA-256：`fe4f3a15c91069eb1d229fc1a3adbfedafe020827525f5198b51a112df6dbe3d`。
根 README、后续验证说明和验证环境清单不属于历史代码。
三 owner 现在位于同一个 Git 分支，运行时 Git 信息会记录这个重建提交；原始三仓
SHA 和 dirty 状态保存在 [sources.json](history/2026-09-16/sources.json)。

## 历史基线的单后端与四后端训练

以下表格记录纯历史恢复点，单源当前规模见上节。

四个单后端分别训练自己的模型，历史队列顺序为 **Motrix → Isaac Sim → Isaac Gym
→ Genesis**。四源联合训练通过中央 actor/critic 推理、分发 action、等待四源完成
同一控制步，沿环境列汇合数据，最后更新唯一 PPO 模型。

| 项目 | 单后端：每一个模型 | 四后端联合 |
| --- | --- | --- |
| 来源 | 仅一个 simulator | Isaac Sim / Isaac Gym / Genesis / Motrix |
| 环境数量 | 1024 | 每源 1024，总计 4096 |
| 来源比例 | 该来源 100% | 固定各 25%，无来源自适应 |
| 每轮采集 | 24 步，24,576 transitions | 每源 24 步，总计 98,304 transitions |
| 每轮优化 | 5 epochs × 4 minibatches，20 次 | 相同，20 次 |
| 归档 owner 默认轮数 | 20,000 | 10,000 |
| 后来完成的比较实验轮数 | 20,000 | 显式配置 5,000 |
| 比较实验总采样量 | 每模型 491,520,000 | 总计 491,520,000 |
| 比较实验每来源采样量 | 491,520,000 | 122,880,000 |
| 比较实验总优化次数 | 每模型 400,000 | 100,000 |
| 比较实验最终 checkpoint | 每源 `model_19999.pt` | `model_4999.pt` |

联合与每个单源模型的总采样量相等，但每个来源的数据量和优化次数不同。
四个单源的训练配置与后来旧联合实验的对应来源 `env`、PPO 参数和 78 个资产指纹
全部匹配。配置证据保存在 [single-experiments.json](history/2026-09-16/single-experiments.json)。

归档时联合核心已经存在，但 `unidr_comparison.yaml`、`joint_comparison.py`、
`run_joint_comparison.sh` 尚未加入。本分支不会补入这些后来的文件。
旧联合实验目录虽然含 `20260916`，实际训练发生于 **9 月 18 日**。

共同训练设置：

- G1 29 DoF，actor/critic 输入 160/286，29 维动作；两个网络均为
  `512 → 256 → 128`，ELU，经验观测归一化开启。
- 初始学习率 0.001，`gamma=0.99`、`lam=0.95`、clip 0.2、entropy 0.005、
  value loss 1.0、gradient norm 1.0、`desired_kl=0.01`，原生 adaptive KL。
- learner seed 1；Sim/Gym/Genesis/Motrix 环境 seed 2/3/4/5。
- 物理步长 0.005 秒、控制步长 0.02 秒；四后端自碰撞均关闭。
- 使用成功 Motrix 配置的分关节动作缩放与奖励。root 位置/朝向奖励权重为
  0.5/0.5，body 位置/朝向为 2.0/1.5，线/角速度为 1.0/1.0，末端 Z 为 2.0；
  action rate 惩罚 −0.005、joint limit −10、undesired body height −0.1。
- 未配置物理参数 DR；motion pose/velocity/joint perturbation 为零。
  联合训练中的动力学差异来自四个仿真器本身，各物理求解器并不相同。

终止与 bootstrap：anchor 或手腕/脚踝的高度跟踪误差超过 0.5 米会终止；监测身体
的世界高度低于 0.05 米也会终止，此项是高度判断，不是接触力。方向终止阈值
`1e9`，实际关闭。episode 上限 10 秒：真正终止优先，包括双标记，不 bootstrap；
纯 timeout 使用 final observation bootstrap；正常窗口尾也 bootstrap。

参考片段为 `flip_360_001__A304.npz`，从开头开始，片段结束会重设 root/joint
状态开始循环，不因此标记 done。训练没有额外“动作后站立 5 秒”要求；这属于另外的
sim2sim 评测协议。MuJoCo 仅用于留出测试。

## 环境与运行

当前服务器已经为本分支创建独立 `.venv`，没有修改正在训练使用的环境。
RSL-RL 实际使用 5.0.1、Torch 2.8.0+cu128。历史根锁文件与 runtime 锁文件均原样
保存；原始配置没有绑定本分支的 editable 路径，runtime 锁还使用 RSL 5.5.0，因此
安装后使用 `uv run --no-sync`。下面的清单固定的是**本次验证环境**，不是
声称完整恢复了 9 月 16 日的虚拟环境或外部 Isaac SDK。

在分支根目录，新安装时执行：

```bash
uv venv --python /usr/bin/python3.10 .venv
uv pip install --python .venv/bin/python --no-deps \
  --index https://download.pytorch.org/whl/cu128 --index https://pypi.org/simple \
  -r history/2026-09-16/validation-requirements.txt
uv pip install --python .venv/bin/python --no-deps \
  -e UniLab -e unilab_rl -e unisim-unidr-backends
```

这套安装对应本机 Linux/Python 3.10。Isaac SDK 继续使用独立 vendor worker 环境。
机器人 mesh、texture、动作文件与 checkpoint 不进入 Git；G1 资产在本机已按历史
指纹校验，其他机器需通过 UniLab asset hub 取得相应资产。
历史配置排除了 torchvision，因此 `uv pip check` 会报告 RSL 的该项声明缺失；
本分支未擅自补装或修改历史配置。

从分支根目录设置本次 shell 的运行环境：

```bash
export UNIDR_ARCHIVE_ROOT="$PWD"
export UV_PROJECT_ENVIRONMENT="$UNIDR_ARCHIVE_ROOT/.venv"
export UV_NO_SYNC=1
export UNILAB_LOCAL_UNISIM="$UNIDR_ARCHIVE_ROOT/unisim-unidr-backends"
unset PYTHONPATH
mkdir -p "$UNIDR_ARCHIVE_ROOT/experiments"
cd UniLab
```

以下为当前分支的训练用法，本次修改没有执行这些训练。必须使用新的输出目录。
四种单后端顺序训练，并在各自结束后执行原版 MuJoCo 留出录制：

```bash
bash scripts/run_single_comparison.sh "$UNIDR_ARCHIVE_ROOT/experiments/singles-new" 4096 5000
```

仅训练某个单后端（示例 Motrix，其他选择 `isaacsim`、`isaacgym`、`genesis`）：

```bash
uv run --no-sync python -m unilab.training.single_comparison motrix \
  "$UNIDR_ARCHIVE_ROOT/experiments/motrix-new" --num-envs 4096 --iterations 5000
uv run --no-sync train --algo ppo --task g1_flip_tracking --sim motrix \
  --profile comparison algo.num_envs=4096 algo.max_iterations=5000 \
  training.device=cuda:0 training.log_dir="$UNIDR_ARCHIVE_ROOT/experiments/motrix-new"
```

用恢复版本已有入口，显式配置四源各 1024、单卡联合 5000 轮：

```bash
uv run --no-sync python scripts/train_unidr.py \
  task=g1_flip_tracking/unidr_single_gpu \
  algo.num_envs=1024 algo.max_iterations=5000 \
  algo.resume=false algo.resume_path=null \
  training.log_dir="$UNIDR_ARCHIVE_ROOT/experiments/joint-new/train"
```

单卡模式三个 GPU 后端与唯一 learner 使用 GPU 0，Motrix 物理使用 CPU。
四卡模式改用 `task=g1_flip_tracking/unidr_four_gpu`：Sim/Gym/Genesis 分别使用
GPU 0/1/2，learner 使用 GPU 3，Motrix CPU。四卡配置通过映射检查，未做四卡实机验证。
所有模式都保留各自原生 worker 和 Python 环境。

## 验证和适用边界

历史恢复时执行了源码哈希核验、四个单源和两种联合布局的配置检查、隔离环境导入检查、
PPO fake smoke，以及三个 owner 的格式、类型和测试检查。
实际结果与未通过项目见 [VALIDATION.md](history/2026-09-16/VALIDATION.md)。
为保留历史字节，使用 `ruff format --check` / `ruff check` 等只读门禁，没有执行
会改写历史源文件的格式修复。既有类型/测试问题如实保留。

当前增量的地面修复、验证命令和结果见
[分支改动验证](docs/9-16-resume-ground-fix.md)。本轮通过纯 USD 结构测试和 CPU
回归验证，未在此分支重新启动真实 Isaac Sim 4096 环境容量测试；既有真实引擎
证据与本轮测试分开记录。相同 seed 不保证逐位相同的 GPU 轨迹或 sim2sim 成功率。
没有加入后来的来源自适应、成功率统计、录像工具或五秒站立评测代码。
