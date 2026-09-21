# 四源自适应采集比例接口

本文保留 2026-09-20 接口交付时的验收边界；下列操作命令已适配 UniDR
内置的 `vendor/unilab_rl` 和 `vendor/unisim`。请先完成[当前 README](../README_zh.md)
中的安装与后端环境准备，再从仓库根目录运行。历史 `logs/` 记录未随源码发布。

2026-09-20 本地实现。用户随后将范围收窄为接口开发：**没有启动新的训练实验**。
实现基于已验证的四源共同配置，保持 G1 flip、动作缩放、奖励、终止、自碰撞、
观测、模型和 PPO 参数。UniLab 注入任务工厂；`uni_rl` 拥有采集、probe、调度和恢复。
结构决策见 runtime 的[自适应配额 ADR](../vendor/unilab_rl/docs/adr/0003-adaptive-source-quotas.md)。

## 开关与入口

单卡配置是 `task=g1_flip_tracking/unidr_adaptive`；四卡配置是
`task=g1_flip_tracking/unidr_adaptive_four_gpu`。
`unidr.adaptive.enabled=true/false` 控制采集比例是否自适应。
关闭开关时仍执行相同 probe 和训练环境重置，比例固定各 25%，用于控制变量比较。
旧 `unidr_single_gpu` / `unidr_four_gpu` 入口仍保持原先无 probe 的固定采集行为。

只查看完整配置，不创建环境或启动训练：

```bash
uv run --no-sync python scripts/train_unidr.py \
  task=g1_flip_tracking/unidr_adaptive --cfg job --resolve
```

以下是留待用户决定运行的训练命令，**本次未执行**：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
uv run --no-sync python scripts/train_unidr.py \
  task=g1_flip_tracking/unidr_adaptive \
  algo.num_envs=1024 algo.max_iterations=5000 \
  training.log_dir=/absolute/new/adaptive-run
```

四卡配置中 Isaac Sim/Gym/Genesis 对应 GPU 0/1/2，Motrix 物理为 CPU，
唯一中央策略推理与 PPO learner 为 GPU 3；单卡全部 GPU 工作在 GPU 0。
此机器只有 RTX3090，四卡仅验证配置映射。

## 配额、采集与归一化

每源常驻池容量不变，每轮分配总计 96 个整池控制步。初始是 `[24,24,24,24]`，
例如 `[25,24,24,23]` 表示多采一个来源、少采另一个来源，**没有丢弃或补造样本**。
每源 1024 时仍是 98,304 transitions，5 epochs × 4 minibatches，20 次优化。

一个 wave 只推进尚有配额的来源。提前采满的来源暂停物理；中央只对实际推进的
post-step observations 更新一次 normalizer。每源连续轨迹先计算原生 GAE，再合并
全 batch、统一归一化优势。纯 timeout 使用该步更新后的 normalizer 计算 final value；
真正终止及双标记不 bootstrap；窗口尾使用所有来源采满后的 normalizer。
原始 rollout 保留 observation dict、动作、reward、old value/log-prob/distribution、
终止信息、final observation、bootstrap 和来源/env/episode/segment/window 版本。
每源使用完整窗口 wave 版本序列的前 `quota_steps` 项。

## 独立 probe 与调度

每完成 100 轮，冻结当前 actor/critic 和 normalizer，在四个训练后端各重置一次
完整环境池到相同 frame 0、seed=1、零附加 DR，运行 224 控制步（4.48 秒）。
每个环境只有首次 episode 算一次机会，后续 autoreset 不重复评分。

| 指标 | 固定定义与归一化 |
| --- | --- |
| E | 受跟踪身体的平均世界位置误差，逐步上限 0.5 m；提前终止后剩余步按 0.5 m 补齐；时域平均再除以 0.5 |
| S | 整个 224 步未触发原生 terminated/truncated 的环境比例 |
| R | 首 episode 原始奖励求和后取环境均值；提前终止后补零；固定用 `[0,38.08]` 映射并裁剪至 `[0,1]` |

38.08 是共同任务所有正奖励速率上界之和 `8.5 × 0.02 × 224`。
负回报仍保存原值，只在归一化时裁剪。尺度在运行期间固定，不做来源内 min/max。
E 的几何误差在 autoreset 前采集。Probe 完成后恢复环境 RNG/训练计数并全池 reset，
开始新的训练 episode；不宣称恢复 probe 前物理状态。
Probe 数据不进入 rollout、normalizer 或 optimizer。

**S 是训练后端的原生存活判据，并非 MuJoCo“动作后 5 秒不摔倒”的留出判据。**
MuJoCo 完全不参与在线 probe、配比或 checkpoint 选择。
相较历史无 probe 的联合实验，新增的定期全池 reset 是实验条件变化；公平对照应
使用本入口 `enabled=false`，保留相同 probe/reset。

E/S/R 分别 EMA（alpha=0.2，首次直接初始化），困难度为
`0.50*E + 0.25*(1-S) + 0.25*(1-R)`，比较基准固定四源等权。
噪声门槛默认 0.02，增益 0.10，困难来源增配、容易来源减配。
连续目标通过有界单纯形配平，整数配额通过凸整数分配求解：
总和严格为 96，目标与实际比例均保持 `[0.10,0.70]`，每次变化至多 0.02。
因 `2/96 > 0.02`，实际每源一次最多改变一步；实际范围为至少 10/96，
最多 66/96（还要给其他三源各保留 10 步）。小于整数粒度的变化可能保持不变。

指标权重策略独立：默认开启，四源 EMA 全部连续 3 次达到
E≤0.20、S≥0.95、R≥0.80 后，每次把 0.01 失败率权重转给 E，失败率权重最低 0.05。
这些门槛、窗口和速度均在 YAML 中配置。调度器遇到无效/缺测指标原子保持状态；
物理服务返回非有限数据、缺字段、重复或错位响应则中止，不能缺源继续或重试 step。

## 日志和恢复

`synchronous_metrics.jsonl` 记录每轮实际分源预算、当前目标/实际比例、下一轮比例、
原始/归一化 probe 指标、EMA、权重、难度、调度状态和耗时。
TensorBoard 新增 `Source/<source>/actual_ratio`、累计预算及 `Probe/<source>/*`。
`sources_manifest.json` 将模式、尺度、probe 和调度参数计入配置指纹。

完整 iteration 边界才保存 checkpoint，且必须已完成该轮到期的 probe/调度。
除原 PPO/optimizer/normalizer/RNG 外，保存 EMA/权重/比例/计数及压缩配额历史。
恢复时重新验证每轮配额边界、总量、变化速度、分源累计样本和精确 wave 数；
恢复 learner 后重置全部物理环境进入新 generation，不复用异常窗口。
原固定模式 checkpoint 不能当作自适应实验的无缝续训 checkpoint。

## 本地验收与后续边界

本次验证包括 fake 四进程真实 PPO 更新、与原均匀 collector 的逐步比较、
不均匀配额手算 GAE、20 次优化、完整 epochs、精确整数 normalizer 计数、
独立 probe、坏数据回收、调度方向/噪声/边界/保存恢复及配置映射。

真实接口验证使用四后端各 2 环境，采集 `[25,24,24,23]`、192 条样本，
然后执行四源独立 224 步 probe；optimizer 从始至终没有执行一次更新。
结果与精确复现命令见
`logs/adaptive-implementation-20260920/DELIVERY.md`。
完整测试执行于 RSL-RL 5.0.1 和 5.5.0；没有重装系统或 vendor 依赖。

每源 1024 的自适应容量验证、真实后端自适应 PPO 训练、四卡实机、训练效果和
最终 sim2sim 尚未验证。历史固定预算的专用报告/holdout 审计器尚未扩展到
新的可变 wave checkpoint；不得用它们的固定等配额假设审计自适应运行。
