# CUDA 环境恢复与 G1 物理复验

日期：2026-09-12。结论：**CUDA 环境完成；四后端 G1 物理效果通过；完整 G1
仍未通过，不能标记训练就绪。** 本文补充而不覆盖同日首次 CPU/阻塞记录。

本轮没有启动 G4–G7、容量测试、60 分钟或正式 24 小时训练，没有生成真实
短训 checkpoint。固定实验仍是四源各 2000、25% × 4；四环境物理探针不是
缩容后的训练。没有修改驱动、SDK、已有 conda 环境、`uv.lock` 或
`docs/research/`，也没有提交、建分支、推送或发布。

## CUDA 环境

| 项目 | 已验证值 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3090，24 GB，compute capability 8.6 |
| 驱动 | 已加载内核与 NVML 均为 580.178.04 |
| 独立环境 | `/home/wsm/wang-sm/UniLab/.tmp/multisim-gpu` |
| Python / Torch | 3.11.16 / `2.8.0+cu128` |
| CUDA / cuDNN | 12.8 / 91002 |
| editable | UniLab、`../unilab_rl`、`../unisim` 开发源码 |
| 计算验证 | GPU matmul 对照 CPU、卷积、backward、Adam update 均通过，结果有限 |

按原 lock 安装 mujoco/motrix/genesis extras。六个下载缓慢的 NVIDIA wheel
通过镜像取得，并按 lock 中的精确 SHA256 校验；没有换包版本或改锁文件。
`.venv` 现在指向新 GPU 副本，`.venv-cpu`、未完成的旧环境和下载缓存仍保留。

```bash
source /home/wsm/wang-sm/UniLab/.tmp/multisim-gpu.env
uv run --no-sync python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

本地配置设置 `UV_PROJECT_ENVIRONMENT`、`UV_NO_SYNC=1`、
`UNILAB_LOCAL_UNISIM`，并选择已有 Node 24 供 Pyright 使用。可复现安装流程
见 `docs/multisim_training.md`，依赖清单见证据目录中的 `dependencies-gpu.txt`
和 `cuda-runtime-requirements.txt`。

`uv pip check` **不是通过**：它报告 `rsl-rl-lib` 要求的 torchvision 未安装。
这是现有 `pyproject.toml` 的 `exclude-dependencies = ["torchvision"]`
决定，本轮没有擅自安装或更改该依赖策略；计算与现有 PPO 导入/测试可以执行。

## 物理复验

| 后端 | 真实 G1 mass/KP/KD、重复 reset、接触效果 | Native 参数/惯量读回 |
| --- | --- | --- |
| IsaacGym | 1 passed，零 skip | 1 passed，零 skip |
| IsaacSim | 1 passed，零 skip | 1 passed，零 skip，进程正常退出 |
| Motrix | 1 passed，零 skip | 1 failed，缺少 live per-environment inertia getter |
| Genesis | 1 passed，零 skip | 1 passed，零 skip |

Isaac native 读回使用既有三环境、两关节专用 fixture，覆盖原生参数写入、
选中行隔离、绝对值重复写入和惯量不变；不声称这是 G1 全机器人逐 link 惯量
认证。四后端的真实 G1 效果测试均覆盖 29 个关节。Genesis/Motrix native
读回使用实际四环境 G1，Motrix 在首轮选中 mass/KP/KD 读回通过后，于必需
惯量断言失败；后续循环不能记作已完成。

正式驱动按照 IsaacGym → IsaacSim → Motrix → Genesis 执行，在 Motrix
native 测试处退出 1，生成 `final-gates/G1.json`，状态为 `failed`。
Genesis 的上述通过结果来自单独执行的最终复测；不能说失败的正式驱动也
执行了 Genesis。所有 native 运行依次进行，结束后无遗留 GPU compute 进程。

### 修复与测试边界

- IsaacGym：合并连续 root/DOF reset，setter 每次 simulate 前只提交一次，
  保留索引张量，禁止 reset 后刷新覆盖待提交状态。进一步发现 Preview 4
  GPU tensor pipeline 在运行时 mass 写入后丢失 root reset，即使质量没有
  改变也会把机器人置于地面下并产生巨大真实接触冲量。改用 SDK 支持的
  **GPU PhysX + CPU 状态张量**后，原效果测试与落地/离地接触通过。没有额外
  step、合成接触或改用 CPU solver；该传输方式对全部 IsaacGym 入口生效，
  吞吐成本留待实际容量 gate 测量。新增真实 G1 回归 3 passed、聚焦 CPU
  90 passed，诊断详情见 `isaacgym-debug/contact-resolution.md`。
- Genesis：先写入绝对 DR，再执行 `set_qpos` 重建 mass-dependent forward
  缓存，避免下一步仍使用前一次质量的 COM/spatial-inertia 缓存。固定 world
  link 初始 native mass 是 `gs.EPS` 而非 canonical 0；读回测试精确区分
  未选中行的初值与已写入行的 0，所有机器人质量、gains 和所有 link 的惯量
  不变断言仍保留。静态 world 不要求非零惯量，但变更其惯量仍会失败。
- IsaacSim：在 Kit 关闭 stage 前清除 IsaacLab callbacks 和 singleton，
  避免 STOP callback 再次进入 render loop。原来断言完成后仍持续挂起的
  native 进程现在正常退出，没有跳过清理或缩短物理测试。
- 公共 KD 效果探针：初始关节速度从 0.6 改为 1.2 rad/s，四后端统一。
  原速度下 IsaacSim 的 nominal/high-KD 左 ankle roll 都在首步停止，无法
  区分响应；1.2 下全部 29 关节可测。只改诊断激励，不改训练 reset 分布、
  KD 区间、步长、接触判据或数值阈值。
- Native 测试启动复用 adapter 的 vendor runtime 发现和 library env；未
  安装 runtime 仍明确失败。新增父任务聚焦回归 18 passed。

Motrix 0.8.2 的 Link、SceneData.low/LowData 没有 live inertia 读取接口；
`LowSceneModel.link_inertias` 是 model-only 数据，不能替代逐环境 solver
读回。这是**未能验证，不是检测到惯量变化**。需要 SDK 提供可验证的接口并
添加 assertion-only 测试绑定；不能拿 nominal 缓存、效果通过或跳过来抵消。

## 精确复验命令

以下命令先加载前述本地环境，分别在对应仓库目录运行。每次正式驱动使用
新的输出目录；已有 gate 不覆盖。真实 SDK 运行不要并发争用这张 GPU。

```bash
cd /home/wsm/wang-sm/UniLab
uv run --no-sync -m unilab.training.validation --gate physics \
  --output-root /home/wsm/wang-sm/UniLab/val/g1-multisim-cuda-20260912/final-gates

UNILAB_RUN_MULTISIM_PHYSICS=1 uv run --no-sync pytest \
  tests/backends/test_g1_multisim_physics.py -q -k genesis -m ''

cd /home/wsm/wang-sm/unisim
UNILAB_RUN_MULTISIM_PHYSICS=1 \
UNILAB_G1_SCENE=/home/wsm/wang-sm/UniLab/src/unilab/assets/robots/g1/scene_flat.xml \
  uv run --no-sync pytest tests/test_multisim_native_readback.py -q -k genesis -m ''
```

可独立将第一个 pytest 的 `-k` 换为任何一个后端；Isaac native 命令为
`UNISIM_ISAAC_READBACK=1 uv run --no-sync pytest tests/test_isaac_native_readback.py -q -k isaacgym -m ''`
或 `-k isaacsim`。正式驱动为每个用例保存日志和 JUnit，严格拒绝 skip。

## 代码 Gate

| 工作目录 / 命令 | 本轮结果 |
| --- | --- |
| UniLab / `UV_NO_SYNC=1 make check` | 通过；mypy 142 源文件、Pyright 0 errors，缺少可选 drake 的 1 warning |
| UniLab / `UV_NO_SYNC=1 make test` | 1694 passed、30 skipped、850 deselected、1 xfailed，62.36 s |
| unisim / `UV_NO_SYNC=1 make check` | Ruff 通过；344 passed、19 skipped、**1 failed**，9.56 s |
| unisim / 本轮 13 个 Python 文件 `ruff format --check` | 通过；Genesis adapter 保留最小差异，不扩大其基线格式修改 |
| 三仓 / `git diff --check` | 通过 |

unisim 唯一失败是未修改的
`../unisim/tests/test_debug_overlay.py` 中的
`TestOnFrameEndToEnd::test_on_frame_paints_video_frames`：
回调索引 `[0,1]` 已通过，但生成的 GIF 解码为 1 帧而断言要求 2 帧，红通道
最小值仍为 255。相关测试、MuJoCo playback 和 render_many 对 HEAD 无差异。
本轮没有修改无关视频编码或测试断言；不把整仓 gate 报为通过。

unilab_rl 本轮未改动，此前 CPU gate 为 436 passed，不冒充 CUDA 下重新跑过。
无 PR，因此未执行 `make test-all`。可选测试的 skip 不替代必需物理 gate。

## 文件与证据

本轮生产修复位于 sibling unisim 的 `backend/isaacgym/worker.py`、
`backend/isaacsim/worker.py`、`backend/genesis/backend.py`；新增对应 reset、
初始化、shutdown、live readback 和 native G1 回归测试。UniLab 仅调整
`tests/backends/test_g1_multisim_physics.py` 的 KD 探针并更新说明/本报告。
unisim `CHANGELOG.md` 和 `docs/isaac-reset-contract.md` 记录行为变化。

完整证据目录：`/home/wsm/wang-sm/UniLab/val/g1-multisim-cuda-20260912/`。
其中 `gpu-compute.json` 记录实际计算，`attempt-02/` 记录各单独复测，
`isaacgym-debug/` 保存故障的 red/green 及原生轨迹，`final-gates/` 保存正式
G1 失败和三仓源码指纹，`readiness.json` 汇总当前结论。

下一步先补齐 Motrix 实时惯量验证能力并修复必需 native 测试，再完整重跑
G1。只有通过后才能进入各后端 2000-env、四源容量、60 分钟训练与评估。
