# 9-16-resume：单源预算与 Isaac Sim 地面修复

本次增量基于纯历史恢复提交 `72b76f255e4a37ea7388895bb806a921f0758178`。
原始三仓 SHA、Apache-2.0 LICENSE、锁文件与归档完整性见
[历史恢复记录](../history/2026-09-16/VALIDATION.md)。历史证据目录没有改写。

## 修改范围

- UniLab 的四个 `g1_flip_tracking/*_comparison.yaml`：各 4096 环境、5000 轮。
  `training/single_comparison.py` 和 `scripts/run_single_comparison.sh` 默认值同步。
  队列顺序仍为 Motrix、Isaac Sim、Isaac Gym、Genesis；支持显式小预算覆盖。
- UniSim 的 `backend/isaacsim/worker.py`：共享导入的静态水平地面。
  `backend/subprocess_ipc/backend.py`：已知致命物理日志使采样失败并回收资源。
- 新增单源预算/队列测试、USD 地面测试、worker 物理错误测试；更新根 README。

联合训练的 owner、设备映射和 `uni_rl` 没有修改：每源 1024，固定各 25%，
默认 10000 轮，比较命令显式指定 5000 轮。共享后端修复也作用于联合 Isaac Sim。
单源仅改训练预算；奖励、动作、资产、归一化、终止、bootstrap 和 PPO 超参数不变。
没有加入来源比例自适应，也没有增加物理参数 DR。

每个单源模型每轮采集 `4096 × 24 = 98,304` 条 transition，执行
`5 × 4 = 20` 次优化；5000 轮共 491,520,000 条和 100,000 次优化，最终文件
为 `model_4999.pt`。这些是配置预算，本轮没有启动训练来消耗该预算。

## 地面问题及修复

MJCF 导入的 USD 同时包含机器人和无限地面；旧实现将地面及其运动学关节包装
随环境一同复制。4096 份无限平面使 PhysX pair buffer 耗尽，出现
`simulation will miss interactions`，但 SDK 仍可能返回正常 step 响应。

修复在初始化阶段从原始 USD 共享静态水平碰撞平面，保留变换和材质绑定，去除
仅用于地面的刚体/关节包装，并关闭机器人副本中的重复平面。共享地面加入 cloner
全局碰撞路径，机器人仍按环境隔离；用于显示的额外地面关闭碰撞。动态、倾斜或
动画平面直接拒绝，复杂地形不在此修复范围。没有扩大 PhysX buffer 掩盖问题。

worker stderr 保存在 `${XDG_CACHE_HOME:-~/.cache}/unisim/worker-logs/`，使用独立
读取句柄在请求前和响应后增量检查六类已知致命物理诊断。命中后拒绝继续 step，
关闭 worker 和共享内存，保留日志；不重试有状态 step。普通 warning 不触发失败。
日志没有轮转；异步延迟写出的诊断可能到下一请求才捕获。这没有新增 IPC 消息契约。

## 验证环境与命令

工作区：`/home/wsm/wang-sm/UniDR-9-16-resume`。使用隔离 `.venv`，实际 RSL-RL
5.0.1、Torch 2.8.0+cu128；runtime 锁中的 5.5.0 未同步覆盖实际消费版本。
没有安装或更新外部 Isaac SDK。CPU 检查使用：

```bash
export UNIDR_ARCHIVE_ROOT=/home/wsm/wang-sm/UniDR-9-16-resume
export UV_PROJECT_ENVIRONMENT="$UNIDR_ARCHIVE_ROOT/.venv" UV_NO_SYNC=1
export UNILAB_LOCAL_UNISIM="$UNIDR_ARCHIVE_ROOT/unisim-unidr-backends"
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1
export PATH=/home/wsm/.vscode-server/cli/servers/Stable-520fb30b2d3d324b4cb2342f6e88e2cd93751de1/server:$PATH
unset PYTHONPATH
cd "$UNIDR_ARCHIVE_ROOT/UniLab"
uv run --no-sync pytest -q tests/scripts/test_single_comparison_defaults.py tests/tasks/test_g1_flip_unidr.py tests/training/test_synchronous.py
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync mypy src/unilab
uv run --no-sync pyright
uv run --no-sync ruff check tests --select F401,F821,F811,F841 --output-format concise
uv run --no-sync pytest -m 'not slow' --cov=src/unilab --cov-report=term
bash -n scripts/run_single_comparison.sh
cd "$UNIDR_ARCHIVE_ROOT/unisim-unidr-backends"
uv run --no-sync pytest -q tests/test_worker_physics_errors.py tests/test_import_boundary.py
uv run --no-sync ruff format --check src/unisim/backend/isaacsim/worker.py src/unisim/backend/subprocess_ipc/backend.py tests/test_isaacsim_ground.py tests/test_worker_physics_errors.py
make check
```

UniLab 使用 `make check` 的只读等价格式/类型命令，以保留历史文件。
完整 non-slow 测试没有跳过已知失败用例。没有创建 PR，未执行额外 benchmark gate。

| 本轮检查 | 结果 |
| --- | --- |
| UniLab 配置、准备/队列、联合契约回归 | 40 passed，1 deselected |
| UniLab Ruff、格式、测试 lint、mypy、shell 语法 | 通过；mypy 142 个源文件 |
| UniLab Pyright | 历史基线同样的 10 errors、1 warning |
| UniLab 完整 non-slow 与覆盖率 | 1686 passed、27 skipped、849 deselected、1 inherited failure；71% |
| UniSim 物理错误及 import boundary | 24 passed |
| UniSim `make check` | Ruff 通过；313 passed、15 skipped、1 inherited failure |
| vendor 纯 USD 地面组合测试 | 6 passed |

UniLab Pyright 的错误位于未修改的 `reset_state.py`、`circular_buffer.py` 和
motion tracking `manager_terms.py`；可选 Drake 导入产生 warning。
UniLab 的既有文档检查失败源于历史文档引用未归档的
`logs/unidr/stop-at5000-20260916/delivery.md`，与本次训练配置无关。
UniSim 原有失败是 `test_on_frame_paints_video_frames`：GIF 编码合并了重复帧，
实际帧数 1 与断言的 2 不符。没有放宽断言或修改这些无关历史问题。
本地完整输出位于忽略目录 `.validation/update-*.log`。

独立环境未安装 `pxr`，该环境的 ground 测试跳过；另用现有 vendor Python
执行真实 USD 组合测试，未启动 Kit 或物理仿真，精确命令如下：

```bash
cd /home/wsm/wang-sm/UniDR-9-16-resume/unisim-unidr-backends
env -u UV_PROJECT_ENVIRONMENT UV_NO_SYNC=1 CUDA_VISIBLE_DEVICES='' \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  PYTHONPATH=/home/wsm/wang-sm/UniDR-9-16-resume/unisim-unidr-backends/src:/home/wsm/miniconda3/envs/atec_isaac51/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311 \
  LD_LIBRARY_PATH=/home/wsm/miniconda3/envs/atec_isaac51/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311/bin:/home/wsm/miniconda3/envs/atec_isaac51/lib \
  uv run --no-project --no-sync --python /home/wsm/miniconda3/envs/atec_isaac51/bin/python \
  -m pytest tests/test_isaacsim_ground.py -q
```

## 既有真实引擎证据与本轮限制

移植的两个后端实现及其地面/错误测试与当前已验证修复逐字节一致。
此前在 live 工作区：4096 环境、32 步零动作 PD 的提前终止数从 3763/4096 降至 0；
1024 修复前后均为 0。另通过参考相位、部分 reset、渲染开关和 4096 环境
100 轮 PPO 检查。原始证据位于本机：
`/home/wsm/wang-sm/UniLab-unidr-backends/logs/isaacsim-4096-diagnosis-20260921/`。

这些是移植前的真实引擎证据，不能替代本分支的真实容量复测。本轮仅在 CPU 和
纯 USD 环境执行检查，以免与正在运行的 GPU 训练争抢资源；未更改或停止该训练。
尚未在此分支执行 4096 环境的新训练、四卡实机测试或新的 MuJoCo 成功率评测。
训练命令见根 README；运行时需要重新启用 CUDA，使用新输出目录并从头初始化模型。
