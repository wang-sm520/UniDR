# G1 Four-Simulator PPO: Final Checkpoint

`model_9999.pt` is the unchanged final checkpoint of the completed 10,000-update
run `20260912_205814_multisim_10000`. The index is zero-based; the original
`run_summary.json` reports `completed_iterations: 9999`, while the total of
1,920,000,000 transitions at 192,000 per update corresponds to 10,000 updates.

| Property | Value |
| --- | --- |
| Task | G1WalkFlat, flat ground, 29 actions |
| Training sources | IsaacGym, IsaacSim, Motrix, Genesis; fixed 25% each |
| Environments | 2,000 per source, 8,000 total |
| Actor / critic observations | 98 / 101 |
| Network hidden layers | 512, 256, 128; ELU |
| Control period / action scale | 0.02 s / 0.25 |
| Episode horizon | 20 s |
| Command | Body-local vx [0.4, 0.7] m/s; vy=0, wz=0 |
| Training DR | Pelvis mass [0.8, 1.2]; joint KP and KD each [0.9, 1.1] |
| Checkpoint SHA256 | `2d5414d5af4e21626d4bb77c9f32f62a339cf35524dce8a48739668b7a7e647a` |

The checkpoint is an RSL-RL training dictionary, not a standalone TorchScript or
ONNX module. It includes actor, critic, optimizer, iteration, and logger state.
`run_config.json` preserves the original training configuration and strict
Sim2Sim contract. `run_summary.json` is the original completion record; historical
machine paths in these files are provenance, not instructions to create those
paths. SHA256 values and file sizes are in `manifest.json`.

## Install From This Repository

```bash
git clone https://github.com/wang-sm520/UniDR.git
cd UniDR
uv sync --python 3.11 --locked --extra mujoco --extra motrix --extra genesis
export UNILAB_LOCAL_UNISIM="$(pwd)/vendor/unisim"
uv run --no-sync python -c 'import uni_rl, unisim; print(uni_rl.__file__); print(unisim.__file__)'
```

The two dependency imports should resolve inside `vendor/`. Robot meshes are
downloaded from the existing asset hub when needed. Isaac SDKs are not required
for MuJoCo evaluation; their external runtimes are required to retrain with all
four original simulators. This is a source-checkout workflow, not a new package
release. See [dependency provenance](../../vendor/README.md).

## Evaluate in MuJoCo

Run from the repository root:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 uv run --no-sync eval \
  --algo ppo --task g1_walk_flat --sim mujoco --profile multisim --metrics \
  algo.load_run=checkpoints/g1_walk_flat_multisim_10000/model_9999.pt \
  training.device=cpu
```

This performs strict Sim2Sim and checkpoint dimension checks before environment
construction, then disables evaluation observation corruption and physics DR
while keeping the training task reset distribution. The actions are
deterministic. Four seeds [101, 102, 103, 104] with 25 environments each contribute
their first complete episodes. Reports are written to a new `evaluation/`
directory beside the checkpoint, not to the historical training log paths.

For a 720p, 50 FPS video of the first seed's environment zero:

```bash
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 uv run --no-sync eval \
  --algo ppo --task g1_walk_flat --sim mujoco --profile multisim --metrics \
  algo.load_run=checkpoints/g1_walk_flat_multisim_10000/model_9999.pt \
  training.device=cpu training.evaluation.record_video=true \
  training.cam_distance=3.5 training.cam_elevation=-15
```

EGL needs a working offscreen OpenGL driver. The policy and MuJoCo simulation
use CPU here; rendering may use the GPU. Recording stops at the first episode
termination/timeout and does not select a successful retry.

## Measured Transfer

The 2026-09-13 MuJoCo evaluation completed 100 episodes, all lasting 20 seconds,
with no non-timeout terminations. Mean episode-aggregated vx MAE was 0.030846 m/s,
mean absolute lateral velocity 0.020777 m/s, and mean absolute torso-local yaw
angular velocity 0.189584 rad/s. The yaw metric is from the torso IMU, not pelvis
angular velocity. These are nominal-physics, clean-observation flat-ground
results, not proof of physical-DR robustness, real-world deployment, or all
four-backend physics/capacity/stability gates passing.

The bundled-dependency recheck on 2026-09-14 reproduced all 100 episode records
exactly. The portable records and metric definitions are in
`mujoco_evaluation.json`; source and packaging checks are recorded in
[the bundle validation report](../../docs/validation/unidr_bundle_2026-09-14.md).
