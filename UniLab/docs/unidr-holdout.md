[Fixed-final MuJoCo instructions and validation](unidr-synchronous-training.md#fixed-final-model-mujoco-holdout). Real evaluation remains pending; run only after the formal `model_9999.pt` exists.

```bash
MUJOCO_GL=egl uv run --no-sync python scripts/play_unidr_holdout.py logs/unidr/single-formal-1024-10000-20260916/model_9999.pt --output logs/unidr/mujoco-final-20260916
```
