# Fixed-final MuJoCo holdout

Follow the [current README](../README.md) for installation and the
[holdout contract and historical validation](unidr-synchronous-training.md#fixed-final-model-mujoco-holdout).
Run from the UniDR repository root only after the selected training budget is
complete and audited. Checkpoints and server-side historical `logs/` artifacts
are not bundled. MuJoCo results must not select a checkpoint or tune training.

For a completed 10000-update fixed-share run, the final model is `model_9999.pt`:

```bash
MUJOCO_GL=egl uv run --no-sync python scripts/play_unidr_holdout.py /absolute/completed-run/model_9999.pt --output /absolute/new/holdout
```

For 5000 completed updates use `model_4999.pt` and add
`--expected-iterations 5000`. These fixed-share budget checks do not yet support
adaptive checkpoints. A successful recording validates execution and decoding;
it does not by itself establish a complete flip followed by five seconds without falling.
