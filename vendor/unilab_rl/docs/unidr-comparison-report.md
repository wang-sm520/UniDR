# Single-source and shared PPO comparison

After the four native runs and the shared run finish, generate the combined
training report from their fixed final checkpoints. This offline logging child
uses the existing budget auditors; it does not change sampling, learning,
checkpoint selection or the environment contract in ADR-0002.

From the UniLab consumer environment containing TensorBoard and Matplotlib:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --no-sync python \
  ../unilab_rl/examples/report_comparison.py \
  /absolute/completed-single-root \
  /absolute/completed-joint-root/train \
  /absolute/new-comparison-report
```

Defaults require four fresh 1024-environment, 20000-update single-source runs
and one fresh four-source 5000-update run. The existing auditors recheck each
fixed final checkpoint, actual optimizer counts, normalization and sample
budgets. The joint run must contain source timings for every iteration. A
source directory containing the wrong backend, missing/duplicate/non-finite
event data or unequal per-policy sample budgets fails before output creation.
Episode statistics may be absent before any episode ends; paired missing
return/length observations remain missing. Required loop timings must cover
every iteration. Output must be a new directory.

`comparison.png/pdf` pairs each native policy's rolling episode return and
length with the shared policy's corresponding source metrics. The horizontal
axis is global transitions; smoothing spans 200 native or 50 joint updates,
the same number of samples. Every policy receives 491520000 samples, while
the shared policy sees 122880000 samples per source. The native policies each
perform 400000 optimizer steps; the shared policy performs 100000 with larger
minibatches. This is an equal-sample comparison, not an equal-update comparison.

`series.csv` preserves raw metrics, global sample counts and per-source sample
counts. `loop_time.csv` and `loop_time.png/pdf` contain one global timing series
per policy, so the shared learner is counted once. Native GAE is measured inside
learning time while joint GAE is inside collection time; only their sums share
the same boundary. These recorded loops exclude startup and some logging and
checkpoint overhead. The source timing summary remains in the embedded joint
audit; concurrent source durations must not be added as elapsed training time.

`comparison.json` contains the audits, input event/metric hashes, recent metrics
and interpretation limits. It does not replace UniLab's separate configuration,
asset and video gate (`preflight.json`), or the inspection of actual final
MuJoCo videos. Episode length and reference resets alone cannot establish a
completed flip or landing. No holdout outcome selects or tunes a checkpoint.

Focused validation: `uv run --no-sync pytest tests/logging/test_comparison_report.py`.
These tests use real TensorBoard files and controlled audit boundaries; existing
audit tests verify checkpoint tensors. Report tests are not training evidence.
