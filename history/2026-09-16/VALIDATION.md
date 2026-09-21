# 9-16-resume reconstruction validation

Executed on 2026-09-21 in the isolated checkout
`/home/wsm/wang-sm/UniDR-9-16-resume`. This is a historical-source reconstruction,
not a new training run or an upstream release. No historical implementation was
modified to make a check pass.

## Integrity and configuration

- The original ZIP digest matches `sources.json`; all 644 archived file hashes
  match the restored files. All 1228 files reconstructed from baseline Git trees,
  patches and ZIP contents match `reconstruction.json`, including after tests.
- The three baseline Git commits exist locally. Three `git apply --check` checks
  passed before reconstruction. The original patches and ZIP are included here.
- Ten README/validation Bash blocks passed `bash -n`, their local links resolve,
  and both historical orchestration shell scripts passed syntax checks. Staged
  source contents were verified again, including Git symlink targets.
- Seven untracked docs/tests were not in the archive; their names are recorded
  in `reconstruction.json`. All archived production modules and entrypoints were
  recovered. This is not a complete OS, SDK, environment or worktree backup.
- Source scheduling, quota collection and probe modules/configs are absent.
  The old runner has no adaptive-source argument and the old IPC has no
  `step_selected`/probe lifecycle. Native adaptive KL PPO remains present.
- Isaac Sim `worker.py` matches the old snapshot and does not contain the later
  shared-ground repair. Historical limitations are retained deliberately.
- Each single-source comparison owner was composed and passed native
  `prepare_run` config/materialization validation. All four actual configs match
  their recorded experiment configs except the new output directory; all 78
  asset fingerprints match. No physical environment was constructed by this
  configuration check.
- `build_sources` and `build_manifest` passed for single-GPU and four-GPU joint
  owners. Both default to 1024 environments/source, 24 steps and 10000 iterations;
  neither exposes source adaptation. Mapping results are in
  [config-verification.json](validation/config-verification.json).

## Environment

An independent Python 3.10.12 virtual environment was created. Third-party pins
were copied from the already validated publication environment, excluding all
three editable owners, then the restored owners were installed editable with
`--no-deps`. This records the reconstruction validation environment; it is not a
claim that all third-party packages were frozen on September 16.

```bash
uv venv --python /usr/bin/python3.10 .venv
uv pip install --offline --no-deps --python .venv/bin/python \
  --index https://download.pytorch.org/whl/cu128 --index https://pypi.org/simple \
  -r history/2026-09-16/validation-requirements.txt
uv pip install --offline --no-deps --python .venv/bin/python \
  -e UniLab -e unilab_rl -e unisim-unidr-backends
```

All three imports and editable distributions point into the restored checkout.
Versions: unilab-rl 1.2.0, unisim-core 1.3.0, RSL-RL 5.0.1,
Torch 2.8.0+cu128, Genesis 1.3.3, Motrix 0.8.2. The original runtime lock remains
on RSL 5.5.0; the original root lock selects Torch 2.8.0+cu128 on Linux x86_64. No lock was
regenerated, and no live environment or system dependency was changed.

The 78 historical asset files were copied/verified locally; missing ignored asset
cache files, including the G1 texture completeness marker, were copied from the
existing asset cache. An initial config check waited for unavailable Hugging Face
network access and was interrupted. After restoring the missing caches, the
check passed with `HF_HUB_OFFLINE=1`. Asset binaries are not committed.

`uv pip check --python .venv/bin/python` reports one declared dependency issue:
RSL requests torchvision, while the original UniLab configuration deliberately
excludes torchvision. The existing installation used the same choice; no new
exclusion or package substitution was introduced to hide the report.

## Executed gates

Commands ran from the respective owner directory with this environment:

```bash
export UV_PROJECT_ENVIRONMENT=/home/wsm/wang-sm/UniDR-9-16-resume/.venv
export UV_NO_SYNC=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export UNILAB_LOCAL_UNISIM=/home/wsm/wang-sm/UniDR-9-16-resume/unisim-unidr-backends
unset PYTHONPATH
```

Pyright additionally used the existing modern Node at
`/home/wsm/.vscode-server/cli/servers/Stable-520fb30b2d3d324b4cb2342f6e88e2cd93751de1/server`
on PATH. No Node installation was performed.
Owner `make check` contains mutating formatting commands, so its read-only
equivalents were run to preserve historical bytes. No PR is being opened.

| Owner / check | Result |
| --- | --- |
| UniLab Ruff lint, format check, test-import lint | Passed |
| UniLab mypy | Passed |
| UniLab Pyright | 10 errors, 1 optional-import warning |
| UniLab complete non-slow suite | 1673 passed, 27 skipped, 849 deselected, 1 failed; coverage 70% |
| UniLab benchmark entrypoint import smoke | 33/34 module and 34/35 script imports passed; optional MLX skipped |
| runtime Ruff lint / format / mypy / Pyright | All passed; Pyright 0 errors, 0 warnings |
| runtime complete default suite | 505 passed, 34 skipped, 3 deselected; coverage 64.81% |
| runtime fake PPO, uniform and nonuniform | Both passed, real actor/critic changes, 192 samples and 20 optimizer steps |
| UniSim Ruff | Passed |
| UniSim complete default suite | 291 passed, 14 skipped, 1 failed |

UniLab:

```bash
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync mypy src/unilab
uv run --no-sync pyright --pythonpath "$UV_PROJECT_ENVIRONMENT/bin/python"
uv run --no-sync ruff check tests --select F401,F821,F811,F841 --output-format concise
uv run --no-sync pytest -m 'not slow' --cov=src/unilab --cov-report=term-missing
uv run --no-sync python scripts/benchmark/smoke_test.py
```

runtime:

```bash
uv run --no-sync ruff check src tests examples
uv run --no-sync ruff format --check src tests examples
uv run --no-sync mypy src/uni_rl
uv run --no-sync pyright --pythonpath "$UV_PROJECT_ENVIRONMENT/bin/python"
uv run --no-sync pytest --cov=src/uni_rl --cov-report=term --cov-fail-under=50 -p no:cacheprovider
uv run --no-sync python examples/unidr_fake_ppo.py
uv run --no-sync python examples/unidr_fake_ppo.py 'steps=[25,24,24,23]'
```

UniSim:

```bash
uv run --no-sync ruff check .
uv run --no-sync pytest -q
```

Original output and command records are included in [validation/](validation/).
No format/type/test exclusion was added to change the outcomes. There was no
formal training, four-source capacity run, four-card execution or new sim2sim
evaluation. Default tests include MuJoCo offscreen rendering; hiding CUDA does
not prohibit EGL use, so this is not a claim of strictly GPU-free rendering.

## Failures retained

UniLab Pyright errors are in `base/reset_state.py` (2),
`managers/_buffers/circular_buffer.py` (2) and
`tasks/motion_tracking/common/manager_terms.py` (6). These historical files match
the archive. The optional import warning concerns `drake_uni.runtime`.

The UniLab test failure is the documentation link checker:
`docs/unidr-synchronous-training.md` links to
`../logs/unidr/stop-at5000-20260916/delivery.md`, an ignored experiment artifact
outside this source reconstruction. The historical document is retained intact.

The UniSim failure is
`tests/test_debug_overlay.py::TestOnFrameEndToEnd::test_on_frame_paints_video_frames`.
Both expected callbacks run, but the GIF decodes as one frame while the assertion
expects two. This is consistent with an encoder coalescing identical frames;
it does not establish a training/physics failure. The diagnostic is preserved.

The complete repository gates are therefore not all green. The recovered PPO
runtime, configuration checks and source integrity checks passed, while the
above limitations remain visible for historical review.

`git diff --cached --check` also reports whitespace inside the original patch
archives (including context-only spaces) and inherited documentation whitespace.
Those evidence bytes are retained rather than rewriting the historical patches.
