# UniDR Single-Repository Bundle, 2026-09-14

## Scope

The user requested the final trained policy and both modified dependencies in
the existing public `wang-sm520/UniDR` repository, not separate GitHub
repositories. The Python package boundaries and ADR-0010 synchronous multisim
architecture are unchanged; only source distribution/layout and cold-path
acceptance root selection are adapted.

- `vendor/unilab_rl`: 139 source-repository files, 1,505,535 bytes.
- `vendor/unisim`: 145 source-repository files, 2,024,542 bytes.
- Both original working trees, HEADs, refs, and all 284 copied files remained
  unchanged. Their upstream URLs, baseline commits, dirty-file lists, and
  per-file hashes are in `vendor/manifest.json`.
- Original package LICENSE files and instructions are preserved. There are no
  nested Git repositories, SDK copies, virtual environments, or cached assets.
- The final checkpoint, original run configuration, and original completion
  record are byte-identical copies under
  `checkpoints/g1_walk_flat_multisim_10000/`. The checkpoint is 5,243,829 bytes.
- Checkpoint SHA256:
  `2d5414d5af4e21626d4bb77c9f32f62a339cf35524dce8a48739668b7a7e647a`.

Root uv sources and the lockfile use the bundled editable packages. The locked
third-party dependency versions did not change. The isolated development
environment sync replaced only UniLab and the two editable distributions; it
did not upgrade CUDA, PyTorch, conda environments, or vendor SDKs. Module origin
and installed editable metadata were checked against both `vendor/` paths.

The acceptance owner prefers bundled roots and retains sibling-root fallback.
Native readback uses the existing `UNILAB_G1_SCENE` option for this repository's
scene, preserving an explicit caller override. CPU tests cover these choices
and independent source fingerprints, including untracked dependency changes.
No native physical test was made optional or weakened.

## Validation Results

| Check | Result |
| --- | --- |
| Root `uv lock --check --offline` | Passed, 252 packages |
| Root locked offline sync | Passed; only three editable distributions replaced |
| Root `UV_NO_SYNC=1 make check` | Passed; one optional Drake import warning |
| Root non-slow `make test` | 1716 passed, 30 skipped, 851 deselected, 1 xfailed; 56.66 s |
| unilab_rl lint/format/mypy/pyright | Passed; 125 formatted files, 81 typed source files |
| unilab_rl default tests with coverage | 459 passed, 34 skipped, 3 deselected; 63.19% coverage |
| unisim lint and offline lock check | Passed |
| unisim format check | 21 need formatting, 102 already formatted; all 21 also fail at upstream baseline HEAD |
| unisim tests | 344 passed, 19 skipped, 1 failed; 8.83 s |
| unisim offline wheel/sdist build | Passed, including wheel import and fake-backend smoke |
| Real bundled MuJoCo policy evaluation | 100 first episodes completed; all 100 records exactly match the pre-bundle run |

The unisim failure is the unchanged
`vendor/unisim/tests/test_debug_overlay.py::TestOnFrameEndToEnd::test_on_frame_paints_video_frames`:
GIF encoding merges identical images into one frame while the test expects two.
Both frame callbacks executed. It reproduces independently of the bundle; no
source rewrite or format-only churn was applied to hide this failure.

Skipped tests include opt-in native readback, unavailable optional backends,
unsupported Python/backend combinations, and unsupported geometry. They do not
establish physics acceptance. The outstanding Motrix live-inertia readback
limitation is unchanged. This upload is not a release or a declaration that all
four-source G1-G7 gates passed. No training was started.

The previous root documentation-path failure is resolved by pointing the
historical source-timing report to its now-bundled dependency files. Its measured
results and historical status text are unchanged.

The staged diff check passes when excluding `vendor/`. Checking the complete
import also reports inherited blank lines at EOF in `vendor/unisim/.gitignore`,
`vendor/unisim/LICENSE`, and `vendor/unisim/tests/test_superdex_contract.py`.
These original files are intentionally kept byte-identical to their source
snapshot rather than changing license or unrelated formatting during upload.

## Commands

All Python checks used the existing Python 3.11.16 development environment and
PyTorch 2.8.0+cu128. The source-copy audit preceded the editable-path switch;
root checks and the real policy recheck ran after that switch.

From UniDR's root, after loading the local environment:

```bash
source .tmp/multisim-gpu.env
uv lock
uv lock --check --offline
uv sync --locked --offline --extra mujoco --extra motrix --extra genesis
UV_NO_SYNC=1 make check
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 UV_NO_SYNC=1 make test
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 uv run --no-sync eval \
  --algo ppo --task g1_walk_flat --sim mujoco --profile multisim --metrics \
  algo.load_run=checkpoints/g1_walk_flat_multisim_10000/model_9999.pt \
  training.device=cpu
git diff --check
```

The first offline lock attempt lacked cached Drake metadata; an online lock
resolution succeeded, and the subsequent offline lock check and sync passed.

From the original unilab_rl source root, with `UV_NO_SYNC=1`, hidden CUDA, and
one OMP thread:

```bash
uv run ruff check --no-cache .
uv run ruff format --check --no-cache .
uv run mypy src/uni_rl
uv run pyright
uv run pytest -p no:cacheprovider --cov=src/uni_rl --cov-report=term --cov-fail-under=50
```

From the original unisim source root, with the same environment:

```bash
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync pytest -q
uv lock --check --offline
uv build --no-build-isolation --offline --python "$UV_PROJECT_ENVIRONMENT/bin/python" --out-dir "$AUDIT_DIR/dist" --no-create-gitignore
```

Audit caches, basetemp, coverage files, JUnit XML, package builds, and detailed
logs were directed under the ignored `.tmp/unidr-bundle-audit/` directory.
Portable policy evaluation records are committed as
`checkpoints/g1_walk_flat_multisim_10000/mujoco_evaluation.json`.
