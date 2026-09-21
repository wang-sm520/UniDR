# G1 flip source publication — 2026-09-21

This publication replaces UniDR's old WalkFlat development snapshot with the
current G1FlipTracking implementation. It includes task/configuration owners,
four-source synchronous PPO, adaptive quotas/probes, comparison and MuJoCo
holdout tools, and the repaired Isaac Sim ground-cloning implementation.
The prior snapshot remains in Git history at
[`22225bcc`](https://github.com/wang-sm520/UniDR/tree/22225bcc24272a9434a6fbae7dba42447a93ff2a).
No checkpoint, recording, downloaded robot asset or vendor SDK is uploaded.

## Source and scope

| Owner | Base commit | Imported source files |
| --- | --- | --- |
| UniLab | `b4e6b58fe0861a435fd19c0f0206bd84f4427a9c` | 974 |
| uni_rl | `79418e0cbff7b95fe6e49709b194454701ba20fa` | 165 |
| UniSim | `4270aa81d868744980db90dac6dd959d3f542f50` | 141 |

These are dirty development snapshots, not unmodified upstream commits.
[The manifest](../../vendor/manifest.json) records actual remotes, baseline
commits, source working-tree status, original file hashes, final published file
hashes and packaging modifications. UniLab's actual source remote is
`Motphys/UniLab`; the runtime and physics remotes are `unilabsim/unilab_rl`
and `unilabsim/unisim`. All three original Apache-2.0 licenses remain intact.

This is consolidation of existing, separately developed children into one
source repository; its import size is not a new 15-file/800-line algorithm
child. Owner boundaries and algorithms are unchanged. The applicable decisions
are runtime ADRs [C1](../../vendor/unilab_rl/docs/adr/0001-multi-source-ppo-window.md),
[central PPO](../../vendor/unilab_rl/docs/adr/0002-central-synchronous-ppo.md),
and [adaptive quotas](../../vendor/unilab_rl/docs/adr/0003-adaptive-source-quotas.md).
No public protocol or runner lifecycle is added by packaging.

The isolated publication checkout was based on the exact GitHub parent tree.
All three live source trees and their editable environments were left untouched.
Source synchronization copied tracked and nonignored additions, removed obsolete
snapshot-only code/tests, and preserved current upstream generic tasks. Old
WalkFlat experiment entrypoints and checkpoint copies were removed from this
head; this is not a Git history rewrite.

Publication adaptations are limited to:

- Root `pyproject.toml` / `uv.lock` select editable packages under `vendor/`.
  Root Ruff excludes vendor files; their own lint/type/test configurations run
  separately. Third-party locked versions match the current live UniLab lock.
- The synchronous manifest records the imported dependency locations rather
  than assuming adjacent external checkouts.
- The two orchestration shell scripts resolve the bundled runtime report
  entrypoint first, retaining the separate-checkout fallback.
- The dependency-origin test accepts the exact repository's bundled editable
  UniSim and still verifies both metadata and actual module location.
- The docs checker now captures Markdown path links correctly; a regression
  test covers valid and missing targets. No strict environment check is disabled.
- READMEs and current operating instructions use clone-local paths and focus
  on flip; historical experiment commands remain labeled as historical evidence.

## Installation and validation environment

The isolated checkout uses Python 3.10.12, PyTorch 2.8.0+cu128, RSL-RL 5.0.1,
unilab-rl 1.2.0, unisim-core 1.3.0, Genesis 1.3.3, Motrix 0.8.2,
MuJoCo 3.11.0 and mjbatch-uni 0.2.0. Runtime's independent development lock
continues to use RSL-RL 5.5.0; it does not override the root environment.

```bash
uv lock
uv lock --check --offline
uv sync --python /usr/bin/python3.10 --locked --offline \
  --extra mujoco --extra motrix --extra genesis
uv run --no-sync python -c 'import unilab, uni_rl, unisim; print(unilab.__file__); print(uni_rl.__file__); print(unisim.__file__)'
```

All three imports were verified under this checkout. The first offline lock
attempt lacked cached ARM Torch metadata; online resolution succeeded and the
subsequent offline lock check and locked sync passed (250 packages).
No live environment or system/SDK dependency was reinstalled.

Ignored asset caches were copied from the existing installation for tests,
without changing or publishing the binaries. The README gives asset-hub setup
commands. [Measured asset hashes](flip-asset-hashes-2026-09-21.json) describe the
78 files used by the repaired experiment. The hub currently fetches assets
without an explicit repository revision; code SHA alone does not pin downloaded
meshes. The training owner checks the exact reference-motion and G1 XML hashes
and snapshots all effective asset hashes.

## Executed checks

All test commands hid CUDA and used one OMP/MKL thread. These are software and
packaging checks, not a repeat of the four-engine training experiment.

| Check | Result |
| --- | --- |
| Locked isolated install and module-origin assertions | Passed |
| Runtime Ruff lint / format | Passed; 143 Python files formatted |
| Runtime Mypy / Pyright | Passed; 87 typed source files, zero Pyright errors |
| Runtime full default suite, RSL-RL 5.0.1 | 671 passed, 34 skipped, 3 deselected; 68.61% coverage |
| Fake uniform and nonuniform PPO smoke | Both passed: 192 samples, 20 optimizer steps, actor and critic changed |
| Root lint / format / Mypy / test-import lint | Passed; 149 source files checked by Mypy |
| Root complete `make test-all` | Blocked by 10 existing Pyright errors; see below |
| Root non-slow suite, final rerun | 1879 passed, 27 skipped, 851 deselected; 72% coverage |
| Focused packaging/docs regression checks | 26 passed |
| Benchmark entrypoint import smoke | 33/34 module and 34/35 script imports passed; optional MLX skipped |
| UniSim lint / full tests | Lint passed; 329 passed, 15 skipped, 1 existing GIF test failed |
| UniSim offline wheel and sdist | Both built successfully |
| Hydra and command routes | Four joint layouts built; four single owners composed; four evaluation/report help routes passed |
| README shell snippets and links | 17 Bash blocks passed syntax checks |
| Owner operating docs | 21 Bash blocks and 19 local links passed |
| Sphinx HTML with autodoc skipped | Passed |

Root commands:

```bash
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 UV_NO_SYNC=1
make test-all
COVERAGE_FILE=.tmp/flip-publication/.coverage make test-cov
uv run --no-sync ruff check tests --select F401,F821,F811,F841 --output-format concise
make test-benchmark-smoke
uv run --no-sync pytest tests/scripts/test_check_docs.py -q
uv run --no-sync python vendor/unilab_rl/examples/unidr_fake_ppo.py
uv run --no-sync python vendor/unilab_rl/examples/unidr_fake_ppo.py 'steps=[25,24,24,23]'
```

The first root suite found three publication-environment issues: the editable
vendor path was not yet recognized by the dependency-origin test; the extracted
old snapshot retained an ignored `__pycache__` directory that made the removed
DR package importable as a namespace; and a Markdown link exposed the docs
checker's missing capture group. The exact vendor-origin check and docs parser
were corrected, the obsolete ignored cache was removed, and the full suite
then passed. No assertion was disabled.

Documentation build, from `docs/sphinx`:

```bash
UNILAB_DOCS_SKIP_AUTODOC=1 uv run --no-project --with-requirements requirements.txt \
  sphinx-build -b html -n source /tmp/unidr-flip-sphinx-html
```

The nonuniform smoke used source counts `[50,48,48,46]`; the uniform smoke
used `[48,48,48,48]`. Each changed the real actor and critic parameters and
recorded Adam step 20. The runtime suite also exercises synchronous collection,
normalizers, GAE, quotas, probes, faults and recovery. It is not evidence of
real adaptive training or four-GPU execution.

Runtime commands, from `vendor/unilab_rl` with
`UV_PROJECT_ENVIRONMENT=/home/wsm/wang-sm/UniDR/.venv`:

```bash
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync mypy src/uni_rl
uv run --no-sync pyright --pythonpath /home/wsm/wang-sm/UniDR/.venv/bin/python
uv run --no-sync pytest --cov=src/uni_rl --cov-report=term --cov-fail-under=50
```

UniSim commands, from `vendor/unisim` using the same root environment:

```bash
UV_NO_SYNC=1 make check
uv build --no-build-isolation --offline \
  --python /home/wsm/wang-sm/UniDR/.venv/bin/python \
  --out-dir ../../.tmp/flip-publication/unisim-dist --no-create-gitignore
```

The initial Pyright invocation hit the host's old Node interpreter. Repeating
with the already installed VS Code server's Node on `PATH` ran the checks.
The exact prefix was
`/home/wsm/.vscode-server/cli/servers/Stable-520fb30b2d3d324b4cb2342f6e88e2cd93751de1/server`.
No Node installation or version change was performed.

The remaining root Pyright diagnostics are in `base/reset_state.py` (2),
`managers/_buffers/circular_buffer.py` (2), and
`tasks/motion_tracking/common/manager_terms.py` (6). These three files match
the imported live snapshot. There is also one optional Drake import warning.
No exclusion or type-ignore was added to hide them, so the complete gate is
explicitly not green.

UniSim's unchanged
`TestOnFrameEndToEnd.test_on_frame_paints_video_frames` expects two GIF frames
after rendering identical images; the encoder merges them into one. Both frame
callbacks execute. This also failed in the September 14 bundle validation;
the test was not skipped or changed for this publication.

Staged `git diff --check` also reports an inherited extra blank line at EOF in
`scripts/tools/drift_baseline/drift_report.md`. That imported experiment report
is retained byte-for-byte; no code whitespace failure was found.

## Experiment status and limits

The repaired single Isaac Sim 4096-env/5000-update run completed with
491,520,000 transitions and 100,000 optimizer steps. Its fixed final
`model_4999.pt` SHA-256 is
`4a484be33632dae719b72bfdd0eab6dd40b1f71e009996e6fe4d95983d6126b4`.
The repaired joint 4×1024-env run was still active while this snapshot was
prepared; publishing source does not stop or restart it. Final training-budget,
checkpoint and process/resource audits are separate from this source check.

Four-GPU support has mapping tests but no four-card physical validation.
Adaptive scheduling has fake PPO coverage and a small real environment/probe
check; no formal adaptive training or 4×1024 adaptive capacity is claimed.
MuJoCo remains a fixed-final-model holdout. Current deterministic single-policy
trials did not pass the full flip followed by five seconds without falling.
No sim2real benefit or transfer success is inferred from training completion.

This is a source snapshot pushed to the user's research repository, not an
upstream PyPI release, a release tag, or a claim that all CI gates pass.
