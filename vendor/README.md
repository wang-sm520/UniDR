# Bundled G1 Flip Dependencies

UniDR includes the modified runtime and physics adapters required by its
G1FlipTracking experiments. They remain separate Python distributions; the
runtime does not import UniLab or UniSim.

| Directory | Distribution | Import | Responsibility |
| --- | --- | --- | --- |
| [unilab_rl](unilab_rl) | `unilab-rl` | `uni_rl` | PPO, collection, storage, IPC, probes, allocation, logs and recovery |
| [unisim](unisim) | `unisim-core` | `unisim` | Physics adapters, backend contracts and Isaac subprocess workers |

The source snapshots include their tests, documentation, LICENSE, AGENTS.md,
packaging metadata and standalone lockfiles. They do not include nested Git
repositories, virtual environments, robot meshes, motion downloads or external
Isaac SDKs. Nested GitHub workflow files are preserved source; only the root
`.github/workflows/` directory defines workflows for this GitHub repository.

[manifest.json](manifest.json) records upstream baselines, original working-tree
state and copied-file hashes. Baseline commits and unchanged package version
numbers alone are insufficient to reproduce this modified source. Packaging
edits and validation are documented in the
[publication report](../docs/validation/unidr-flip-publication-2026-09-21.md).
The original development checkouts and running experiments are not edited by
this publication.

## Root environment

Install from the UniDR root:

```bash
uv sync --python 3.10 --locked --extra mujoco --extra motrix --extra genesis
uv run --no-sync python -c 'import uni_rl, unisim; print(uni_rl.__file__); print(unisim.__file__)'
```

Both imports must resolve inside this repository's `vendor/` directory. The
root `pyproject.toml` and `uv.lock` select these packages as editable sources
and preserve the experiment's RSL-RL **5.0.1**. A sync inside `vendor/unilab_rl`
would instead select its separate development lock, which uses **5.5.0**.
Do not substitute that environment when reproducing the root training workflow.

The root MuJoCo extra uses `mjbatch-uni`; the earlier historical bundle used
`mujoco-uni-runtime`. Follow the current root lockfile rather than mixing those
backend generations. Python 3.10 matches the measured root environment; the
two Isaac SDK workers retain their separate Python 3.8 and Python 3.11
environments. Native worker entrypoints are resolved from the imported UniSim
source, so an additional sibling checkout is unnecessary.

See the root [README](../README.md#installation-and-assets) for assets and
`UNISIM_ISAACGYM_HOME` / `UNISIM_ISAACSIM_HOME` SDK discovery. No SDK install,
driver change, release tag or PyPI publication is part of this source bundle.

## Owner-specific checks

The root Ruff configuration excludes `vendor/`; each owner must be checked
separately with its own configuration. Reuse the root environment without
resolving the standalone dependency locks:

```bash
# Run from the UniDR root after the root sync above.
export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
export UV_NO_SYNC=1
make check
make test

(
  cd vendor/unilab_rl
  uv run --no-sync ruff check .
  uv run --no-sync ruff format --check .
  uv run --no-sync mypy src/uni_rl
  uv run --no-sync pyright
  uv run --no-sync pytest --cov=src/uni_rl
)
(
  cd vendor/unisim
  uv run --no-sync ruff check .
  uv run --no-sync pytest -q
)
```

Use `make test-all` at the root before a PR, as required by its agent guide.
Native tests require their declared optional SDKs and explicit opt-in controls;
a CPU/fake test does not establish four-engine physics support, adaptive
capacity, four-GPU execution or policy quality. The publication report records
actual results, including skipped tests and unresolved inherited failures.
