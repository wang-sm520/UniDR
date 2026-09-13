# Bundled Development Dependencies

UniDR keeps the modified `unilab_rl` and `unisim` source trees in this one Git
repository, as requested for the G1 multisim training snapshot. They remain
separate Python distributions with their original module and ownership boundaries:

| Directory | Distribution | Import | Responsibility |
| --- | --- | --- | --- |
| `unilab_rl/` | `unilab-rl` | `uni_rl` | Learners, runners, collection, IPC, training logs |
| `unisim/` | `unisim-core` | `unisim` | Backend adapters, physics contracts, Isaac subprocess workers |

Each directory contains the source repository's tracked files plus its current
non-ignored working-tree additions, including tests, documentation, LICENSE,
AGENTS.md, packaging configuration, and its development lockfile. Nested `.git`
directories, virtual environments, cached/downloaded assets, proprietary Isaac
SDKs, and generated outputs are not included. The original repositories outside
UniDR were neither changed nor pushed. Nested GitHub workflows are preserved as
source files; only root `.github/workflows/` files are active workflows in UniDR.

`manifest.json` records upstream URLs, baseline commits, original working-tree
status, and the SHA256 of each copied file. It identifies the current development
snapshot, not an assertion that every file was loaded by the original training
process. Subsequent vendor edits must update the recorded file hashes or be
explicitly recorded as modifications to this snapshot.

## Installation

Run from the UniDR repository root with Python 3.11:

```bash
uv sync --python 3.11 --locked --extra mujoco --extra motrix --extra genesis
uv run --no-sync python -c 'import uni_rl, unisim; print(uni_rl.__file__); print(unisim.__file__)'
```

The root `pyproject.toml` and `uv.lock` select these directories as editable
sources. The imported modules must resolve under `vendor/`, not to published
1.2.0 distributions. Package versions are unchanged, so source hashes, not
version numbers alone, identify the experiment implementation.

For the Isaac subprocesses, expose the same source root explicitly:

```bash
export UNILAB_LOCAL_UNISIM="$(pwd)/vendor/unisim"
```

IsaacGym and IsaacSim still require their separately installed external Python
runtimes. See [multisim setup](../docs/multisim_training.md) for those prerequisites.
The acceptance driver now prefers these bundled roots while retaining the older
sibling-repository layout when `vendor/` is absent. No physics gate is weakened.
Native readback receives UniDR's G1 scene through the existing `UNILAB_G1_SCENE`
option; an explicit user-provided scene value is preserved.

## Validation

Run dependency checks separately so the original owner lint/test configuration
applies. The root Ruff configuration excludes `vendor/` to avoid rewriting
snapshotted source during an unrelated root-format command. The root pytest
suite also does not replace these independent suites.

Use the root environment for each dependency's checks, with `UV_NO_SYNC=1`;
running a fresh sync inside a vendor directory instead selects that package's
standalone development environment and lockfile. No release tags or package
publication are part of this source upload.
