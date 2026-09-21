"""Rebuild the three historical source owners in a clean UniDR worktree."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def command(*args: str, cwd: Path) -> bytes:
    return subprocess.check_output(args, cwd=cwd)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("provenance", type=Path)
    parser.add_argument("worktree", type=Path)
    args = parser.parse_args()
    provenance, worktree = args.provenance.resolve(), args.worktree.resolve()
    metadata = json.loads((provenance / "sources.json").read_text())
    assert (
        command("git", "branch", "--show-current", cwd=worktree).strip()
        == b"9-16-resume"
    )
    assert not command("git", "status", "--porcelain", cwd=worktree).strip()
    assert (worktree / ".git").is_file(), "require an isolated linked worktree"
    parent = command("git", "rev-parse", "HEAD", cwd=worktree).decode().strip()
    archive_path = provenance / "local-source.zip"
    assert sha(archive_path.read_bytes()) == metadata["archive_sha256"]
    mapping = {
        "unilab": Path("UniLab"),
        "uni_rl": Path("unilab_rl"),
        "unisim": Path("unisim-unidr-backends"),
    }
    evidence = {
        "reconstructed_at_utc": datetime.now(timezone.utc).isoformat(),
        "historical_capture_utc": metadata["captured_utc"],
        "historical_stage": metadata["stage"],
        "branch_parent": parent,
        "archive_sha256": metadata["archive_sha256"],
        "owners": {},
    }
    with (
        tempfile.TemporaryDirectory(prefix="unidr-9-16-") as scratch,
        zipfile.ZipFile(archive_path) as archive,
    ):
        desired = Path(scratch) / "tree"
        desired.mkdir()
        expected_names = {
            f"{owner}/{name}"
            for owner, record in metadata["repositories"].items()
            for name in record["files"]
        }
        assert set(archive.namelist()) == expected_names
        for owner, record in metadata["repositories"].items():
            baseline_repo = Path(record["root"])
            destination = desired / mapping[owner]
            destination.mkdir(parents=True, exist_ok=True)
            data = command("git", "archive", record["head"], cwd=baseline_repo)
            with tarfile.open(fileobj=io.BytesIO(data)) as baseline:
                for member in baseline.getmembers():
                    assert (
                        not Path(member.name).is_absolute()
                        and ".." not in Path(member.name).parts
                    )
                baseline.extractall(destination)
            patch = provenance / f"{owner}-worktree.patch"
            command("git", "apply", "--check", str(patch), cwd=destination)
            command("git", "apply", str(patch), cwd=destination)
            for relative, digest in record["files"].items():
                assert (
                    not Path(relative).is_absolute()
                    and ".." not in Path(relative).parts
                )
                content = archive.read(f"{owner}/{relative}")
                assert sha(content) == digest, relative
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
                assert sha(target.read_bytes()) == digest
            missing = [
                line[3:]
                for line in record["status"].splitlines()
                if line.startswith("?? ") and line[3:] not in record["files"]
            ]
            evidence["owners"][owner] = {
                "base_commit": record["head"],
                "destination": str(mapping[owner]),
                "verified_archived_files": len(record["files"]),
                "patch_sha256": sha(patch.read_bytes()),
                "unarchived_untracked_files": missing,
            }
        forbidden = [
            "unilab_rl/src/uni_rl/algos/source_schedule.py",
            "unilab_rl/src/uni_rl/algos/quota_ppo.py",
            "unilab_rl/src/uni_rl/algos/source_probe.py",
            "UniLab/src/unilab/training/source_probe.py",
            "UniLab/src/unilab/conf/ppo/task/g1_flip_tracking/unidr_adaptive.yaml",
            "UniLab/src/unilab/conf/ppo/task/g1_flip_tracking/unidr_comparison.yaml",
        ]
        assert all(not (desired / name).exists() for name in forbidden)
        evidence["later_files_verified_absent"] = forbidden
        evidence["restored_file_hashes"] = {
            str(p.relative_to(desired)): sha(p.read_bytes())
            for p in sorted(desired.rglob("*"))
            if p.is_file()
        }
        # Only remove the clean, newly created historical worktree's tracked files.
        command("git", "rm", "-r", "--quiet", "--", ".", cwd=worktree)
        shutil.copytree(desired, worktree, dirs_exist_ok=True, symlinks=True)
    history = worktree / "history/2026-09-16"
    history.mkdir(parents=True)
    for name in (
        "sources.json",
        "local-source.zip",
        "unilab-worktree.patch",
        "uni_rl-worktree.patch",
        "unisim-worktree.patch",
    ):
        shutil.copy2(provenance / name, history / name)
    shutil.copy2(Path(__file__), history / "reconstruct.py")
    (history / "reconstruction.json").write_text(json.dumps(evidence, indent=2) + "\n")
    experiments = {}
    for source in ("motrix", "isaacsim", "isaacgym", "genesis"):
        directory = provenance.parent / source
        manifest = json.loads((directory / "single_manifest.json").read_text())
        experiments[source] = {
            "config": manifest["config"],
            "assets": manifest["assets"],
            "packages": manifest["packages"],
            "historical_audit": json.loads(
                (directory / "single_audit.json").read_text()
            ),
        }
    (history / "single-experiments.json").write_text(
        json.dumps(experiments, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "worktree": str(worktree),
                "restored_files": len(evidence["restored_file_hashes"]),
                "owners": evidence["owners"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
