"""Retrospective checkpoint evaluation; never selects or modifies trained policies."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import Any

import imageio_ffmpeg
import numpy as np
from uni_rl.logging.synchronous_report import audit_run

from unilab.visualization.single_reference import _record_plan, _render_recording, preflight
from unilab.visualization.trajectory_metrics import evaluate_directory
from unilab.visualization.unidr_holdout import preflight as joint_preflight
from unilab.visualization.unidr_holdout import validate_video

PROTOCOL = "retrospective-500-v2"


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def validate_native_prefix(native_dir: Path, video_dir: Path) -> int:
    """Require the diagnostic replay to match native physics through its first done."""
    native = [
        json.loads(line) for line in (native_dir / "telemetry.jsonl").read_text().splitlines()
    ]
    video = [json.loads(line) for line in (video_dir / "telemetry.jsonl").read_text().splitlines()]
    first = next(i for i, row in enumerate(native) if row["terminated"] or row["truncated"])
    if len(native) != len(video) or not video[first]["first_native_failure"]:
        raise ValueError("Diagnostic replay changed the first native failure")
    for original, diagnostic in zip(native[: first + 1], video[: first + 1], strict=True):
        if (original["terminated"], original["truncated"]) != (
            diagnostic["native_terminated"],
            diagnostic["native_truncated"],
        ):
            raise ValueError("Diagnostic replay changed native termination conditions")
    with (
        np.load(native_dir / "physics_snapshots.npz") as original,
        np.load(video_dir / "physics_snapshots.npz") as diagnostic,
    ):
        np.testing.assert_array_equal(
            original["states"][: first + 1], diagnostic["states"][: first + 1]
        )
        np.testing.assert_array_equal(original["phase"][:first], diagnostic["phase"][:first])
    return first


def record_checkpoint(
    run_dir: Path, source: str, index: int, output: Path, *, root: Path
) -> dict[str, Any]:
    """Audit one selected model, score native physics, and record a diagnostic video."""
    planned = 5000 if source == "joint" else 20000
    if source not in ("motrix", "isaacsim", "isaacgym", "genesis", "joint"):
        raise ValueError("Unknown comparison source")
    if type(index) is not int or index not in (*range(0, planned, 500), planned - 1):
        raise ValueError("Expected a saved 500-iteration checkpoint or the fixed final")
    checkpoint = run_dir.resolve(strict=True) / f"model_{index}.pt"
    digest = sha256(checkpoint.read_bytes()).hexdigest()
    output = output.resolve()
    destination = output / "checkpoints" / source / f"model_{index:05d}"
    result_file = destination / "result.json"
    if result_file.exists():
        cached: dict[str, Any] = json.loads(result_file.read_text())
        expected = dict(
            protocol=PROTOCOL,
            checkpoint_sha256=digest,
            source=source,
            checkpoint_index=index,
            completed_updates=index + 1,
            total_transitions=(index + 1) * (98304 if source == "joint" else 24576),
            optimizer_steps=(index + 1) * 20,
        )
        if any(cached.get(key) != value for key, value in expected.items()):
            raise ValueError("Existing checkpoint evidence has a different protocol or model")
        required = [
            cached["video"],
            cached["poster"],
            str((destination / "metrics.json").relative_to(output)),
            str((destination / "native/verification.json").relative_to(output)),
        ]
        if not set(required) <= cached["artifact_sha256"].keys():
            raise ValueError("Cached evidence is missing required artifacts")
        for relative, artifact_hash in cached["artifact_sha256"].items():
            artifact = (output / relative).resolve(strict=True)
            if (
                not artifact.is_relative_to(output)
                or sha256(artifact.read_bytes()).hexdigest() != artifact_hash
            ):
                raise ValueError(f"Recorded artifact changed: {relative}")
        return cached
    if destination.exists():
        raise FileExistsError(
            f"Incomplete output must be inspected before a fresh rerun: {destination}"
        )
    if source == "joint":
        audit = audit_run(
            run_dir, expected_iterations=index + 1, num_envs=1024, final_checkpoint=checkpoint
        )
        plan = joint_preflight(checkpoint, root, expected_iterations=index + 1)
        if audit["final_checkpoint_sha256"] != plan.checkpoint_sha256 or audit["generations"] != [
            0
        ]:
            raise ValueError("Joint checkpoint provenance differs from the fresh run")
        plan = replace(plan, metadata={**plan.metadata, "training_audit": audit})
    else:
        plan = preflight(
            checkpoint,
            root,
            expected_iterations=planned,
            expected_num_envs=1024,
            selected_checkpoint=True,
        )
        if plan.metadata["source"] != source:
            raise ValueError("Requested source differs from training source")
    destination.mkdir(parents=True)
    native_dir, video_dir = destination / "native", destination / "video"
    native = _record_plan(plan, native_dir, root=root, render_video=False)
    metrics = evaluate_directory(native_dir, root=root)
    _json(destination / "metrics.json", metrics)
    if native["terminated"]:
        video = _record_plan(plan, video_dir, root=root, failure_tail_seconds=3.5)
        validate_native_prefix(native_dir, video_dir)
        mode = "failure-tail-3.5s"
    else:
        video_dir.mkdir()
        with np.load(native_dir / "physics_snapshots.npz") as snapshots:
            path = _render_recording(
                snapshots["states"],
                snapshots["reference"],
                [str(native_dir / name) for name in ("classic.mjb", "reference.mjb")],
                video_dir,
            )
        video = {
            **native,
            "video": str(path),
            "format": validate_video(path),
            "video_sha256": sha256(path.read_bytes()).hexdigest(),
        }
        _json(video_dir / "verification.json", video)
        mode = "native"
    media = output / "media"
    media.mkdir(exist_ok=True)
    filename = f"{source}_{index:05d}"
    movie, poster = media / f"{filename}.mp4", media / f"{filename}.jpg"
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-v",
            "error",
            "-nostdin",
            "-n",
            "-i",
            str(video["video"]),
            "-map",
            "0:v:0",
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(movie),
        ],
        check=True,
        timeout=120,
    )
    shutil.copyfile(video_dir / "preview.jpg", poster)
    if sha256(checkpoint.read_bytes()).hexdigest() != digest:
        raise ValueError("Checkpoint changed during evaluation")
    artifacts = [
        movie,
        poster,
        destination / "metrics.json",
        native_dir / "verification.json",
        native_dir / "physics_snapshots.npz",
        native_dir / "telemetry.jsonl",
        native_dir / "mujoco_config.json",
        video_dir / "verification.json",
    ]
    if mode != "native":
        artifacts.extend(
            video_dir / name
            for name in ("physics_snapshots.npz", "telemetry.jsonl", "playback_overrides.json")
        )
    result = dict(
        protocol=PROTOCOL,
        source=source,
        checkpoint_index=index,
        completed_updates=index + 1,
        planned_updates=planned,
        checkpoint=str(checkpoint),
        checkpoint_sha256=digest,
        total_transitions=(index + 1) * (98304 if source == "joint" else 24576),
        optimizer_steps=(index + 1) * 20,
        video_mode=mode,
        video=str(movie.relative_to(output)),
        poster=str(poster.relative_to(output)),
        video_format=video["format"],
        scoring="native independent rollout; first 224 steps fixed horizon",
        metrics=metrics,
        artifact_sha256={
            str(p.relative_to(output)): sha256(p.read_bytes()).hexdigest() for p in artifacts
        },
    )
    _json(result_file, result)
    return result
