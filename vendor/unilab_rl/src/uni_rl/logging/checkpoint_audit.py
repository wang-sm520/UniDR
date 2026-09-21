"""Audit a selected native PPO checkpoint for retrospective, fixed-run evaluation."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import torch

from uni_rl.logging.single_run_audit import (
    _checkpoint_state,
    _require,
    _run_metadata,
    _sha256,
)


def audit_single_checkpoint(
    run_dir: Path,
    *,
    checkpoint: Path,
    expected_iterations: int = 20000,
    num_envs: int = 1024,
) -> dict:
    """Validate a selected saved boundary inside an already completed native run.

    ``model_N.pt`` contains N+1 updates, not N updates. This checks the run's
    completed summary/config and the selected checkpoint's learner state; use
    ``audit_single_run`` separately to audit the final learner state as well.
    Evaluation does not rank checkpoints or change the fixed final selection.
    """
    try:
        run_dir = Path(run_dir).resolve(strict=True)
        requested = Path(checkpoint)
        checkpoint = requested.resolve(strict=True)
        _require(checkpoint.parent == run_dir, "checkpoint must belong to the completed run")
        _require(
            checkpoint.name == requested.name, "checkpoint symlink changes the requested index"
        )
        match = re.fullmatch(r"model_(0|[1-9][0-9]*)\.pt", checkpoint.name)
        _require(match is not None, "checkpoint must be named model_<iteration>.pt")
        assert match is not None
        index = int(match.group(1))
        algo, source, final_path = _run_metadata(run_dir, expected_iterations, num_envs)
        _require(index < expected_iterations, "checkpoint exceeds the completed run")
        completed = index + 1
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        lr = _checkpoint_state(saved, algo, completed, num_envs)
        seconds = saved["unilab_logger_state"]["tot_time"]
        _require(
            type(seconds) in (int, float) and math.isfinite(seconds) and seconds >= 0,
            "invalid saved logger elapsed time",
        )
        samples = completed * num_envs * 24
        return {
            "audit_passed": True,
            "scope": "native_single_source_selected_checkpoint_budget",
            "run_dir": str(run_dir),
            "source": source,
            "run_expected_iterations": expected_iterations,
            "run_final_checkpoint": str(final_path),
            "checkpoint_index": index,
            "completed_iterations": completed,
            "num_envs": num_envs,
            "transitions_per_iteration": num_envs * 24,
            "total_transitions": samples,
            "optimizer_steps": completed * 20,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "normalizer_counts": {"actor": samples, "critic": samples},
            "learning_rate": lr,
            "saved_logger_elapsed_seconds": seconds,
            "evidence_limit": (
                "Completed run metadata and selected learner state; final learner state must be "
                "audited separately. Does not prove task or holdout success."
            ),
        }
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        raise ValueError(f"invalid single-source checkpoint audit schema: {exc}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--expected-iterations", type=int, default=20000)
    parser.add_argument("--num-envs", type=int, default=1024)
    args = parser.parse_args()
    print(json.dumps(audit_single_checkpoint(**vars(args)), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
