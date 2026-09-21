"""Run ten native MuJoCo single-flip attempts for each fixed final policy."""

import argparse
import json
from pathlib import Path

import torch

from unilab.visualization.flip_trials import run_trials

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--single-run",
        nargs=2,
        action="append",
        metavar=("SOURCE", "RUN_DIR"),
        help="Evaluate this single-source run's fixed final model; repeat for multiple sources",
    )
    parser.add_argument("--expected-iterations", type=int, default=20000)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument(
        "--checkpoint-series",
        action="store_true",
        help="Evaluate every saved 500-iteration checkpoint once at seed 1",
    )
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(
        json.dumps(
            run_trials(
                Path(__file__).resolve().parents[1],
                args.output,
                checkpoint_series=args.checkpoint_series,
                single_runs=[(source, Path(run)) for source, run in args.single_run]
                if args.single_run is not None
                else None,
                expected_iterations=args.expected_iterations,
                expected_num_envs=args.num_envs,
            ),
            indent=2,
        )
    )
