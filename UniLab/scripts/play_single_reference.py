"""Record a fixed final single-source policy on MuJoCo with its reference motion."""

import argparse
import json
from pathlib import Path

import torch

from unilab.visualization.single_reference import record_reference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-iterations", type=int, default=20000)
    parser.add_argument("--num-envs", type=int, default=1024)
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(
        json.dumps(
            record_reference(
                args.checkpoint,
                args.output,
                root=Path(__file__).resolve().parents[1],
                expected_iterations=args.expected_iterations,
                expected_num_envs=args.num_envs,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
