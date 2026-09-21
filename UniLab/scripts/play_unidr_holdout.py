"""Record the fixed final four-source policy on the MuJoCo holdout."""

import argparse
import json
from pathlib import Path

import torch

from unilab.visualization.unidr_holdout import record_holdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="new evidence directory")
    parser.add_argument(
        "--expected-iterations",
        type=int,
        default=10000,
        help="explicit completed-update budget; checkpoint index must equal this minus one",
    )
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = record_holdout(
        args.checkpoint,
        args.output,
        root=Path(__file__).resolve().parents[1],
        expected_iterations=args.expected_iterations,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
