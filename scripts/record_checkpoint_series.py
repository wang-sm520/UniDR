"""Record one explicitly named checkpoint for the completed experiment review."""

import argparse
import json
from pathlib import Path

import torch

from unilab.visualization.checkpoint_series import record_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=["motrix", "isaacsim", "isaacgym", "genesis", "joint"])
    parser.add_argument("index", type=int)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = record_checkpoint(
        args.run_dir, args.source, args.index, args.output, root=Path(__file__).resolve().parents[1]
    )
    print(json.dumps({k: result[k] for k in ("source", "checkpoint_index", "video", "video_mode")}))


if __name__ == "__main__":
    main()
