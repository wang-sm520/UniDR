"""Build the portable completed-experiment comparison after all videos exist."""

import argparse
import json

from unilab.visualization.experiment_report import report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root")
    args = parser.parse_args()
    result = report(args.output_root)
    print(json.dumps({k: result[k] for k in ("complete", "checkpoints", "embedded_videos")}))


if __name__ == "__main__":
    main()
