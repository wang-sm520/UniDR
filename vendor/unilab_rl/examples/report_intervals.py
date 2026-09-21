"""Export complete training logs in non-overlapping completed-update intervals."""

import argparse
import json

from uni_rl.logging.interval_report import report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("single_root")
    parser.add_argument("joint_run")
    parser.add_argument("output_dir")
    parser.add_argument("--interval", type=int, default=500)
    args = parser.parse_args()
    result = report(**vars(args))
    print(
        json.dumps(
            {
                "output": args.output_dir,
                "intervals": len(result["intervals"]),
                "source_intervals": len(result["source_intervals"]),
            }
        )
    )


if __name__ == "__main__":
    main()
