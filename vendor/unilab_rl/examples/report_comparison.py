"""Read completed single-source and joint runs and write the final comparison."""

import argparse
import json

from uni_rl.logging.comparison_report import report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("single_root")
    parser.add_argument("joint_run")
    parser.add_argument("output_dir")
    args = parser.parse_args()
    result = report(**vars(args))
    print(json.dumps({"output_dir": args.output_dir, "scope": result["scope"]}))


if __name__ == "__main__":
    main()
