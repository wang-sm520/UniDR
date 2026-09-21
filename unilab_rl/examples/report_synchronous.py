"""Audit completed synchronous PPO artifacts and render their learning curves."""

import argparse
import json

from uni_rl.logging.synchronous_report import report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--expected-iterations", type=int, default=10000)
    parser.add_argument("--num-envs", type=int, default=1024, help="Environments per source")
    parser.add_argument(
        "--final-checkpoint", help="Explicit completed prefix; original run intent is preserved"
    )
    args = vars(parser.parse_args())
    print(json.dumps(report(**args), indent=2))


if __name__ == "__main__":
    main()
