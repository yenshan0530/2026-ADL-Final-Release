#!/usr/bin/env python3
"""Run the whole class as an economy: partition the graded agents into sub-markets.

  --phase 1   student sellers + TA buyers        (graded: sellers)
  --phase 2   TA sellers + student buyers        (graded: buyers)
  --phase 3   student sellers + student buyers   (graded: both)

Every setting is in the config file (default configs/economy.yaml): the agent sets
(`agents:` {ta, student} x {sellers, buyers}), market sizes (sellers_per_market,
buyers_per_market; TA agents fill seats students do not), the item file, repeats (how many
economies; each splits the students into markets again), seed, ... Copy the file to try
other settings. Every market is played once per seller seat (rotation).

  python run_economy.py --phase 2 --dry-run            # who plays in which market
  python run_economy.py --phase 2 --mock-llm           # no model needed
  python run_economy.py --phase 1 --config configs/my_economy.yaml
  python run_economy.py --resume results/run_economy_<time>
"""

import argparse
import sys

from adlmarket.cli import add_common_args, execute
from adlmarket.setups import PHASE_MODE, economy_run


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", type=int, choices=sorted(PHASE_MODE),
                    help="1: student sellers vs TA buyers, 2: TA sellers vs student buyers, "
                         "3: student sellers vs student buyers")
    ap.add_argument("--config", default="configs/economy.yaml")
    add_common_args(ap)
    args = ap.parse_args()
    if not (args.resume or args.phase):
        ap.error("give --phase 1, 2 or 3 (or --resume RUN_DIR)")

    def build():
        info: list = []
        run, _ = economy_run(args.config, PHASE_MODE[args.phase], info=info)
        for line in info[0].lines():
            print(line)
        return run

    return execute(args, build)


if __name__ == "__main__":
    sys.exit(main())
