#!/usr/bin/env python3
"""Run one market with exactly the sellers and buyers listed in the config.

Use it to test your own agents against chosen opponents. Every setting (agents, items,
repeats, seed, ...) is in the config file; copy it to try other settings.

  python run_market.py                                     # configs/market.yaml
  python run_market.py --config configs/my_market.yaml
  python run_market.py --mock-llm --dry-run                # who plays; no model needed
  python run_market.py --resume results/run_market_<time>  # continue a stopped run
"""

import argparse
import sys

from adlmarket.cli import add_common_args, execute
from adlmarket.setups import market_run


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/market.yaml")
    add_common_args(ap)
    args = ap.parse_args()
    return execute(args, lambda: market_run(args.config))


if __name__ == "__main__":
    sys.exit(main())
