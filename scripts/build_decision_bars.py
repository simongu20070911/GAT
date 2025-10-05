"""Build decision bars for a given OHLCV dataset."""
from __future__ import annotations

import argparse

import pandas as pd

from candlestrats.bars import DecisionBarConfig, build_decision_bars


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate decision bars")
    parser.add_argument("path", help="Path to 1-minute OHLCV parquet")
    parser.add_argument("--target-hours", type=int, default=4, help="Target median decision horizon in hours")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ohlcv = pd.read_parquet(args.path)
    config = DecisionBarConfig(target_median_seconds=args.target_hours * 3600)
    result = build_decision_bars(ohlcv, config)
    print(f"Method: {result.method}")
    print(result.bars.head())


if __name__ == "__main__":
    main()
