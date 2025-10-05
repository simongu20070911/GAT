"""Run evolution over a continuous window across multiple months."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
import full_run_pipeline as full_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full window candlestick pipeline")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start", required=True, help="ISO date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="ISO date YYYY-MM-DD (inclusive)")
    parser.add_argument("--output-root", type=Path, default=full_run.OUTPUT_ROOT)
    parser.add_argument("--label", help="Optional label for output directory")
    parser.add_argument("--population", type=int, default=16, help="GP population size")
    parser.add_argument("--generations", type=int, default=4, help="GP generations")
    return parser.parse_args()


def months_between(start: str, end: str) -> List[str]:
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    months = []
    year = start_dt.year
    month = start_dt.month
    while (year, month) <= (end_dt.year, end_dt.month):
        months.append(f"{year:04d}-{month:02d}")
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1
    return months


def main() -> None:
    args = parse_args()
    symbol = args.symbol
    months = months_between(args.start, args.end)
    frames = []
    for m in months:
        try:
            frame = full_run.load_monthly_data(symbol, m)
        except FileNotFoundError:
            continue
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("No data found for requested window")
    combined = pd.concat(frames, ignore_index=True)
    label = args.label or f"{args.start}_to_{args.end}"
    full_run.run_stage_pipeline(symbol, label, combined, args.output_root, population_size=args.population, n_generations=args.generations)


if __name__ == "__main__":
    main()
