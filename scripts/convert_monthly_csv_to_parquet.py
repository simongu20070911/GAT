"""Convert Binance monthly CSVs to parquet for smoke runs."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert monthly CSV to parquet")
    parser.add_argument("csv", type=Path, help="Path to Binance monthly CSV (1m)")
    parser.add_argument("--output", type=Path, required=True, help="Destination parquet path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)
    df.rename(
        columns={
            "open_time": "open_time",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        },
        inplace=True,
    )
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    out_df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.output, index=False)
    print(f"Wrote parquet to {args.output}")


if __name__ == "__main__":
    main()
