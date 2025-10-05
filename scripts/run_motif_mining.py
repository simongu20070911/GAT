"""Mine frequent motif atoms from candlestick bars."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from candlestrats.motifs import mine_motifs_from_bars, motifs_to_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine frequent candlestick motifs")
    parser.add_argument("bars", type=Path, help="Path to decision bars parquet")
    parser.add_argument("labels", type=Path, help="Path to label parquet/CSV with column 'y'")
    parser.add_argument("--output", type=Path, help="Output CSV path for motifs")
    parser.add_argument("--min-support", type=float, default=0.05)
    parser.add_argument("--min-lift", type=float, default=1.2)
    parser.add_argument("--max-size", type=int, default=3)
    parser.add_argument("--spec-config", type=Path, default=None, help="Predicate spec YAML override")
    return parser.parse_args()


def load_labels(path: Path) -> pd.Series:
    if path.suffix == ".csv":
        df = pd.read_csv(path, parse_dates=[0])
        if "y" in df.columns:
            return df.set_index(df.columns[0])["y"]
        return df.set_index(df.columns[0]).iloc[:, 0]
    df = pd.read_parquet(path)
    if "y" in df.columns:
        series = df.set_index(df.columns[0])["y"] if df.columns[0] != "y" else df["y"]
    else:
        series = df.iloc[:, 0]
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    return series


def main() -> None:
    args = parse_args()
    bars = pd.read_parquet(args.bars)
    labels = load_labels(args.labels)
    motifs = mine_motifs_from_bars(
        bars,
        labels,
        min_support=args.min_support,
        min_lift=args.min_lift,
        max_size=args.max_size,
        spec_config=args.spec_config,
    )
    frame = motifs_to_frame(motifs)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output, index=False)
        print(f"Wrote {len(frame)} motifs to {args.output}")
    else:
        print(frame.head())


if __name__ == "__main__":
    main()
