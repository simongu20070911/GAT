"""Nightly motif mining runner per cluster."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from candlestrats.motifs import mine_motifs_from_bars, motifs_to_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nightly motif mining")
    parser.add_argument("bars", type=Path, help="Decision bars parquet path")
    parser.add_argument("labels", type=Path, help="Labels parquet/CSV")
    parser.add_argument("--spec-config", type=Path, default=Path("config/research/predicate_specs.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/stage_02_priors/motifs"))
    parser.add_argument("--min-support", type=float, default=0.05)
    parser.add_argument("--min-lift", type=float, default=1.1)
    parser.add_argument("--max-size", type=int, default=3)
    return parser.parse_args()


def load_labels(path: Path) -> pd.Series:
    if path.suffix == ".csv":
        df = pd.read_csv(path, parse_dates=[0])
        if "y" in df.columns:
            return df.set_index(df.columns[0])["y"]
        return df.set_index(df.columns[0]).iloc[:, 0]
    df = pd.read_parquet(path)
    if "y" in df.columns:
        return df.set_index(df.columns[0])["y"] if df.columns[0] != "y" else df["y"]
    return df.iloc[:, 0]


def main() -> None:
    args = parse_args()
    bars = pd.read_parquet(args.bars)
    labels = load_labels(args.labels)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    motifs = mine_motifs_from_bars(
        bars,
        labels,
        min_support=args.min_support,
        min_lift=args.min_lift,
        max_size=args.max_size,
        spec_config=args.spec_config,
    )
    frame = motifs_to_frame(motifs)
    output = args.output_dir / f"motifs_{Path(args.bars).stem}.csv"
    frame.to_csv(output, index=False)
    print(f"Wrote {len(frame)} motifs to {output}")


if __name__ == "__main__":
    main()
