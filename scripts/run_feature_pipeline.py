"""Execute morphology and motif feature pipelines."""
from __future__ import annotations

import argparse

import pandas as pd

from candlestrats.features import compute_morphology_features, discover_motifs, encode_motif_hits
from candlestrats.features.motifs import MotifSpec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run feature pipeline")
    parser.add_argument("path", help="Path to decision bar parquet")
    parser.add_argument("--motif-window", type=int, default=24, help="Lookback window for motifs")
    parser.add_argument("--motif-clusters", type=int, default=8, help="Number of motif clusters")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bars = pd.read_parquet(args.path)
    features = compute_morphology_features(bars.rename(columns={"ts": "timestamp"}))
    spec = MotifSpec(window=args.motif_window, n_clusters=args.motif_clusters)
    motifs = discover_motifs(features, spec)
    hits = encode_motif_hits(motifs, universe=range(args.motif_clusters))
    print(features.head())
    print(hits.head())


if __name__ == "__main__":
    main()
