"""Run the genetic programming search scaffold."""
from __future__ import annotations

import argparse

import pandas as pd

from candlestrats.gp import GeneticProgramConfig, GeneticProgramMiner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GP search")
    parser.add_argument("features", help="Path to feature parquet")
    parser.add_argument("labels", help="Path to label parquet")
    parser.add_argument("--generations", type=int, default=10, help="Number of GP generations")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_frame = pd.read_parquet(args.features)
    label_frame = pd.read_parquet(args.labels)
    ts_col = feature_frame.get("ts")
    if ts_col is not None:
        ts_index = pd.to_datetime(ts_col, utc=True)
        feature_frame = feature_frame.drop(columns=["ts", "timestamp"], errors="ignore")
        feature_frame.index = ts_index
        feature_frame.index.name = "ts"
    else:
        feature_frame = feature_frame.drop(columns=["timestamp"], errors="ignore")
    feature_cols = list(feature_frame.columns)
    label_ts_col = label_frame.get("ts")
    if label_ts_col is not None:
        label_ts = pd.to_datetime(label_ts_col, utc=True)
        labels = label_frame.set_index(label_ts, drop=False)
        labels.index.name = "ts"
    else:
        labels = label_frame.set_index("bar_id")
    miner = GeneticProgramMiner(
        GeneticProgramConfig(
            n_generations=args.generations,
            feature_columns=tuple(feature_cols),
        )
    )
    best = miner.evolve(feature_frame[feature_cols], labels)
    print("Best rule:", best.expression)
    print("Fitness:", best.fitness)


if __name__ == "__main__":
    main()
