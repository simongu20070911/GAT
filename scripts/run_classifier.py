"""Train the candlestick state classifier scaffold."""
from __future__ import annotations

import argparse

import pandas as pd

from candlestrats.classifiers import CandlestickStateClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train candlestick state classifier")
    parser.add_argument("features", help="Path to feature parquet")
    parser.add_argument("labels", help="Path to label parquet")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_frame = pd.read_parquet(args.features)
    label_frame = pd.read_parquet(args.labels)
    clf = CandlestickStateClassifier()
    clf.fit(feature_frame, label_frame["y"])
    preds = clf.predict(feature_frame)
    print(preds.head())


if __name__ == "__main__":
    main()
