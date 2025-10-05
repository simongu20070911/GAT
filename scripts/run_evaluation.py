"""Evaluate strategies with CPCV and DSR/PBO metrics."""
from __future__ import annotations

import argparse

import pandas as pd

from candlestrats.evaluation import (
    CombinatorialPurgedCV,
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
    evaluate_strategy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evaluation harness")
    parser.add_argument("returns", help="Path to returns parquet")
    parser.add_argument("--splits", type=int, default=6, help="Number of CPCV splits")
    parser.add_argument("--embargo", type=int, default=60, help="Embargo in minutes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    returns = pd.read_parquet(args.returns)["returns"]
    cv = CombinatorialPurgedCV(n_splits=args.splits, embargo_minutes=args.embargo)
    folds = evaluate_strategy(returns, cv)
    sharpes = [fold.sharpe for fold in folds]
    values = returns.to_numpy()
    n_obs = len(values)
    if n_obs > 1:
        mean = values.mean()
        std = values.std(ddof=1)
        sharpe_total = mean / std if std > 0 else float("nan")
        demeaned = values - mean
        skewness = (demeaned**3).mean() / (std**3) if std > 0 else 0.0
        kurtosis = (demeaned**4).mean() / (std**4) if std > 0 else 3.0
    else:
        sharpe_total = float("nan")
        skewness = 0.0
        kurtosis = 3.0

    trials = max(len(sharpes), 1)
    dsr = deflated_sharpe_ratio(sharpe_total, n_obs, trials, skewness=skewness, kurtosis=kurtosis)
    pbo = probability_of_backtest_overfitting(sharpes)
    print(f"DSR={dsr:.4f}, PBO={pbo:.4f}")


if __name__ == "__main__":
    main()
