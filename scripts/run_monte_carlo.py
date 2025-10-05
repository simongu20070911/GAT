"""Run Monte Carlo leakage check on strategy returns."""
from __future__ import annotations

import argparse

import pandas as pd

from candlestrats.evaluation import CombinatorialPurgedCV, monte_carlo_null_distribution, permutation_mcpt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monte Carlo leakage check")
    parser.add_argument("returns", help="Path to returns parquet with 'returns' column")
    parser.add_argument("--splits", type=int, default=6, help="Number of CPCV splits")
    parser.add_argument("--embargo", type=int, default=60, help="Embargo in minutes")
    parser.add_argument("--runs", type=int, default=100, help="Number of Monte Carlo runs")
    parser.add_argument("--mode", choices=["random", "permutation"], default="random", help="Null model to use")
    parser.add_argument("--block-size", type=int, default=24, help="Block size for permutation MCPT")
    parser.add_argument("--seed", type=int, help="Optional RNG seed for reproducible null draws")
    return parser.parse_args()


def _load_returns(path: str) -> pd.Series:
    frame = pd.read_parquet(path)
    if "ts" in frame.columns:
        index = pd.to_datetime(frame["ts"], utc=True)
    elif "timestamp" in frame.columns:
        index = pd.to_datetime(frame["timestamp"], utc=True)
    elif isinstance(frame.index, pd.DatetimeIndex):
        index = frame.index
    else:
        raise TypeError("Returns parquet must include a 'ts'/'timestamp' column or DatetimeIndex")
    series = pd.Series(frame["returns"].to_numpy(), index=index, name="returns")
    return series.sort_index()


def main() -> None:
    args = parse_args()
    returns = _load_returns(args.returns)
    cv = CombinatorialPurgedCV(n_splits=args.splits, embargo_minutes=args.embargo)
    if args.mode == "random":
        result = monte_carlo_null_distribution(returns, cv, runs=args.runs, random_state=args.seed)
    else:
        result = permutation_mcpt(
            returns,
            cv,
            block_size=args.block_size,
            runs=args.runs,
            random_state=args.seed,
        )
    print(f"Real Sharpe: {result.real_metric:.4f}")
    mean_mc = sum(result.synthetic_metrics) / len(result.synthetic_metrics) if result.synthetic_metrics else float("nan")
    print(f"Null mean: {mean_mc:.4f}")
    print(f"p-value: {result.p_value:.4f}")


if __name__ == "__main__":
    main()
