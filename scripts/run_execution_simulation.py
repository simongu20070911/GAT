"""Run execution simulation on an orders file."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from candlestrats.evaluation import resolve_fee_model
from candlestrats.simulation import ExecutionSimulator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate fills with fee/slippage model")
    parser.add_argument("orders", type=Path, help="Path to orders parquet or csv")
    parser.add_argument("--output", type=Path, help="Optional output parquet path")
    parser.add_argument("--symbol", required=True, help="Symbol for fee overrides")
    parser.add_argument("--fee-venue", default="binance")
    parser.add_argument("--fee-market", default="futures_usdm")
    parser.add_argument("--fee-tier")
    parser.add_argument(
        "--fees-config",
        type=Path,
        default=None,
        help="Path to base fee schedule YAML",
    )
    parser.add_argument(
        "--strategy-fees",
        "--fee-config",
        dest="strategy_fees",
        type=Path,
        default=None,
        help="Path to strategy fee overrides YAML",
    )
    parser.add_argument("--queue-bps", type=float, default=1.0)
    parser.add_argument("--impact-coeff", type=float, default=0.1, help="Impact bps per unit size fraction")
    return parser.parse_args()


def read_orders(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path, parse_dates=["timestamp"])
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported orders format: {path}")


def main() -> None:
    args = parse_args()
    fee_model, fee_tier = resolve_fee_model(
        args.fee_venue,
        args.fee_market,
        symbol=args.symbol,
        tier=args.fee_tier,
        fees_config=args.fees_config if args.fees_config else None,
        strategy_config=args.strategy_fees if args.strategy_fees else None,
    )
    orders = read_orders(args.orders)
    simulator = ExecutionSimulator(
        queue_penalty_bps=args.queue_bps, impact_coeff=args.impact_coeff, fee_model=fee_model
    )
    fills = simulator.simulate(orders)
    fills["fee_tier"] = fee_tier
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.suffix == ".csv":
            fills.to_csv(args.output, index=False)
        else:
            fills.to_parquet(args.output)
        print(f"Wrote fills to {args.output}")
    else:
        print(fills.head())


if __name__ == "__main__":
    main()
