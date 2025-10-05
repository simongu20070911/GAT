"""Aggregate MCPT diagnostics into scorecard outputs."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from candlestrats.evaluation.reporting import MCPTSummaryInput, mcpt_summary, write_mcpt_summary
from candlestrats.evaluation import resolve_fee_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate MCPT diagnostics")
    parser.add_argument("returns", help="Path to returns parquet with column 'returns'")
    parser.add_argument("--symbol", required=True, help="Symbol identifier")
    parser.add_argument("--splits", type=int, default=6)
    parser.add_argument("--embargo", type=int, default=60)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--block-size", type=int, default=24)
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
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/stage_02_priors/scorecard/mcpt_summary.csv"),
        help="Output CSV path",
    )
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
    summary_df = mcpt_summary(
        MCPTSummaryInput(
            returns=returns,
            symbol=args.symbol,
            splits=args.splits,
            embargo_minutes=args.embargo,
            runs=args.runs,
            permutation_block=args.block_size,
        )
    )
    fee_model, fee_tier = resolve_fee_model(
        args.fee_venue,
        args.fee_market,
        symbol=args.symbol,
        tier=args.fee_tier,
        fees_config=args.fees_config if args.fees_config else None,
        strategy_config=args.strategy_fees if args.strategy_fees else None,
    )
    summary_df["fee_maker_bps"] = fee_model.maker_bps
    summary_df["fee_taker_bps"] = fee_model.taker_bps
    summary_df["fee_tier"] = fee_tier
    write_mcpt_summary(args.output, summary_df)
    print(f"Wrote MCPT summary for {args.symbol} to {args.output}")


if __name__ == "__main__":
    main()
