"""Batch MCPT scorecard builder for nightly diagnostics."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from candlestrats.evaluation import resolve_fee_model
from candlestrats.evaluation.reporting import MCPTSummaryInput, mcpt_summary, write_mcpt_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate MCPT summary across return files")
    parser.add_argument(
        "returns_dir", type=Path, default=Path("reports/stage_02_priors/returns"), help="Directory of returns parquet files"
    )
    parser.add_argument(
        "--pattern", default="*.parquet", help="Glob pattern within returns_dir to match returns files"
    )
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
        help="Destination CSV",
    )
    return parser.parse_args()


def derive_symbol(path: Path) -> str:
    name = path.stem
    return name.split("_")[0]


def _load_returns(path: Path) -> pd.Series:
    frame = pd.read_parquet(path)
    if "ts" in frame.columns:
        index = pd.to_datetime(frame["ts"], utc=True)
    elif "timestamp" in frame.columns:
        index = pd.to_datetime(frame["timestamp"], utc=True)
    elif isinstance(frame.index, pd.DatetimeIndex):
        index = frame.index
    else:
        raise TypeError(
            f"Returns parquet {path} must include a 'ts'/'timestamp' column or DatetimeIndex"
        )
    series = pd.Series(frame["returns"].to_numpy(), index=index, name="returns")
    return series.sort_index()


def main() -> None:
    args = parse_args()
    returns_paths = sorted(args.returns_dir.glob(args.pattern))
    if not returns_paths:
        raise FileNotFoundError(f"No returns files matching {args.pattern} in {args.returns_dir}")

    summaries = []
    for returns_path in returns_paths:
        symbol = derive_symbol(returns_path)
        fee_model, fee_tier = resolve_fee_model(
            args.fee_venue,
            args.fee_market,
            symbol=symbol,
            tier=args.fee_tier,
            fees_config=args.fees_config if args.fees_config else None,
            strategy_config=args.strategy_fees if args.strategy_fees else None,
        )
        returns = _load_returns(returns_path)
        summary = mcpt_summary(
            MCPTSummaryInput(
                returns=returns,
                symbol=symbol,
                splits=args.splits,
                embargo_minutes=args.embargo,
                runs=args.runs,
                permutation_block=args.block_size,
            )
        )
        summary["fee_maker_bps"] = fee_model.maker_bps
        summary["fee_taker_bps"] = fee_model.taker_bps
        summary["fee_tier"] = fee_tier
        summaries.append(summary)

    final = pd.concat(summaries, ignore_index=True)
    write_mcpt_summary(args.output, final)
    print(f"Aggregated {len(final)} entries to {args.output}")


if __name__ == "__main__":
    main()
