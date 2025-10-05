"""Execute end-to-end candlestick research pipeline for configured symbols."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from candlestrats.bars import DecisionBarConfig
from candlestrats.data import MinuteBarStore, build_default_store
from candlestrats.features.motifs import MotifSpec
from candlestrats.labeling import TripleBarrierConfig
from candlestrats.pipeline import PipelineConfig, run_pipeline
from candlestrats.evaluation import resolve_fee_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run end-to-end research pipeline")
    parser.add_argument("symbols", nargs="+", help="Symbols to process")
    parser.add_argument("--start", type=str, help="ISO8601 start timestamp")
    parser.add_argument("--end", type=str, help="ISO8601 end timestamp")
    parser.add_argument("--output", type=Path, help="Optional output directory for artifacts")
    parser.add_argument("--fee-venue", default="binance", help="Fee venue key in fees.yaml")
    parser.add_argument("--fee-market", default="futures_usdm", help="Fee market key")
    parser.add_argument("--fee-tier", help="Override fee tier")
    parser.add_argument(
        "--fees-config",
        type=Path,
        default=None,
        help="Optional path to base fee schedule YAML (fees.yaml)",
    )
    parser.add_argument(
        "--strategy-fees",
        "--fee-config",
        dest="strategy_fees",
        type=Path,
        default=None,
        help="Optional path to strategy fee overrides YAML",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = build_default_store(args.symbols)
    store = MinuteBarStore(specs=specs)
    output_dir = args.output

    fees_config = args.fees_config
    strategy_config = args.strategy_fees
    for symbol in args.symbols:
        fee_model, fee_tier = resolve_fee_model(
            args.fee_venue,
            args.fee_market,
            symbol=symbol,
            tier=args.fee_tier,
            fees_config=fees_config if fees_config else None,
            strategy_config=strategy_config if strategy_config else None,
        )
        fee_payload = {
            "maker_bps": fee_model.maker_bps,
            "taker_bps": fee_model.taker_bps,
            "half_spread_bps": fee_model.half_spread_bps,
            "impact_perc_coeff": fee_model.impact_perc_coeff,
            "tier": fee_tier,
        }

        pipeline_config = PipelineConfig(
            symbol=symbol,
            bar=DecisionBarConfig(),
            triple_barrier=TripleBarrierConfig(),
            motif=MotifSpec(window=24, n_clusters=6),
            start=pd.Timestamp(args.start) if args.start else None,
            end=pd.Timestamp(args.end) if args.end else None,
        )
        result = run_pipeline(store, pipeline_config)
        print(f"{symbol}: {len(result.decision_bars)} decision bars, {len(result.labels)} labels | fee tier {fee_payload['tier']}")
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            result.decision_bars.to_parquet(output_dir / f"{symbol}_decision_bars.parquet")
            result.labels.to_parquet(output_dir / f"{symbol}_labels.parquet")
            result.features.to_parquet(output_dir / f"{symbol}_features.parquet")
            meta_path = output_dir / f"{symbol}_metadata.json"
            meta_path.write_text(json.dumps({
                "symbol": symbol,
                "fee": fee_payload,
                "records": {
                    "decision_bars": len(result.decision_bars),
                    "labels": len(result.labels),
                    "features": len(result.features),
                },
            }, indent=2))


if __name__ == "__main__":
    main()
