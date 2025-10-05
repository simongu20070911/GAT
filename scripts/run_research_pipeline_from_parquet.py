"""Run research pipeline on explicit parquet paths."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from candlestrats.bars import DecisionBarConfig
from candlestrats.data import MinuteBarStore
from candlestrats.data.ingestion import BinanceIngestionSpec
from candlestrats.labeling import TripleBarrierConfig
from candlestrats.pipeline import PipelineConfig, run_pipeline
from candlestrats.features.motifs import MotifSpec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pipeline on local parquet")
    parser.add_argument("parquet", type=Path)
    parser.add_argument("symbol")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def make_store(parquet: Path, symbol: str) -> MinuteBarStore:
    spec = BinanceIngestionSpec(symbol=symbol)
    spec.raw_path = lambda: parquet  # type: ignore[attr-defined]
    return MinuteBarStore(specs=[spec])


def main() -> None:
    args = parse_args()
    store = make_store(args.parquet, args.symbol)
    pipeline_config = PipelineConfig(
        symbol=args.symbol,
        bar=DecisionBarConfig(),
        triple_barrier=TripleBarrierConfig(),
        motif=MotifSpec(window=24, n_clusters=6),
    )
    result = run_pipeline(store, pipeline_config)
    args.output.mkdir(parents=True, exist_ok=True)
    result.decision_bars.to_parquet(args.output / f"{args.symbol}_decision_bars.parquet")
    result.labels.to_parquet(args.output / f"{args.symbol}_labels.parquet")
    result.features.to_parquet(args.output / f"{args.symbol}_features.parquet")
    print(f"{args.symbol}: {len(result.decision_bars)} bars")


if __name__ == "__main__":
    main()
