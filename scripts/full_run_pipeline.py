"""Batch controller for full candlestick pipeline across symbols/months."""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from candlestrats.bars import DecisionBarConfig
from candlestrats.cluster.embeddings import PatternStats, build_feature_matrix
from candlestrats.data import MinuteBarStore
from candlestrats.data.ingestion import BinanceIngestionSpec
from candlestrats.evaluation import CombinatorialPurgedCV, FeeModel, resolve_fee_model
from candlestrats.features.motifs import MotifSpec
from candlestrats.gp import GeneticProgramConfig, GeneticProgramMiner
from candlestrats.gp.evaluation import StrategyEvaluator
from candlestrats.gp.miner import CandidateRule
from candlestrats.labeling import TripleBarrierConfig
from candlestrats.motifs import mine_motifs_from_bars, motifs_to_frame
from candlestrats.motifs.pipeline import DEFAULT_SPEC_PATH
from candlestrats.pipeline import PipelineConfig, run_pipeline
from candlestrats.utils.monte_carlo import generate_random_walk_ohlcv

OUTPUT_ROOT = Path("/mnt/timemachine/binance/features/full_run")
MONTHLY_ROOT = Path("/mnt/timemachine/binance/futures/um/monthly/klines")
SYNTHETIC_RUNS = 5
SYNTHETIC_THRESHOLD = 0.3
SYNTHETIC_RATIO = 0.75
SYNTHETIC_HITS_REQUIRED = 2
EVALUATOR_MIN_TRADES = 30
SYNTHETIC_SEED = 42


logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)

def extract_tokens(expression: str) -> list[str]:
    tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expression))
    return sorted(token for token in tokens if not token.replace('_', '').isdigit())


@dataclass
class BatchResult:
    symbol: str
    month: str
    output_dir: Path
    motifs_count: int
    gp_rules: int


def list_symbols_with_month(month: str, limit: int | None = None) -> list[str]:
    symbols = []
    for p in sorted(MONTHLY_ROOT.iterdir()):
        if not p.is_dir():
            continue
        csv_path = p / '1m' / f"{p.name}-1m-{month}.csv"
        if csv_path.exists():
            symbols.append(p.name)
    return symbols[:limit] if limit else symbols


def load_monthly_data(symbol: str, month: str) -> pd.DataFrame:
    csv_path = MONTHLY_ROOT / symbol / "1m" / f"{symbol}-1m-{month}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing monthly CSV {csv_path}")
    frame = pd.read_csv(csv_path, header=None)
    if frame.shape[1] >= 6 and frame.columns[0] == 0:
        frame.columns = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"][: frame.shape[1]]
        if isinstance(frame.loc[0, 'open_time'], str) and frame.loc[0, 'open_time'].lower() == 'open_time':
            frame = frame.drop(index=0).reset_index(drop=True)
    if "open_time" not in frame.columns:
        frame = pd.read_csv(csv_path)
    frame["open_time"] = pd.to_numeric(frame["open_time"], errors='coerce')
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors='coerce')
    frame["timestamp"] = pd.to_datetime(frame["open_time"], errors='coerce', unit="ms", utc=True)
    frame = frame.dropna(subset=['timestamp'])
    frame = frame.dropna(subset=numeric_cols)
    return frame[["timestamp", "open", "high", "low", "close", "volume"]]



def _finite_or_none(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None

def _sanitize_metadata(meta):
    sanitized = {}
    for key, value in (meta or {}).items():
        if isinstance(value, (int, float)):
            clean = _finite_or_none(value)
            if clean is None:
                continue
            sanitized[key] = clean
        else:
            sanitized[key] = value
    return sanitized


def run_stage_pipeline(symbol: str, month: str, frame: pd.DataFrame, output_root: Path, *, population_size: int = 16, n_generations: int = 4) -> BatchResult:
    output_dir = output_root / symbol / month
    output_dir.mkdir(parents=True, exist_ok=True)

    spec = BinanceIngestionSpec(symbol=symbol)
    spec.raw_path = lambda frame=frame: frame  # type: ignore[attr-defined]

    class InMemoryStore(MinuteBarStore):
        def __init__(self, symbol: str, frame: pd.DataFrame) -> None:
            spec = BinanceIngestionSpec(symbol=symbol)
            spec.raw_path = lambda: "<in-memory>"
            super().__init__([spec])
            self._frame = frame

        def load(self, symbol: str, start=None, end=None) -> pd.DataFrame:  # type: ignore[override]
            data = self._frame.copy()
            if start:
                ts_start = pd.Timestamp(start)
                if ts_start.tzinfo is None:
                    ts_start = ts_start.tz_localize("UTC")
                else:
                    ts_start = ts_start.tz_convert("UTC")
                data = data[data["timestamp"] >= ts_start]
            if end:
                ts_end = pd.Timestamp(end)
                if ts_end.tzinfo is None:
                    ts_end = ts_end.tz_localize("UTC")
                else:
                    ts_end = ts_end.tz_convert("UTC")
                data = data[data["timestamp"] <= ts_end]
            return data.reset_index(drop=True)

    store = InMemoryStore(symbol, frame)
    pipeline_config = PipelineConfig(
        symbol=symbol,
        bar=DecisionBarConfig(),
        triple_barrier=TripleBarrierConfig(),
        motif=MotifSpec(window=24, n_clusters=8),
    )
    result = run_pipeline(store, pipeline_config)

    try:
        fee_model, _ = resolve_fee_model("binance", "futures_usdm", symbol=symbol)
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("Falling back to default fee model for %s: %s", symbol, exc)
        fee_model = FeeModel()

    bars_path = output_dir / f"{symbol}_decision_bars.parquet"
    labels_path = output_dir / f"{symbol}_labels.parquet"
    features_path = output_dir / f"{symbol}_features.parquet"
    result.decision_bars.to_parquet(bars_path)
    result.labels.to_parquet(labels_path)
    result.features.to_parquet(features_path)

    motifs = mine_motifs_from_bars(
        result.decision_bars,
        result.labels.set_index("bar_id")["y"],
        spec_config=DEFAULT_SPEC_PATH,
        min_support=0.05,
        min_lift=0.9,
        max_size=2,
    )
    motifs_frame = motifs_to_frame(motifs)
    motifs_frame["symbol"] = symbol
    motifs_frame.to_csv(output_dir / f"motifs_{symbol}_{month}.csv", index=False)

    feature_ts = pd.to_datetime(result.features.get("ts"), utc=True)
    feature_df = result.features.drop(columns=["ts"], errors="ignore").copy()
    feature_df.index = feature_ts
    feature_df.index.name = "ts"
    feature_cols = list(feature_df.columns)
    label_ts = pd.to_datetime(result.labels.get("ts"), utc=True)
    label_frame = result.labels.set_index(label_ts, drop=False)
    label_frame.index.name = "ts"
    feature_df, label_frame = feature_df.align(label_frame, join="inner", axis=0)

    gp_config = GeneticProgramConfig(
        population_size=population_size,
        n_generations=n_generations,
        feature_columns=tuple(feature_cols),
        turnover_weight=0.5,
        breadth_weight=1.0,
    )
    cv = CombinatorialPurgedCV(n_splits=3, embargo_minutes=720)
    gp_trials = max(gp_config.population_size * max(gp_config.n_generations, 1), 1)
    evaluator = StrategyEvaluator(
        cv=cv,
        fee_model=fee_model,
        min_trades=EVALUATOR_MIN_TRADES,
        trials=gp_trials,
    )
    miner = GeneticProgramMiner(gp_config, strategy_evaluator=evaluator)
    population, best_real = miner.evolve_population(feature_df, label_frame)

    population = list(population)
    if best_real is not None:
        # Ensure the best rule is part of the evaluated population with up-to-date metadata.
        population = [rule for rule in population if rule.expression != best_real.expression]
        population.append(best_real)
        best_real_fitness = float(best_real.fitness) if best_real.fitness is not None else float("-inf")
    else:
        best_real_fitness = float("-inf")

    synthetic_rng = np.random.default_rng(SYNTHETIC_SEED)

    synthetic_length = max(len(label_frame), 10)
    synthetic_frame = generate_random_walk_ohlcv(synthetic_length, random_state=synthetic_rng)
    if "timestamp" in synthetic_frame.columns:
        if synthetic_frame["timestamp"].dt.tz is None:
            synthetic_frame["timestamp"] = synthetic_frame["timestamp"].dt.tz_localize("UTC")
    synthetic_frame = synthetic_frame.rename(columns={"ts": "timestamp"})

    synthetic_store = InMemoryStore(symbol, synthetic_frame)
    synthetic_result = run_pipeline(synthetic_store, pipeline_config)
    synthetic_feature_ts = pd.to_datetime(synthetic_result.features.get("ts"), utc=True)
    synthetic_features = (
        synthetic_result.features.drop(columns=["ts"], errors="ignore")
        .reindex(columns=feature_cols)
    )
    synthetic_features.index = synthetic_feature_ts
    synthetic_features.index.name = "ts"
    synthetic_label_ts = pd.to_datetime(synthetic_result.labels.get("ts"), utc=True)
    synthetic_labels = synthetic_result.labels.set_index(synthetic_label_ts, drop=False)
    synthetic_labels.index.name = "ts"
    synthetic_features, synthetic_labels = synthetic_features.align(synthetic_labels, join="inner", axis=0)
    synthetic_fits: dict[str, list[float]] = defaultdict(list)
    synthetic_trade_counts: dict[str, list[int]] = defaultdict(list)
    synthetic_meta: dict[str, dict[str, float]] = {}

    synthetic_rules = [CandidateRule(expression=rule.expression) for rule in population]
    synthetic_population = miner.evaluate(synthetic_rules, synthetic_features, synthetic_labels)
    pairs = list(zip(population, synthetic_population))

    for real_rule, synth_rule in pairs:
        real_fit = float(real_rule.fitness) if real_rule.fitness is not None else float("-inf")
        synth_fit = float(synth_rule.fitness) if synth_rule.fitness is not None else float("-inf")
        synthetic_fits[real_rule.expression].append(synth_fit)
        trade_count = int((synth_rule.metadata or {}).get("trade_count", 0))
        synthetic_trade_counts[real_rule.expression].append(trade_count)
        if synth_rule.metadata:
            synthetic_meta.setdefault(real_rule.expression, synth_rule.metadata)

    # Additional synthetic runs to reduce false positives
    for _ in range(1, SYNTHETIC_RUNS):
        synthetic_length = max(len(label_frame), 10)
        synthetic_frame_iter = generate_random_walk_ohlcv(synthetic_length, random_state=synthetic_rng)
        if "timestamp" in synthetic_frame_iter.columns and synthetic_frame_iter["timestamp"].dt.tz is None:
            synthetic_frame_iter["timestamp"] = synthetic_frame_iter["timestamp"].dt.tz_localize("UTC")
        synthetic_frame_iter = synthetic_frame_iter.rename(columns={"ts": "timestamp"})
        synthetic_store_iter = InMemoryStore(symbol, synthetic_frame_iter)
        synthetic_result_iter = run_pipeline(synthetic_store_iter, pipeline_config)
        synth_iter_feature_ts = pd.to_datetime(synthetic_result_iter.features.get("ts"), utc=True)
        synthetic_features_iter = (
            synthetic_result_iter.features.drop(columns=["ts"], errors="ignore")
            .reindex(columns=feature_cols)
        )
        synthetic_features_iter.index = synth_iter_feature_ts
        synthetic_features_iter.index.name = "ts"
        synth_iter_label_ts = pd.to_datetime(synthetic_result_iter.labels.get("ts"), utc=True)
        synthetic_labels_iter = synthetic_result_iter.labels.set_index(synth_iter_label_ts, drop=False)
        synthetic_labels_iter.index.name = "ts"
        synthetic_features_iter, synthetic_labels_iter = synthetic_features_iter.align(synthetic_labels_iter, join="inner", axis=0)
        synthetic_rules_iter = [CandidateRule(expression=rule.expression) for rule in population]
        synthetic_population_iter = miner.evaluate(synthetic_rules_iter, synthetic_features_iter, synthetic_labels_iter)
        for real_rule, synth_rule in zip(population, synthetic_population_iter):
            fit = float(synth_rule.fitness) if synth_rule and synth_rule.fitness is not None else float("-inf")
            synthetic_fits[real_rule.expression].append(fit)
            trade_count = int((synth_rule.metadata or {}).get("trade_count", 0))
            synthetic_trade_counts[real_rule.expression].append(trade_count)
            if synth_rule.metadata:
                synthetic_meta.setdefault(real_rule.expression, synth_rule.metadata)

    max_synth_overall = float("-inf")
    for fits in synthetic_fits.values():
        finite = [fit for fit in fits if np.isfinite(fit)]
        if finite:
            candidate = max(finite)
            if candidate > max_synth_overall:
                max_synth_overall = candidate
    max_synth_positive = float("-inf")
    leak_rule: dict[str, float | str] | None = None
    for real_rule in population:
        real_fit = float(real_rule.fitness) if real_rule.fitness is not None else float("-inf")
        if not (np.isfinite(real_fit) and real_fit > 0):
            continue
        real_meta = real_rule.metadata or {}
        real_trades = int(real_meta.get("trade_count", 0))
        if real_trades < EVALUATOR_MIN_TRADES:
            continue
        fits = synthetic_fits.get(real_rule.expression, [])
        counts = synthetic_trade_counts.get(real_rule.expression, [])
        finite_pairs = [
            (fit, count)
            for fit, count in zip(fits, counts)
            if np.isfinite(fit)
        ]
        if finite_pairs:
            finite_values = [fit for fit, _ in finite_pairs]
            candidate_max = max(finite_values)
            if candidate_max > max_synth_positive:
                max_synth_positive = candidate_max
            qualified_hits = [
                fit
                for fit, count in finite_pairs
                if count >= EVALUATOR_MIN_TRADES
                and fit >= SYNTHETIC_THRESHOLD
                and (real_fit > 0 and fit / real_fit >= SYNTHETIC_RATIO)
                and (real_trades > 0 and count / real_trades <= 2.0)
            ]
            hits = len(qualified_hits)
            if hits >= SYNTHETIC_HITS_REQUIRED and real_fit >= 0.5:
                leak_rule = {
                    "expression": real_rule.expression,
                    "real_fitness": real_fit,
                    "synthetic_mean": float(np.mean(finite_values)),
                    "synthetic_max": float(candidate_max),
                    "hits_above_threshold": hits,
                    "ratio_max": float(candidate_max / real_fit) if real_fit else float("inf"),
                    "runs": len(finite_pairs),
                }
                break

    leak_flag = leak_rule is not None

    synthetic_summary = {
        "symbol": symbol,
        "month": month,
        "real_best_fitness": _finite_or_none(best_real_fitness),
        "synthetic_max_positive_fitness": _finite_or_none(max_synth_positive),
        "synthetic_max_overall_fitness": _finite_or_none(max_synth_overall),
        "synthetic_runs": SYNTHETIC_RUNS,
        "threshold_hits_required": SYNTHETIC_HITS_REQUIRED,
        "leak_rule": leak_rule,
        "flagged": leak_flag,
        "real_best_metadata": best_real.metadata if best_real else {},
        "synthetic_best_metadata": synthetic_meta.get(leak_rule["expression"], {}) if leak_rule else {},
    }

    (output_dir / f"synthetic_check_{symbol}_{month}.json").write_text(json.dumps(synthetic_summary, indent=2, allow_nan=False))

    stats = []
    rule_records = []
    for idx, rule in enumerate(population):
        meta = rule.metadata or {}
        metrics = {k: float(v) for k, v in meta.items() if isinstance(v, (int, float)) and math.isfinite(float(v))}
        tokens = extract_tokens(rule.expression) + list(meta.keys())
        stats.append(PatternStats(name=f"{symbol}_{month}_rule_{idx}", items=tuple(tokens), metrics=metrics))
        rule_records.append({
            'name': f"{symbol}_{month}_rule_{idx}",
            'expression': rule.expression,
            'fitness': _finite_or_none(rule.fitness),
            'metadata': _sanitize_metadata(meta),
        })

    gp_matrix = build_feature_matrix(stats)
    gp_matrix.to_csv(output_dir / f"gp_matrix_{symbol}_{month}.csv")
    (output_dir / f"gp_rules_{symbol}_{month}.json").write_text(json.dumps(rule_records, indent=2))

    (output_dir / "_SUCCESS").write_text("\n")

    return BatchResult(
        symbol=symbol,
        month=month,
        output_dir=output_dir,
        motifs_count=len(motifs_frame),
        gp_rules=len(gp_matrix),
    )


def process_batch(symbols: Sequence[str], months: Sequence[str], output_root: Path, force: bool = False) -> list[BatchResult]:
    results: list[BatchResult] = []
    for symbol in symbols:
        for month in months:
            output_dir = output_root / symbol / month
            marker = output_dir / "_SUCCESS"
            if marker.exists() and not force:
                logger.info("Skipping %s %s (already completed)", symbol, month)
                continue
            logger.info("Processing %s %s", symbol, month)
            try:
                frame = load_monthly_data(symbol, month)
            except FileNotFoundError:
                logger.warning("Missing data for %s %s; skipping", symbol, month)
                continue
            result = run_stage_pipeline(symbol, month, frame, output_root)
            logger.info("Completed %s %s motifs=%d rules=%d", symbol, month, result.motifs_count, result.gp_rules)
            results.append(result)
    return results


def chunked(iterable: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full run batch pipeline")
    parser.add_argument("--symbols", nargs="*", help="Symbols to process (default autodetect)")
    parser.add_argument("--months", nargs="*", required=True, help="Months YYYY-MM")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--force", action="store_true", help="Reprocess even if _SUCCESS exists")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbols = args.symbols if args.symbols else list_symbols_with_month(args.months[0])
    months = args.months
    for batch in chunked(symbols, args.batch_size):
        logger.info("Starting batch %s", ",".join(batch))
        process_batch(batch, months, args.output_root, force=args.force)
        logger.info("Batch complete")


if __name__ == "__main__":
    main()
