"""Post-batch Monte Carlo guardrail runner for flagged symbols."""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

from candlestrats.bars import DecisionBarConfig
from candlestrats.data import MinuteBarStore
from candlestrats.data.ingestion import BinanceIngestionSpec
from candlestrats.evaluation import CombinatorialPurgedCV, FeeModel, resolve_fee_model
from candlestrats.features.motifs import MotifSpec
from candlestrats.gp import GeneticProgramConfig, GeneticProgramMiner
from candlestrats.gp.evaluation import EvaluationResult, StrategyEvaluator
from candlestrats.gp.miner import CandidateRule
from candlestrats.labeling import TripleBarrierConfig
from candlestrats.pipeline import PipelineConfig, run_pipeline
from candlestrats.utils.monte_carlo import generate_random_walk_ohlcv

OUTPUT_ROOT = Path("/mnt/timemachine/binance/features/full_run")
MONTHLY_ROOT = Path("/mnt/timemachine/binance/futures/um/monthly/klines")
DEFAULT_THRESHOLD = 0.3
DEFAULT_RATIO = 0.75
DEFAULT_HITS = 2


def _finite_or_none(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


@dataclass
class GuardrailConfig:
    runs: int = 50
    min_trades: int = 30
    threshold: float = DEFAULT_THRESHOLD
    ratio_threshold: float = DEFAULT_RATIO
    hits_required: int = DEFAULT_HITS


MONTH_PATTERN = re.compile(r"^\d{4}-\d{2}$")
WINDOW_PATTERN = re.compile(r"(\d{4}-\d{2})_to_(\d{4}-\d{2})")


def _months_between(start: str, end: str) -> list[str]:
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    months = []
    current = start_dt
    while current <= end_dt:
        months.append(current.strftime("%Y-%m"))
        current += pd.offsets.MonthBegin(1)
    return months


def _load_single_month(symbol: str, month: str) -> pd.DataFrame:
    csv_path = MONTHLY_ROOT / symbol / "1m" / f"{symbol}-1m-{month}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing monthly CSV {csv_path}")
    frame = pd.read_csv(csv_path, header=None)
    if frame.shape[1] >= 6:
        frame.columns = [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "count",
            "taker_buy_volume",
            "taker_buy_quote_volume",
            "ignore",
        ][: frame.shape[1]]
        if isinstance(frame.loc[0, "open_time"], str) and frame.loc[0, "open_time"].lower() == "open_time":
            frame = frame.drop(index=0).reset_index(drop=True)
    if "open_time" not in frame.columns:
        frame = pd.read_csv(csv_path)
    frame["open_time"] = pd.to_numeric(frame["open_time"], errors="coerce")
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["timestamp"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp"])
    frame = frame.dropna(subset=numeric_cols)
    return frame[["timestamp", "open", "high", "low", "close", "volume"]]


def load_symbol_data(symbol: str, label: str) -> pd.DataFrame:
    label = str(label)
    if MONTH_PATTERN.match(label):
        return _load_single_month(symbol, label)
    match = WINDOW_PATTERN.search(label)
    if match:
        start, end = match.groups()
        frames = []
        for month in _months_between(start, end):
            try:
                frames.append(_load_single_month(symbol, month))
            except FileNotFoundError:
                continue
        if not frames:
            raise FileNotFoundError(f"No data found for {symbol} {label}")
        combined = pd.concat(frames, ignore_index=True).drop_duplicates(subset="timestamp")
        combined = combined.sort_values("timestamp").reset_index(drop=True)
        return combined
    raise FileNotFoundError(f"Unrecognized label {label}")





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


def evaluate_expression(expression: str, features: pd.DataFrame) -> pd.Series:
    # Reuse GP miner evaluation helpers for consistency
    config = GeneticProgramConfig(feature_columns=tuple(col for col in features.columns))
    miner = GeneticProgramMiner(config)
    return miner._evaluate_expression(expression, features)


def extract_candidate_expression(summary: dict, rules_path: Path) -> str | None:
    leak_rule = summary.get("leak_rule") or {}
    if leak_rule and leak_rule.get("expression"):
        return str(leak_rule["expression"])
    try:
        records = json.loads(rules_path.read_text())
    except FileNotFoundError:
        return None
    finite = [
        r for r in records
        if isinstance(r.get("fitness"), (int, float)) and np.isfinite(r.get("fitness"))
    ]
    if not finite:
        return None
    finite.sort(key=lambda r: r["fitness"], reverse=True)
    return str(finite[0]["expression"])


def guardrail_for_symbol(symbol: str, month: str, summary_path: Path, cfg: GuardrailConfig, force: bool = False) -> dict:
    summary = json.loads(summary_path.read_text())
    initial_flagged = bool(summary.get("flagged", False))
    if not initial_flagged and not force:
        return {"skipped": True, "reason": "not_flagged"}

    rules_path = summary_path.parent / f"gp_rules_{symbol}_{month}.json"
    expression = extract_candidate_expression(summary, rules_path)
    if not expression:
        summary["post_guard_error"] = "missing_expression"
        summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False))
        return {"skipped": True, "reason": "missing_expression"}

    pipeline_config = PipelineConfig(
        symbol=symbol,
        bar=DecisionBarConfig(),
        triple_barrier=TripleBarrierConfig(),
        motif=MotifSpec(window=24, n_clusters=8),
    )
    features_raw: pd.DataFrame | None = None
    labels_raw: pd.DataFrame | None = None

    use_artifacts = not MONTH_PATTERN.match(month)
    if use_artifacts:
        base_dir = summary_path.parent
        features_path = base_dir / f"{symbol}_features.parquet"
        labels_path = base_dir / f"{symbol}_labels.parquet"
        if features_path.exists() and labels_path.exists():
            features_raw = pd.read_parquet(features_path)
            labels_raw = pd.read_parquet(labels_path)
        else:
            use_artifacts = False

    if not use_artifacts:
        frame = load_symbol_data(symbol, month)
        store = InMemoryStore(symbol, frame)
        result = run_pipeline(store, pipeline_config)
        features_raw = result.features
        labels_raw = result.labels

    assert features_raw is not None and labels_raw is not None

    feature_ts = pd.to_datetime(features_raw.get("ts"), utc=True)
    features = features_raw.drop(columns=["ts"], errors="ignore")
    features.index = feature_ts
    features.index.name = "ts"

    label_ts_series = labels_raw.get("ts")
    if label_ts_series is None:
        label_ts_series = labels_raw.get("timestamp")
    label_ts = pd.to_datetime(label_ts_series, utc=True)
    label_frame = labels_raw.set_index(label_ts, drop=False)
    label_frame.index.name = "ts"

    features, label_frame = features.align(label_frame, join="inner", axis=0)
    feature_cols = list(features.columns)

    try:
        fee_model, _ = resolve_fee_model("binance", "futures_usdm", symbol=symbol)
    except Exception:  # pragma: no cover - defensive fallback
        fee_model = FeeModel()

    embargo_minutes = 720

    match = re.search(r"pop(\d+)_gen(\d+)", month)
    trials = None
    if match:
        try:
            trials = int(match.group(1)) * int(match.group(2))
        except ValueError:
            trials = None

    evaluator = StrategyEvaluator(
        cv=CombinatorialPurgedCV(n_splits=3, embargo_minutes=embargo_minutes),
        fee_model=fee_model,
        min_trades=cfg.min_trades,
        trials=trials,
    )

    real_signal = evaluate_expression(expression, features)
    real_eval = evaluator.evaluate_signals(real_signal, label_frame)
    if real_eval.trade_count < cfg.min_trades:
        summary["post_guard_error"] = "insufficient_real_trades"
        summary["post_guard_checked_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False))
        return {"skipped": True, "reason": "insufficient_real_trades"}
    turnover_weight = 0.5
    breadth_weight = 1.0

    def _composite(result: EvaluationResult) -> float:
        score = result.dsr
        if np.isnan(score):
            return float("-inf")
        score -= turnover_weight * float(result.turnover)
        score += breadth_weight * float(result.breadth)
        score -= float(result.pbo)
        return score

    real_fitness = _composite(real_eval)

    synthetic_metrics: List[float] = []
    positive_metrics: List[float] = []
    hits = 0

    synthetic_rng = np.random.default_rng(42)
    base_length = max(len(label_frame), 10)

    for _ in range(cfg.runs):
        synthetic_length = base_length
        synthetic_frame = generate_random_walk_ohlcv(synthetic_length, random_state=synthetic_rng)
        if "timestamp" in synthetic_frame.columns and synthetic_frame["timestamp"].dt.tz is None:
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
        synthetic_label_frame = synthetic_result.labels.set_index(synthetic_label_ts, drop=False)
        synthetic_label_frame.index.name = "ts"
        synthetic_features, synthetic_label_frame = synthetic_features.align(synthetic_label_frame, join="inner", axis=0)
        synthetic_signal = evaluate_expression(expression, synthetic_features)
        syn_eval = evaluator.evaluate_signals(synthetic_signal, synthetic_label_frame)
        metric = _composite(syn_eval)
        if np.isnan(metric):
            metric = float("-inf")
        synthetic_metrics.append(metric)
        if syn_eval.trade_count >= cfg.min_trades and np.isfinite(metric) and metric >= cfg.threshold and real_fitness > 0:
            ratio = metric / real_fitness
            trade_ratio = syn_eval.trade_count / max(real_eval.trade_count, 1)
            if ratio >= cfg.ratio_threshold and trade_ratio <= 2.0:
                hits += 1
                positive_metrics.append(metric)

    positive_metrics = [m for m in positive_metrics if np.isfinite(m)]
    pass_guard = hits < cfg.hits_required

    summary["flagged_initial"] = initial_flagged
    summary["flagged"] = not pass_guard
    summary["post_guard_runs"] = cfg.runs
    summary["post_guard_hits"] = hits
    summary["post_guard_threshold"] = cfg.threshold
    summary["post_guard_ratio_threshold"] = cfg.ratio_threshold
    summary["post_guard_hits_required"] = cfg.hits_required
    summary["post_guard_max_positive_fitness"] = _finite_or_none(max(positive_metrics) if positive_metrics else None)
    summary["post_guard_mean_positive_fitness"] = float(np.mean(positive_metrics)) if positive_metrics else float("nan")
    summary["post_guard_passed"] = pass_guard
    summary["post_guard_checked_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    summary["post_guard_real_fitness"] = float(real_fitness)
    summary["post_guard_real_dsr"] = _finite_or_none(real_eval.dsr)
    summary["post_guard_metrics"] = synthetic_metrics

    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False))

    return {
        "skipped": False,
        "passed": pass_guard,
        "hits": hits,
        "expression": expression,
        "symbol": symbol,
        "month": month,
    }


def iter_summaries(output_root: Path, symbols: set[str] | None, months: set[str] | None) -> Iterable[tuple[str, str, Path]]:
    for summary_path in output_root.glob("*/*/synthetic_check_*.json"):
        parts = summary_path.stem.split("_")
        if len(parts) < 3:
            continue
        sym = summary_path.parent.parent.name
        month = summary_path.parent.name
        if symbols and sym not in symbols:
            continue
        if months and month not in months:
            continue
        yield sym, month, summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-run Monte Carlo guardrail for flagged symbols")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--symbols", nargs="*", help="Specific symbols to process")
    parser.add_argument("--months", nargs="*", help="Specific months YYYY-MM")
    parser.add_argument("--runs", type=int, default=50, help="Monte Carlo runs per symbol")
    parser.add_argument("--min-trades", type=int, default=30, help="Minimum trades required for DSR/PBO computation")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Fitness threshold for synthetic hits")
    parser.add_argument("--ratio-threshold", type=float, default=DEFAULT_RATIO, help="Ratio vs real fitness to count a hit")
    parser.add_argument("--hits-required", type=int, default=DEFAULT_HITS, help="Number of synthetic hits needed to mark as leak")
    parser.add_argument("--force-all", action="store_true", help="Run guardrail even if summary was not flagged")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = GuardrailConfig(
        runs=args.runs,
        min_trades=args.min_trades,
        threshold=args.threshold,
        ratio_threshold=args.ratio_threshold,
        hits_required=args.hits_required,
    )

    symbols = set(args.symbols) if args.symbols else None
    months = set(args.months) if args.months else None
    output_root = args.output_root

    results: List[dict] = []
    for symbol, month, summary_path in iter_summaries(output_root, symbols, months):
        outcome = guardrail_for_symbol(symbol, month, summary_path, cfg, force=args.force_all)
        outcome.update({"summary": str(summary_path)})
        results.append(outcome)
        status = "skipped" if outcome.get("skipped") else ("PASS" if outcome.get("passed") else "FAIL")
        print(f"{symbol} {month} -> {status}")

    report_path = output_root / f"synthetic_guardrail_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    report_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote guardrail report to {report_path}")


if __name__ == "__main__":
    main()
