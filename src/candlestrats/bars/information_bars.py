"""Information-driven bar construction utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence, Tuple

import numpy as np
import pandas as pd

BarMethod = Literal["time", "dollar", "imbalance"]


@dataclass
class DecisionBarConfig:
    """Configuration for targeting an adaptive decision cadence."""

    target_median_seconds: int = 4 * 3600
    rebalance_minutes: int = 60
    event_preference_tolerance: float = 0.25
    keep_partial_last: bool = False
    min_partial_fraction: float = 0.5


@dataclass
class DecisionBarResult:
    bars: pd.DataFrame
    mapping: pd.DataFrame
    method: BarMethod


def _rolling_threshold(series: pd.Series, target_seconds: int, window_minutes: int) -> pd.Series:
    if series.empty:
        raise ValueError("Cannot derive thresholds from empty series")
    window = max(int(window_minutes), 1)
    multiplier = max(target_seconds // 60, 1)
    rolling_median = series.rolling(window=window, min_periods=1).median()
    threshold = rolling_median * multiplier
    fallback = float(series.median() * multiplier)
    return threshold.fillna(fallback).clip(lower=1e-6)


def _construct_time_bars(ohlcv: pd.DataFrame, target_seconds: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    freq = f"{max(target_seconds // 60, 1)}min"
    resampled = ohlcv.resample(freq, on="timestamp", label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    resampled = (
        resampled.dropna(subset=["open", "high", "low", "close"])
        .reset_index()
        .rename(columns={"timestamp": "ts"})
    )
    resampled["bar_id"] = np.arange(len(resampled), dtype=np.int64)
    bars = resampled.copy()

    mapping = pd.merge_asof(
        ohlcv[["timestamp"]].sort_values("timestamp"),
        bars[["ts", "bar_id"]].rename(columns={"ts": "timestamp"}).sort_values("timestamp"),
        on="timestamp",
        direction="forward",
        allow_exact_matches=True,
    ).dropna()
    return bars, mapping


def _construct_event_bars(
    rows: Sequence[object],
    thresholds: np.ndarray,
    method: BarMethod,
    *,
    keep_partial_last: bool,
    min_partial_fraction: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    mapping_rows = []
    buffer: list[object] = []
    cumulative = 0.0
    last_sign = 1.0
    bar_id = 0
    prev_price = float(rows[0].close) if rows else 0.0

    for idx, row in enumerate(rows):
        price = float(row.close)
        volume = float(row.volume)
        dollar = price * volume
        price_delta = price - prev_price
        if price_delta > 0:
            sign = 1.0
        elif price_delta < 0:
            sign = -1.0
        else:
            sign = last_sign
        if sign == 0:
            sign = last_sign
        last_sign = sign if sign != 0 else last_sign
        prev_price = price

        if method == "dollar":
            increment = abs(dollar)
        elif method == "imbalance":
            increment = sign * volume
        else:
            raise ValueError(f"Unsupported method {method}")  # pragma: no cover - defensive

        buffer.append(row)
        mapping_rows.append({"timestamp": row.timestamp, "bar_id": bar_id})
        cumulative += increment

        threshold = float(thresholds[idx]) if len(thresholds) > idx else 0.0
        trigger_value = abs(cumulative) if method == "imbalance" else cumulative
        if trigger_value >= threshold and threshold > 0:
            records.append(_finalize_bar(buffer, bar_id))
            bar_id += 1
            cumulative = 0.0
            buffer = []

    if buffer:
        keep = keep_partial_last
        if not keep:
            threshold_idx = min(len(thresholds) - 1, len(rows) - 1) if len(thresholds) else -1
            raw_threshold = float(thresholds[threshold_idx]) if threshold_idx >= 0 else 0.0
            progress = abs(cumulative) if method == "imbalance" else cumulative
            if raw_threshold > 0:
                fraction = progress / raw_threshold
                minimum_fraction = min(max(min_partial_fraction, 0.0), 1.0)
                keep = fraction >= minimum_fraction
            else:
                keep = False
        if keep:
            records.append(_finalize_bar(buffer, bar_id))
        else:
            mapping_rows = [row for row in mapping_rows if row["bar_id"] != bar_id]

    bars = pd.DataFrame(records)
    mapping = pd.DataFrame(mapping_rows)
    return bars, mapping


def _finalize_bar(rows: Iterable[object], bar_id: int) -> dict:
    cached = list(rows)
    first = cached[0]
    last = cached[-1]
    highs = [float(row.high) for row in cached]
    lows = [float(row.low) for row in cached]
    volumes = [float(row.volume) for row in cached]
    return {
        "bar_id": bar_id,
        "ts": last.timestamp,
        "open": float(first.open),
        "high": max(highs),
        "low": min(lows),
        "close": float(last.close),
        "volume": float(sum(volumes)),
    }


def build_decision_bars(ohlcv: pd.DataFrame, config: DecisionBarConfig) -> DecisionBarResult:
    """Construct decision bars adapting between time, dollar, and imbalance bars."""
    if "timestamp" not in ohlcv.columns:
        raise KeyError("OHLCV frame must include a 'timestamp' column")
    ohlcv = ohlcv.copy()
    ohlcv["timestamp"] = pd.to_datetime(ohlcv["timestamp"], utc=True)
    ohlcv = ohlcv.sort_values("timestamp").reset_index(drop=True)

    time_bars, time_map = _construct_time_bars(ohlcv, config.target_median_seconds)
    ohlcv = ohlcv.copy()
    ohlcv["dollar_value"] = ohlcv["close"] * ohlcv["volume"]
    index = ohlcv["timestamp"].copy()
    dollar_thresholds = _rolling_threshold(
        ohlcv.set_index("timestamp")["dollar_value"],
        config.target_median_seconds,
        config.rebalance_minutes,
    ).reindex(index).ffill().bfill()
    imbalance_thresholds = _rolling_threshold(
        ohlcv.set_index("timestamp")["volume"],
        config.target_median_seconds,
        config.rebalance_minutes,
    ).reindex(index).ffill().bfill()
    rows = list(ohlcv.itertuples(index=False, name="Bar"))
    partial_kwargs = {
        "keep_partial_last": config.keep_partial_last,
        "min_partial_fraction": config.min_partial_fraction,
    }
    dollar_bars, dollar_map = _construct_event_bars(
        rows,
        dollar_thresholds.to_numpy(),
        "dollar",
        **partial_kwargs,
    )
    imbalance_bars, imbalance_map = _construct_event_bars(
        rows,
        imbalance_thresholds.to_numpy(),
        "imbalance",
        **partial_kwargs,
    )

    candidates = {
        "time": (time_bars, time_map),
        "dollar": (dollar_bars, dollar_map),
        "imbalance": (imbalance_bars, imbalance_map),
    }

    def median_interval(frame: pd.DataFrame) -> float:
        if len(frame) < 2:
            return float("inf")
        deltas = pd.Series(frame["ts"]).sort_values().diff().dropna().dt.total_seconds()
        return float(deltas.median()) if not deltas.empty else float("inf")

    target = config.target_median_seconds
    tolerance = max(config.event_preference_tolerance * target, 0.0)

    def candidate_score(method: BarMethod) -> tuple[float, bool]:
        bars_candidate, _ = candidates[method]
        median = median_interval(bars_candidate)
        diff = abs(median - target)
        valid = not bars_candidate.empty and math.isfinite(diff)
        return diff, valid

    import math

    event_methods: list[tuple[float, str]] = []
    for method in ("dollar", "imbalance"):
        diff, valid = candidate_score(method)
        if valid and diff <= tolerance:
            event_methods.append((diff, method))

    if event_methods:
        event_methods.sort(key=lambda item: item[0])
        best_method = event_methods[0][1]
    else:
        best_method = min(candidates.keys(), key=lambda m: candidate_score(m)[0])

    bars, mapping = candidates[best_method]
    bars = bars.reset_index(drop=True)
    mapping = mapping.reset_index(drop=True)
    return DecisionBarResult(bars=bars, mapping=mapping, method=best_method)


__all__ = [
    "DecisionBarConfig",
    "DecisionBarResult",
    "build_decision_bars",
]
