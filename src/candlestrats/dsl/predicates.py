"""Vectorized predicate evaluators for candlestick morphology."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Protocol

import numpy as np
import pandas as pd


class PredicateFunction(Protocol):
    def __call__(self, data: pd.DataFrame, params: dict[str, float]) -> pd.Series:
        ...


@dataclass
class Predicate:
    name: str
    params: dict[str, float]
    evaluator: PredicateFunction

    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        mask = self.evaluator(data, self.params)
        series = pd.Series(mask, index=data.index, name=self.name)
        series = series.replace([np.inf, -np.inf], np.nan).fillna(False)
        return series.astype(bool)


class PredicateRegistry:
    """Registry for base predicates used by the DSL."""

    def __init__(self) -> None:
        self._predicates: Dict[str, PredicateFunction] = {}

    def register(self, name: str, fn: PredicateFunction) -> None:
        if name in self._predicates:
            raise KeyError(f"Predicate {name} already registered")
        self._predicates[name] = fn

    def list(self) -> Iterable[str]:
        return self._predicates.keys()

    def get(self, name: str) -> PredicateFunction:
        if name not in self._predicates:
            raise KeyError(f"Predicate {name} not found")
        return self._predicates[name]


registry = PredicateRegistry()


def _normalize(df: pd.DataFrame, column: str, window: int) -> pd.Series:
    atr = df["true_range"].rolling(window=window, min_periods=1).mean()
    return (df[column] / atr.replace(0, np.nan)).fillna(0.0)


def _ensure_fields(df: pd.DataFrame) -> pd.DataFrame:
    if "body" not in df:
        df = df.copy()
        df["body"] = (df["close"] - df["open"]).abs()
        df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
        df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
        df["true_range"] = df["high"] - df["low"]
        df["clv"] = (df["close"] - df["low"]) / df["true_range"].replace(0, np.nan)
    return df


def wick_ratio_predicate(df: pd.DataFrame, params: dict[str, float]) -> pd.Series:
    df = _ensure_fields(df)
    kind = params.get("kind", "lower")
    threshold = params.get("threshold", 1.5)
    if kind == "lower":
        ratio = df["lower_wick"] / df["true_range"].replace(0, np.nan)
    else:
        ratio = df["upper_wick"] / df["true_range"].replace(0, np.nan)
    return ratio.fillna(0.0) >= threshold


def range_compression_predicate(df: pd.DataFrame, params: dict[str, float]) -> pd.Series:
    df = _ensure_fields(df)
    window = int(params.get("window", 5))
    quantile = float(params.get("quantile", 0.2))
    rolling = df["true_range"].rolling(window=window, min_periods=window)
    threshold = rolling.quantile(quantile)
    mask = df["true_range"] <= threshold
    return mask.fillna(False)


def volume_spike_predicate(df: pd.DataFrame, params: dict[str, float]) -> pd.Series:
    window = int(params.get("window", 20))
    quantile = float(params.get("quantile", 0.8))
    rolling = df["volume"].rolling(window=window, min_periods=window)
    threshold = rolling.quantile(quantile)
    return (df["volume"] >= threshold).fillna(False)


def close_above_prior_low(df: pd.DataFrame, params: dict[str, float]) -> pd.Series:
    lag = int(params.get("lag", 1))
    prior_low = df["low"].shift(lag)
    return df["close"] > prior_low


def break_above_range(df: pd.DataFrame, params: dict[str, float]) -> pd.Series:
    lookback = int(params.get("lookback", 10))
    eps = float(params.get("eps", 0.0))
    rolling_high = df["high"].rolling(window=lookback, min_periods=lookback).max().shift(1)
    mask = df["close"] >= (rolling_high + eps)
    return mask.fillna(False)




def inside_bar_predicate(df: pd.DataFrame, params: dict[str, float]) -> pd.Series:
    high_prev = df["high"].shift(1)
    low_prev = df["low"].shift(1)
    return (df["high"] < high_prev) & (df["low"] > low_prev)


def outside_bar_predicate(df: pd.DataFrame, params: dict[str, float]) -> pd.Series:
    high_prev = df["high"].shift(1)
    low_prev = df["low"].shift(1)
    return (df["high"] > high_prev) & (df["low"] < low_prev)


def bullish_engulf_predicate(df: pd.DataFrame, params: dict[str, float]) -> pd.Series:
    open_prev = df["open"].shift(1)
    close_prev = df["close"].shift(1)
    min_prev = pd.concat([open_prev, close_prev], axis=1).min(axis=1)
    max_prev = pd.concat([open_prev, close_prev], axis=1).max(axis=1)
    min_curr = df[["open", "close"]].min(axis=1)
    max_curr = df[["open", "close"]].max(axis=1)
    return (close_prev < open_prev) & (df["close"] > df["open"]) & (min_curr <= min_prev) & (max_curr >= max_prev)


def bearish_engulf_predicate(df: pd.DataFrame, params: dict[str, float]) -> pd.Series:
    open_prev = df["open"].shift(1)
    close_prev = df["close"].shift(1)
    min_prev = pd.concat([open_prev, close_prev], axis=1).min(axis=1)
    max_prev = pd.concat([open_prev, close_prev], axis=1).max(axis=1)
    min_curr = df[["open", "close"]].min(axis=1)
    max_curr = df[["open", "close"]].max(axis=1)
    return (close_prev > open_prev) & (df["close"] < df["open"]) & (min_curr <= min_prev) & (max_curr >= max_prev)


def break_below_range(df: pd.DataFrame, params: dict[str, float]) -> pd.Series:
    lookback = int(params.get("lookback", 10))
    eps = float(params.get("eps", 0.0))
    rolling_low = df["low"].rolling(window=lookback, min_periods=lookback).min().shift(1)
    mask = df["close"] <= (rolling_low - eps)
    return mask.fillna(False)


def wick_dominance_predicate(df: pd.DataFrame, params: dict[str, float]) -> pd.Series:
    df = _ensure_fields(df)
    kind = params.get("kind", "upper")
    min_ratio = params.get("ratio", 1.5)
    if kind == "upper":
        ratio = df["upper_wick"] / df["body"].replace(0, np.nan)
    else:
        ratio = df["lower_wick"] / df["body"].replace(0, np.nan)
    return ratio.fillna(0.0) >= min_ratio


def evr_anomaly_predicate(df: pd.DataFrame, params: dict[str, float]) -> pd.Series:
    df = _ensure_fields(df)
    window = int(params.get("window", 20))
    volume_roll = df["volume"].rolling(window=window, min_periods=window)
    vol_mean = volume_roll.mean()
    vol_std = volume_roll.std(ddof=0).replace(0, np.nan)
    zscore = (df["volume"] - vol_mean) / vol_std
    atr_norm = df["true_range"].rolling(window=window, min_periods=window).mean().replace(0, np.nan)
    effort = df["true_range"] / atr_norm
    evr = zscore / effort.replace(0, np.nan)
    evr = evr.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    threshold = params.get("threshold", 1.0)
    return evr >= threshold




def pivot_high_predicate(df: pd.DataFrame, params: dict[str, float]) -> pd.Series:
    left = int(params.get("left", 2))
    right = int(params.get("right", 2))
    highs = df["high"]
    cond = pd.Series(True, index=df.index)
    for i in range(1, left + 1):
        cond &= highs > highs.shift(i)
    for i in range(1, right + 1):
        cond &= highs >= highs.shift(-i)
    return cond.fillna(False)


def pivot_low_predicate(df: pd.DataFrame, params: dict[str, float]) -> pd.Series:
    left = int(params.get("left", 2))
    right = int(params.get("right", 2))
    lows = df["low"]
    cond = pd.Series(True, index=df.index)
    for i in range(1, left + 1):
        cond &= lows < lows.shift(i)
    for i in range(1, right + 1):
        cond &= lows <= lows.shift(-i)
    return cond.fillna(False)

def round_level_proximity_predicate(df: pd.DataFrame, params: dict[str, float]) -> pd.Series:
    df = _ensure_fields(df)
    tick = params.get("tick")
    percent = params.get("percent")
    step_param = params.get("step")
    fallback_step = float(step_param) if step_param is not None else None
    tolerance = float(params.get("tolerance", 0.2))
    atr_window = int(params.get("atr_window", 20))
    atr = df["true_range"].rolling(window=atr_window, min_periods=atr_window).mean().replace(0, np.nan)
    if tick is not None:
        tick_size = float(tick)
        rounded = (df["close"] / tick_size).round() * tick_size
    else:
        closes = df["close"].dropna().astype(float)
        if percent is None and fallback_step is None:
            median_price = float(closes.median()) if not closes.empty else 0.0
            if median_price > 1.0:
                percent = 0.01
            else:
                sorted_prices = closes.sort_values().drop_duplicates()
                diffs = sorted_prices.diff().abs()
                min_step = diffs[diffs > 0].min()
                fallback_step = float(min_step) if min_step and not np.isnan(min_step) else 0.0001
        if percent is not None:
            pct = float(percent)
        elif fallback_step is not None:
            tick_size = float(fallback_step)
            if tick_size <= 0:
                raise ValueError("step must be positive when percent/tick are absent")
            rounded = (df["close"] / tick_size).round() * tick_size
            distance = (df["close"] - rounded).abs()
            scaled = distance / atr
            return scaled.fillna(np.inf) <= tolerance
        else:
            pct = 0.01
        grid = 1.0 + pct
        if grid <= 0:
            raise ValueError("percent grid must be greater than -1")
        close = df["close"].clip(lower=1e-12)
        reference = float(params.get("base", close.iloc[0]))
        if reference <= 0:
            positive = close[close > 0]
            reference = float(positive.iloc[0]) if not positive.empty else 1.0
        ratio = (close / reference).clip(lower=1e-12)
        log_step = np.log(grid)
        exponents = np.rint(np.log(ratio) / log_step)
        rounded = reference * np.exp(exponents * log_step)
    distance = (df["close"] - rounded).abs()
    scaled = distance / atr
    return scaled.fillna(np.inf) <= tolerance


def higher_low(df: pd.DataFrame, params: dict[str, float]) -> pd.Series:
    depth = int(params.get("depth", 2))
    lows = [df["low"].shift(i) for i in range(depth + 1)]
    mask = pd.Series(True, index=df.index)
    for i in range(1, len(lows)):
        mask &= lows[i - 1] > lows[i]
    return mask


def register_default_predicates() -> None:
    registry.register("LONG_LOWER_WICK", wick_ratio_predicate)
    registry.register("RANGE_COMPRESSION", range_compression_predicate)
    registry.register("VOLUME_SPIKE", volume_spike_predicate)
    registry.register("CLOSE_ABOVE_PRIOR_LOW", close_above_prior_low)
    registry.register("BREAK_ABOVE_RANGE", break_above_range)
    registry.register("BREAK_BELOW_RANGE", break_below_range)
    registry.register("HIGHER_LOW", higher_low)
    registry.register("INSIDE_BAR", inside_bar_predicate)
    registry.register("OUTSIDE_BAR", outside_bar_predicate)
    registry.register("BULLISH_ENGULF", bullish_engulf_predicate)
    registry.register("BEARISH_ENGULF", bearish_engulf_predicate)
    registry.register("WICK_DOMINANCE", wick_dominance_predicate)
    registry.register("EVR_ANOMALY", evr_anomaly_predicate)
    registry.register("ROUND_LEVEL_PROXIMITY", round_level_proximity_predicate)
    registry.register("PIVOT_HIGH", pivot_high_predicate)
    registry.register("PIVOT_LOW", pivot_low_predicate)


def evaluate_predicates(df: pd.DataFrame, predicates: Iterable[Predicate]) -> pd.DataFrame:
    df = _ensure_fields(df)
    results = {}
    for predicate in predicates:
        results[predicate.name] = predicate.evaluate(df)
    return pd.DataFrame(results, index=df.index)


register_default_predicates()
