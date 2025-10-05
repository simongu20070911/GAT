"""Candlestick morphology feature computations."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

WINDOW_OPTIONS = (60, 120, 240, 360)
SECONDS_PER_DAY = 24 * 3600


def compute_morphology_features(ohlcv: pd.DataFrame, windows: Iterable[int] = WINDOW_OPTIONS) -> pd.DataFrame:
    """Compute normalized candlestick geometry features across lookbacks."""
    if "timestamp" not in ohlcv.columns:
        raise KeyError("Expected column 'timestamp' in OHLCV frame")

    ohlcv = ohlcv.copy()
    ohlcv["timestamp"] = pd.to_datetime(ohlcv["timestamp"], utc=True)
    ohlcv = ohlcv.set_index("timestamp").sort_index()

    features = {}
    body = (ohlcv["close"] - ohlcv["open"]).abs()
    direction = np.sign(ohlcv["close"] - ohlcv["open"]).fillna(0)
    upper = ohlcv["high"] - ohlcv[["open", "close"]].max(axis=1)
    lower = ohlcv[["open", "close"]].min(axis=1) - ohlcv["low"]
    true_range = (ohlcv["high"] - ohlcv["low"]).replace(0, np.nan)
    ret = ohlcv["close"].pct_change().fillna(0.0)
    cum_ret = ret.cumsum()

    seconds = ohlcv.index.map(lambda ts: ts.hour * 3600 + ts.minute * 60 + ts.second)
    features["tod_sin"] = np.sin(2 * np.pi * seconds / SECONDS_PER_DAY)
    features["tod_cos"] = np.cos(2 * np.pi * seconds / SECONDS_PER_DAY)
    features["direction"] = direction
    features["range"] = true_range
    features["body"] = body
    features["wick_upper"] = upper
    features["wick_lower"] = lower
    features["cum_return"] = cum_ret

    for window in windows:
        roll_tr = true_range.rolling(window=window, min_periods=1)
        roll_vol = ohlcv["volume"].rolling(window=window, min_periods=1)
        prefix = f"w{window}"
        atr = roll_tr.mean().replace(0, np.nan)
        features[f"{prefix}_body_atr"] = body / atr
        features[f"{prefix}_upper_body"] = upper / body.replace(0, np.nan)
        features[f"{prefix}_lower_body"] = lower / body.replace(0, np.nan)
        features[f"{prefix}_range_var"] = roll_tr.var()
        features[f"{prefix}_ret_sum"] = ret.rolling(window=window, min_periods=1).sum()
        features[f"{prefix}_ret_z"] = (ret - ret.rolling(window=window, min_periods=1).mean()) / (
            ret.rolling(window=window, min_periods=1).std(ddof=0)
        )
        features[f"{prefix}_vol_z"] = (ohlcv["volume"] - roll_vol.mean()) / roll_vol.std(ddof=0)
        roll_close = ohlcv["close"].rolling(window=window, min_periods=1)
        rolling_max = roll_close.max()
        rolling_min = roll_close.min()
        features[f"{prefix}_drawup"] = (ohlcv["close"] - rolling_min) / rolling_min.replace(0, np.nan)
        features[f"{prefix}_drawdown"] = (rolling_max - ohlcv["close"]) / rolling_max.replace(0, np.nan)
        narrow_range = true_range <= roll_tr.quantile(0.2)
        features[f"{prefix}_nr_flag"] = narrow_range.astype(int)

    feature_df = pd.DataFrame(features, index=ohlcv.index).reset_index().rename(columns={"index": "timestamp"})
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return feature_df


BAR_WINDOW_OPTIONS = (3, 6, 12)


def compute_morphology_features_on_bars(
    decision_bars: pd.DataFrame, windows: Iterable[int] = BAR_WINDOW_OPTIONS
) -> pd.DataFrame:
    """Compute morphology features directly on decision bars."""

    required = {"ts", "open", "high", "low", "close", "volume"}
    missing = required.difference(decision_bars.columns)
    if missing:
        raise KeyError(f"Decision bars missing columns: {sorted(missing)}")

    frame = decision_bars[list(required)].rename(columns={"ts": "timestamp"}).copy()
    features = compute_morphology_features(frame, windows=windows)
    features = features.rename(columns={"timestamp": "ts"})
    return features
