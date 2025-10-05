"""Triple-barrier labeling for decision bars."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Literal

import pandas as pd

BarrierHit = Literal["up", "down", "time"]


@dataclass
class TripleBarrierConfig:
    """Configuration for triple-barrier labeling."""

    take_profit_sigma: float = 2.0
    stop_loss_sigma: float = 2.0
    volatility_lookback: int = 48
    min_horizon: pd.Timedelta = pd.Timedelta(hours=2)
    max_horizon: pd.Timedelta = pd.Timedelta(hours=12)


@dataclass
class LabelDiagnostics:
    class_balance: Dict[int, float]
    avg_holding_hours: float
    trigger_rates: Dict[BarrierHit, float]


def _realized_volatility(closes: pd.Series, lookback: int) -> pd.Series:
    returns = closes.pct_change().fillna(0.0)
    vol = returns.ewm(span=lookback, adjust=False).std(bias=False)
    fallback = returns.abs().rolling(window=lookback, min_periods=1).mean()
    vol = vol.fillna(fallback)
    return vol.mask(vol <= 0, fallback)


def apply_triple_barrier(decision_bars: pd.DataFrame, config: TripleBarrierConfig) -> pd.DataFrame:
    """Label each decision bar with triple-barrier outcomes."""
    closes = decision_bars["close"].astype(float)
    timestamps = pd.to_datetime(decision_bars["ts"], utc=True)
    vol = _realized_volatility(closes, config.volatility_lookback)

    records = []
    ts_by_idx = timestamps.reset_index(drop=True)
    bar_ids = decision_bars["bar_id"].astype(int).reset_index(drop=True)
    for idx, entry_ts in enumerate(timestamps):
        entry_price = closes.iloc[idx]
        sigma = max(float(vol.iloc[idx]), 1e-6)
        upper = entry_price * (1 + config.take_profit_sigma * sigma)
        lower = entry_price * (1 - config.stop_loss_sigma * sigma)
        expiry = min(entry_ts + config.max_horizon, timestamps.iloc[-1])
        label = 0
        hit: BarrierHit = "time"
        holding = config.max_horizon
        exit_price = entry_price
        exit_idx = idx

        for j in range(idx + 1, len(decision_bars)):
            ts_j = timestamps.iloc[j]
            if ts_j - entry_ts < config.min_horizon and j < len(decision_bars) - 1:
                continue
            price_j = closes.iloc[j]
            if price_j >= upper:
                label = 1
                hit = "up"
                expiry = ts_j
                holding = ts_j - entry_ts
                exit_price = price_j
                exit_idx = j
                break
            if price_j <= lower:
                label = -1
                hit = "down"
                expiry = ts_j
                holding = ts_j - entry_ts
                exit_price = price_j
                exit_idx = j
                break
            if ts_j >= expiry:
                holding = ts_j - entry_ts
                exit_price = price_j
                exit_idx = j
                break
        else:
            # Fallback if loop completes without breaking (e.g., insufficient horizon)
            exit_idx = len(decision_bars) - 1
            exit_price = closes.iloc[exit_idx]
            expiry = timestamps.iloc[exit_idx]
            holding = expiry - entry_ts

        realized_return = float(exit_price / entry_price - 1.0)

        records.append(
            {
                "bar_id": int(decision_bars.iloc[idx]["bar_id"]),
                "ts": ts_by_idx.iloc[idx],
                "t1": expiry,
                "y": label,
                "up_bar": upper,
                "dn_bar": lower,
                "hit": hit,
                "holding_hours": float(holding.total_seconds() / 3600),
                "exit_price": float(exit_price),
                "exit_bar_id": int(bar_ids.iloc[exit_idx]),
                "realized_return": realized_return,
            }
        )
    return pd.DataFrame(records)


def summarize_label_distribution(labels: pd.DataFrame) -> LabelDiagnostics:
    """Return diagnostics for label balance and triggers."""
    class_counts = labels["y"].value_counts(normalize=True)
    class_balance = {int(cls): float(freq) for cls, freq in class_counts.items()}
    trigger_counts = labels["hit"].value_counts(normalize=True)
    trigger_rates = {hit: float(freq) for hit, freq in trigger_counts.items()}  # type: ignore[arg-type]
    avg_holding = labels["holding_hours"].mean() if not labels.empty else 0.0
    return LabelDiagnostics(class_balance=class_balance, avg_holding_hours=float(avg_holding), trigger_rates=trigger_rates)


def enforce_label_sanity(labels: pd.DataFrame, min_class_weight: float = 0.05) -> None:
    """Raise if label distribution becomes degenerate."""
    diagnostics = summarize_label_distribution(labels)
    for cls, weight in diagnostics.class_balance.items():
        if weight < min_class_weight:
            raise ValueError(f"Class {cls} weight {weight:.3f} below minimum {min_class_weight}")
