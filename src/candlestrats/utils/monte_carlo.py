"""Synthetic data generation for leakage checks."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def generate_random_walk_ohlcv(
    length: int,
    start_price: float = 100.0,
    random_state: int | np.random.Generator | None = None,
) -> pd.DataFrame:
    """Generate a simple geometric random walk OHLCV series."""
    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
    returns = rng.normal(loc=0.0, scale=0.01, size=length)
    prices = start_price * np.exp(np.cumsum(returns))
    highs = prices * (1 + rng.uniform(0, 0.005, size=length))
    lows = prices * (1 - rng.uniform(0, 0.005, size=length))
    opens = np.concatenate((np.array([start_price]), prices[:-1]))
    volumes = rng.lognormal(mean=0, sigma=0.1, size=length)
    timestamps = pd.date_range(start="2000-01-01", periods=length, freq="min")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": prices,
            "volume": volumes,
        }
    )
