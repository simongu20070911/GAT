"""Monte Carlo validation harness."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

import numpy as np
import pandas as pd

from candlestrats.evaluation.cpcv import CombinatorialPurgedCV, evaluate_strategy
from candlestrats.utils import generate_random_walk_ohlcv


@dataclass
class MonteCarloResult:
    real_metric: float
    synthetic_metrics: List[float]

    @property
    def p_value(self) -> float:
        if not self.synthetic_metrics:
            return float("nan")
        exceed = sum(metric >= self.real_metric for metric in self.synthetic_metrics)
        return (exceed + 1) / (len(self.synthetic_metrics) + 1)


def monte_carlo_null_distribution(
    returns: pd.Series,
    cv: CombinatorialPurgedCV,
    generator: Callable[[int], pd.Series] | None = None,
    runs: int = 100,
    random_state: int | np.random.Generator | None = None,
) -> MonteCarloResult:
    """Compare real strategy metrics against synthetic random-walk baselines."""
    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)

    def default_generator(length: int) -> pd.Series:
        return _random_walk_returns(length, rng)

    use_generator = generator or default_generator
    real_folds = evaluate_strategy(returns, cv)
    real_metric = float(np.mean([fold.sharpe for fold in real_folds])) if real_folds else float("nan")

    synthetic_metrics: List[float] = []
    for _ in range(runs):
        synthetic = use_generator(len(returns))
        synthetic_folds = evaluate_strategy(synthetic, cv)
        synthetic_metrics.append(float(np.mean([fold.sharpe for fold in synthetic_folds])) if synthetic_folds else 0.0)
    return MonteCarloResult(real_metric=real_metric, synthetic_metrics=synthetic_metrics)


def _random_walk_returns(length: int, rng: np.random.Generator | None = None) -> pd.Series:
    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng()
    ohlcv = generate_random_walk_ohlcv(length + 1, random_state=generator)
    series = pd.Series(ohlcv["close"]).pct_change().dropna()
    series.index = pd.date_range(start="2000-01-01", periods=len(series), freq="min")
    return series


def permutation_mcpt(
    returns: pd.Series,
    cv: CombinatorialPurgedCV,
    block_size: int = 24,
    runs: int = 100,
    random_state: int | np.random.Generator | None = None,
) -> MonteCarloResult:
    """Permutation-based MCPT preserving intraday structure via block shuffling."""
    real_folds = evaluate_strategy(returns, cv)
    real_metric = float(np.mean([fold.sharpe for fold in real_folds])) if real_folds else float("nan")
    synthetic_metrics: List[float] = []
    values = returns.to_numpy()
    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
    for _ in range(runs):
        permuted = _block_shuffle(values, block_size, rng)
        series = pd.Series(permuted, index=returns.index)
        folds = evaluate_strategy(series, cv)
        synthetic_metrics.append(float(np.mean([fold.sharpe for fold in folds])) if folds else 0.0)
    return MonteCarloResult(real_metric=real_metric, synthetic_metrics=synthetic_metrics)


def _block_shuffle(values: np.ndarray, block_size: int, rng: np.random.Generator) -> np.ndarray:
    if block_size <= 1:
        return rng.permutation(values)
    n_blocks = max(len(values) // block_size, 1)
    blocks = [values[i * block_size : (i + 1) * block_size] for i in range(n_blocks)]
    if len(values) % block_size:
        blocks.append(values[n_blocks * block_size :])
    rng.shuffle(blocks)
    return np.concatenate(blocks)
