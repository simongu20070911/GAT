"""Multiple testing corrections and diagnostics."""
from __future__ import annotations

import math
from statistics import NormalDist
from typing import Iterable

import numpy as np


_NORMAL = NormalDist()


def _clean_array(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    return array[~np.isnan(array)]


def probabilistic_sharpe_ratio(
    sharpe: float,
    sharpe_ref: float,
    n_obs: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Compute the Probabilistic Sharpe Ratio (López de Prado, 2018)."""

    if n_obs <= 1 or not math.isfinite(sharpe):
        return float("nan")

    variability = 1.0 - skewness * sharpe + ((kurtosis - 3.0) / 4.0) * (sharpe ** 2)
    if variability <= 0:
        return float("nan")

    numerator = (sharpe - sharpe_ref) * math.sqrt(n_obs - 1)
    denominator = math.sqrt(variability)
    if denominator == 0:
        return float("nan")
    z_score = numerator / denominator
    return _NORMAL.cdf(z_score)


def deflated_sharpe_ratio(
    sharpe: float,
    n_obs: int,
    trials: int,
    sharpe_ref: float = 0.0,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Compute the Deflated Sharpe Ratio (López de Prado, 2018)."""

    if n_obs <= 1 or not math.isfinite(sharpe):
        return float("nan")

    variability = 1.0 - skewness * sharpe + ((kurtosis - 3.0) / 4.0) * (sharpe ** 2)
    if variability <= 0:
        return float("nan")

    sigma_sr = math.sqrt(variability / (n_obs - 1))
    if sigma_sr == 0:
        return float("nan")

    if trials and trials > 1:
        exceedance_prob = 1.0 - 1.0 / trials
        exceedance_prob = min(max(exceedance_prob, 1e-12), 1.0 - 1e-12)
        z_max = _NORMAL.inv_cdf(exceedance_prob)
    else:
        z_max = 0.0

    sharpe_star = sharpe_ref + sigma_sr * z_max
    z_score = (sharpe - sharpe_star) / sigma_sr
    return _NORMAL.cdf(z_score)


def probability_of_backtest_overfitting(
    out_of_sample_sharpes: Iterable[float],
    threshold: float = 0.0,
) -> float:
    """Estimate the Probability of Backtest Overfitting as the share of folds below the threshold."""

    sharpes = _clean_array(out_of_sample_sharpes)
    if sharpes.size == 0:
        return float("nan")
    return float((sharpes <= threshold).mean())


# Backwards-compatible aliases for legacy imports.
psr_proxy = probabilistic_sharpe_ratio
dsr_proxy = deflated_sharpe_ratio
pbo_proxy = probability_of_backtest_overfitting
compute_dsr = deflated_sharpe_ratio
compute_pbo = probability_of_backtest_overfitting
