import pandas as pd

from candlestrats.evaluation import (
    CombinatorialPurgedCV,
    MonteCarloResult,
    dsr_proxy,
    pbo_proxy,
    psr_proxy,
    monte_carlo_null_distribution,
    permutation_mcpt,
)


def test_probabilistic_sharpe_ratio_bounds():
    psr = psr_proxy(0.5, 0.0, 50)
    assert 0.0 <= psr <= 1.0


def test_compute_dsr_produces_value():
    dsr = dsr_proxy(0.5, n_obs=50, trials=10)
    assert dsr == dsr  # not NaN


def test_compute_pbo_fraction():
    pbo = pbo_proxy([0.1, -0.2, 0.3])
    assert 0.0 <= pbo <= 1.0


def test_monte_carlo_null_distribution():
    index = pd.date_range(start="2024-01-01", periods=120, freq="min")
    returns = pd.Series(0.001, index=index)
    cv = CombinatorialPurgedCV(n_splits=4, embargo_minutes=5)
    result = monte_carlo_null_distribution(returns, cv, runs=5)
    assert isinstance(result, MonteCarloResult)
    assert len(result.synthetic_metrics) == 5
    assert 0.0 <= result.p_value <= 1.0



def test_permutation_mcpt():
    index = pd.date_range(start="2024-01-01", periods=120, freq="min")
    returns = pd.Series(0.001, index=index)
    cv = CombinatorialPurgedCV(n_splits=4, embargo_minutes=5)
    result = permutation_mcpt(returns, cv, block_size=12, runs=5)
    assert len(result.synthetic_metrics) == 5
