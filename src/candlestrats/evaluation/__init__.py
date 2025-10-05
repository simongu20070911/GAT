"""Evaluation and validation utilities."""

from .cpcv import CombinatorialPurgedCV, evaluate_strategy  # noqa: F401
from .multiple_testing import (  # noqa: F401
    probabilistic_sharpe_ratio,
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
    psr_proxy,
    dsr_proxy,
    pbo_proxy,
    compute_dsr,
    compute_pbo,
)
from .costs import FeeModel  # noqa: F401
from .monte_carlo import MonteCarloResult, monte_carlo_null_distribution, permutation_mcpt  # noqa: F401

from .fees import FeeConfig, load_fee_model_from_config, resolve_fee_model, resolve_fee_tier  # noqa: F401
from .reporting import mcpt_summary, MCPTSummaryInput, write_mcpt_summary  # noqa: F401
