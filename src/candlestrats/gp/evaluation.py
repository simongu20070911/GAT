"""Strategy evaluation helpers for GP candidates."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import math
import numpy as np
import pandas as pd

from candlestrats.evaluation import (
    CombinatorialPurgedCV,
    FeeModel,
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
    evaluate_strategy,
    probabilistic_sharpe_ratio,
)


@dataclass
class EvaluationResult:
    dsr: float
    pbo: float
    turnover: float
    breadth: float
    metrics: Dict[str, float]
    trade_count: int


class StrategyEvaluator:
    """Evaluate signal series under CPCV with fee-aware adjustments."""

    def __init__(
        self,
        cv: CombinatorialPurgedCV,
        fee_model: FeeModel | None = None,
        min_trades: int = 10,
        entry_threshold: float = 0.1,
        trials: int | None = None,
    ) -> None:
        self.cv = cv
        self.fee_model = fee_model or FeeModel()
        self.min_trades = min_trades
        self.entry_threshold = entry_threshold
        self.trials = trials

    def evaluate_signals(self, signals: pd.Series, labels: pd.Series | pd.DataFrame) -> EvaluationResult:
        """Evaluate a signal series against realized returns (and optional labels)."""

        signals = signals.fillna(0.0)

        if isinstance(labels, pd.DataFrame):
            returns_series = labels.get("realized_return")
            if returns_series is None:
                returns_series = labels.get("gross_return")
            direction_series = labels.get("y")
        else:
            returns_series = labels
            direction_series = None

        if returns_series is None:
            # Fall back to directional labels if no realized returns provided.
            returns_series = direction_series if direction_series is not None else labels

        returns_series = returns_series.fillna(0.0)

        if not isinstance(signals.index, pd.DatetimeIndex):
            raise TypeError("Signals must be indexed by timestamps for CPCV evaluation")
        if not isinstance(returns_series.index, pd.DatetimeIndex):
            raise TypeError("Returns must be indexed by timestamps for CPCV evaluation")

        aligned_signals, aligned_returns = signals.align(returns_series, join="inner")
        signal_array = aligned_signals.to_numpy(dtype=float)
        aligned_returns = aligned_returns.fillna(0.0)

        entry_threshold = self.entry_threshold
        position_array = np.where(np.abs(signal_array) > entry_threshold, np.sign(signal_array), 0.0)
        position_series = pd.Series(position_array, index=aligned_signals.index, dtype=float)
        turnover_series = position_series.diff().abs().fillna(0.0)
        participation = float(np.clip(turnover_series.mean(), 0.0, 1.0))
        per_trade_cost = self.fee_model.cost_bps(maker=False, participation=participation) / 10000.0
        cost_series = turnover_series * per_trade_cost
        gross_returns = position_series * aligned_returns
        net_returns = gross_returns - cost_series
        participation = float(participation)

        entry_mask = (position_series.abs() > 0) & (position_series.shift(1, fill_value=0).abs() == 0)
        trade_count = int(entry_mask.sum())
        folds = evaluate_strategy(net_returns, self.cv)
        sharpes = [fold.sharpe for fold in folds]
        returns_array = net_returns.to_numpy()
        n_obs = len(returns_array)
        if n_obs > 1:
            mean_return = returns_array.mean()
            std_return = returns_array.std(ddof=1)
            sharpe_total = mean_return / std_return if std_return > 0 else float("nan")
            demeaned = returns_array - mean_return
            if std_return > 0:
                skewness = (demeaned**3).mean() / (std_return**3)
                kurtosis = (demeaned**4).mean() / (std_return**4)
            else:
                skewness = 0.0
                kurtosis = 3.0
        else:
            sharpe_total = float("nan")
            skewness = 0.0
            kurtosis = 3.0

        trials = self.trials if self.trials is not None else max(len(sharpes), 1)
        trials = max(int(trials), 1)
        dsr = (
            deflated_sharpe_ratio(sharpe_total, n_obs, trials, skewness=skewness, kurtosis=kurtosis)
            if trade_count >= self.min_trades and n_obs > 1 and math.isfinite(sharpe_total)
            else float("nan")
        )
        pbo = (
            probability_of_backtest_overfitting(sharpes)
            if trade_count >= self.min_trades
            else float("nan")
        )
        turnover = float(turnover_series.mean())
        breadth = float((net_returns > 0).mean())
        sr_mean = float(np.mean(sharpes)) if sharpes else float("nan")
        psr = (
            probabilistic_sharpe_ratio(sharpe_total, 0.0, n_obs, skewness=skewness, kurtosis=kurtosis)
            if n_obs > 1 and math.isfinite(sharpe_total)
            else float("nan")
        )
        metrics = {
            "mean_return": float(net_returns.mean()),
            "std_return": float(net_returns.std(ddof=0)),
            "mean_gross_return": float(gross_returns.mean()),
            "mean_cost": float(cost_series.mean()),
            "mean_participation": participation,
            "sharpe_mean": sr_mean,
            "sharpe_total": float(sharpe_total) if math.isfinite(sharpe_total) else float("nan"),
            "psr": float(psr),
            "trade_count": float(trade_count),
        }
        return EvaluationResult(dsr=dsr, pbo=pbo, turnover=float(turnover), breadth=breadth, metrics=metrics, trade_count=trade_count)
