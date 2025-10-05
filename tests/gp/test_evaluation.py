import pandas as pd

from candlestrats.evaluation import CombinatorialPurgedCV, FeeModel
from candlestrats.gp.evaluation import StrategyEvaluator


def test_strategy_evaluator_outputs_metrics():
    index = pd.date_range("2024-01-01", periods=20, freq="h")
    signals = pd.Series([1, 0, -1, 1, 0] * 4, index=index)
    labels = pd.Series([0.5, -0.2, -0.3, 0.4, 0.1] * 4, index=index)
    cv = CombinatorialPurgedCV(n_splits=4, embargo_minutes=1)
    evaluator = StrategyEvaluator(cv=cv, fee_model=FeeModel(), min_trades=1)
    result = evaluator.evaluate_signals(signals, labels)
    assert result.dsr == result.dsr
    assert "mean_return" in result.metrics
