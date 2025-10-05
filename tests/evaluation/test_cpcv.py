import pandas as pd

from candlestrats.evaluation import CombinatorialPurgedCV, evaluate_strategy


def test_evaluate_strategy_returns_fold_results():
    index = pd.date_range(start="2024-01-01", periods=120, freq="min")
    returns = pd.Series(0.001, index=index)
    cv = CombinatorialPurgedCV(n_splits=4, embargo_minutes=5)
    folds = evaluate_strategy(returns, cv)
    assert len(folds) == 4
    assert all(fold.pnl.index.isin(index).all() for fold in folds)
