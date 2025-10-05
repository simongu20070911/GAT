import pandas as pd

from candlestrats.portfolio import EnsembleConfig, build_ensemble


def test_build_ensemble_combines_signals():
    index = pd.date_range("2024-01-01", periods=5, freq="h")
    panel_a = pd.DataFrame({"score": [0.1, 0.2, -0.1, 0.05, 0.0]}, index=index)
    panel_b = pd.DataFrame({"score": [-0.05, 0.1, 0.15, -0.2, 0.05]}, index=index)
    df = build_ensemble({"a": panel_a, "b": panel_b}, EnsembleConfig())
    assert "ensemble_score" in df.columns
    assert "fee_adjusted_score" in df.columns
    assert (df["turnover"] <= EnsembleConfig().turnover_cap + 1e-9).all()
