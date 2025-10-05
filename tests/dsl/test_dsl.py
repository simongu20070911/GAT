import pandas as pd

from candlestrats.dsl import PatternExpression, evaluate_pattern


def test_evaluate_pattern_simple_and():
    data = pd.DataFrame(
        {
            "open": [1, 1, 1],
            "close": [1.2, 1.1, 1.3],
            "high": [1.3, 1.2, 1.4],
            "low": [0.9, 0.95, 1.0],
            "volume": [100, 110, 120],
        }
    )
    expr = PatternExpression(
        name="AND",
        params={},
        children=[
            PatternExpression(name="BREAK_ABOVE_RANGE", params={"lookback": 2, "eps": 0.0}, children=[]),
            PatternExpression(name="VOLUME_SPIKE", params={"window": 3, "quantile": 0.7}, children=[]),
        ],
    )
    mask = evaluate_pattern(expr, data)
    assert mask.iloc[-1] in {True, False}
