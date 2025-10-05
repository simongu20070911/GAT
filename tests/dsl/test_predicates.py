import pandas as pd

from candlestrats.dsl.predicates import (
    Predicate,
    evaluate_predicates,
    wick_ratio_predicate,
    inside_bar_predicate,
    outside_bar_predicate,
    bullish_engulf_predicate,
    pivot_high_predicate,
    pivot_low_predicate,
    round_level_proximity_predicate,
)


def test_wick_ratio_predicate_lower():
    df = pd.DataFrame(
        {
            "open": [1, 1, 1],
            "close": [1.1, 0.9, 1.2],
            "high": [1.2, 1.1, 1.3],
            "low": [0.8, 0.7, 0.9],
            "volume": [100, 120, 130],
        }
    )
    pred = Predicate(name="LOW_WICK", params={"kind": "lower", "threshold": 0.4}, evaluator=wick_ratio_predicate)
    result = pred.evaluate(df)
    assert result.iloc[0]


def test_evaluate_predicates_multiple():
    df = pd.DataFrame(
        {
            "open": [1, 1, 1],
            "close": [1.2, 1.1, 0.8],
            "high": [1.3, 1.2, 1.0],
            "low": [0.9, 0.95, 0.7],
            "volume": [100, 105, 110],
        }
    )
    predicates = [
        Predicate(name="LONG_LOWER_WICK", params={"kind": "lower", "threshold": 0.3}, evaluator=wick_ratio_predicate)
    ]
    frame = evaluate_predicates(df, predicates)
    assert "LONG_LOWER_WICK" in frame


def test_inside_outside_engulf():
    df = pd.DataFrame({
        "open": [1, 1.1, 1.2],
        "close": [1.05, 1.0, 1.3],
        "high": [1.1, 1.15, 1.35],
        "low": [0.95, 0.9, 1.1],
        "volume": [100, 120, 150],
    })
    inside = Predicate(name="INSIDE", params={}, evaluator=inside_bar_predicate)
    outside = Predicate(name="OUTSIDE", params={}, evaluator=outside_bar_predicate)
    bull = Predicate(name="BULL", params={}, evaluator=bullish_engulf_predicate)
    res_inside = inside.evaluate(df)
    res_outside = outside.evaluate(df)
    res_bull = bull.evaluate(df)
    assert isinstance(res_inside, pd.Series)
    assert isinstance(res_outside, pd.Series)
    assert isinstance(res_bull, pd.Series)


from candlestrats.dsl.predicates import pivot_high_predicate, pivot_low_predicate


def test_pivot_predicates():
    df = pd.DataFrame({
        "high": [1, 1.2, 1.4, 1.1, 1.05],
        "low": [0.9, 0.95, 1.0, 0.85, 0.8],
        "open": [0.95]*5,
        "close": [1.0]*5,
        "volume": [100]*5,
    })
    high = pivot_high_predicate(df, {"left": 1, "right": 1})
    low = pivot_low_predicate(df, {"left": 1, "right": 1})
    assert isinstance(high, pd.Series)
    assert isinstance(low, pd.Series)


def test_round_level_percent_grid_alignment():
    df = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0],
            "close": [100.0, 101.0, 100.5],
            "high": [100.2, 101.2, 100.7],
            "low": [99.8, 100.8, 100.3],
            "volume": [1000, 1000, 1000],
        }
    )
    mask = round_level_proximity_predicate(
        df,
        {"percent": 0.01, "tolerance": 0.2, "atr_window": 1},
    )
    assert mask.iloc[0]
    assert mask.iloc[1]
    assert not mask.iloc[2]
