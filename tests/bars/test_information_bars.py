import pandas as pd

from candlestrats.bars import DecisionBarConfig, build_decision_bars


def test_build_decision_bars(random_ohlcv):
    frame = random_ohlcv.rename(columns={"timestamp": "timestamp"})
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    result = build_decision_bars(frame, DecisionBarConfig(target_median_seconds=3600))
    bars = result.bars
    mapping = result.mapping
    assert not bars.empty
    assert {"ts", "open", "close", "bar_id"}.issubset(bars.columns)
    assert not mapping.empty
    assert {"timestamp", "bar_id"}.issubset(mapping.columns)


def test_event_bar_preference_when_within_tolerance(random_ohlcv):
    frame = random_ohlcv.rename(columns={"timestamp": "timestamp"})
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    config = DecisionBarConfig(target_median_seconds=3600, event_preference_tolerance=10.0)
    result = build_decision_bars(frame, config)
    assert result.method in {"dollar", "imbalance"}
    assert not result.bars.empty
