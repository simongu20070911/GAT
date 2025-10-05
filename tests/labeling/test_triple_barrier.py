import pandas as pd

from candlestrats.bars import DecisionBarConfig, build_decision_bars
from candlestrats.labeling import (
    LabelDiagnostics,
    TripleBarrierConfig,
    apply_triple_barrier,
    summarize_label_distribution,
)


def test_apply_triple_barrier(random_ohlcv):
    frame = random_ohlcv.rename(columns={"timestamp": "timestamp"})
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    bars = build_decision_bars(frame, DecisionBarConfig(target_median_seconds=3600)).bars
    labels = apply_triple_barrier(bars, TripleBarrierConfig())
    assert {"bar_id", "y", "up_bar", "dn_bar", "hit"}.issubset(labels.columns)
    diagnostics = summarize_label_distribution(labels)
    assert isinstance(diagnostics, LabelDiagnostics)
