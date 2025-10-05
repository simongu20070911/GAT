import pandas as pd

from candlestrats.bars import DecisionBarConfig
from candlestrats.data import MinuteBarStore
from candlestrats.data.ingestion import BinanceIngestionSpec
from candlestrats.features.motifs import MotifSpec
from candlestrats.pipeline import PipelineConfig, run_pipeline
from candlestrats.utils import generate_random_walk_ohlcv


class DummyStore(MinuteBarStore):
    def __init__(self, frame):
        self.frame = frame
        spec = BinanceIngestionSpec(symbol="BTCUSDT")
        spec.raw_path = lambda: "unused"  # type: ignore[assignment]
        super().__init__([spec])

    def load(self, symbol: str, start=None, end=None):  # type: ignore[override]
        frame = self.frame.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        return frame


def test_run_pipeline():
    frame = generate_random_walk_ohlcv(240)
    store = DummyStore(frame)
    config = PipelineConfig(
        symbol="BTCUSDT",
        bar=DecisionBarConfig(target_median_seconds=3600),
        motif=MotifSpec(window=24, n_clusters=3),
    )
    result = run_pipeline(store, config)
    assert not result.minute_bars.empty
    assert not result.decision_bars.empty
    assert not result.labels.empty
    assert result.features.shape[0] == result.decision_bars.shape[0]
    assert result.motifs is not None
