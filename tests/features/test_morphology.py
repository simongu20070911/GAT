import numpy as np

from candlestrats.features import compute_morphology_features


def test_compute_morphology_features(random_ohlcv):
    frame = random_ohlcv
    features = compute_morphology_features(frame)
    assert "tod_sin" in features.columns
    assert "w60_drawdown" in features.columns
    assert len(features) == len(frame)
    assert np.all(np.isfinite(features[["tod_sin", "tod_cos"]].to_numpy()))
