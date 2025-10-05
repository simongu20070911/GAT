import pandas as pd

from candlestrats.classifiers import CandlestickStateClassifier


def test_candlestick_state_classifier_predicts():
    features = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=10, freq="h"),
        "feat_a": range(10),
        "feat_b": range(10, 20),
    })
    labels = pd.Series(["upturn", "downturn"] * 5)
    clf = CandlestickStateClassifier()
    clf.fit(features, labels)
    states = clf.predict(features)
    assert set(states.unique()).issubset({"upturn", "downturn"})
    signal = clf.to_signal(states)
    assert not signal.empty
