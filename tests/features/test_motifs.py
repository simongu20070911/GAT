import pandas as pd

from candlestrats.features import compute_morphology_features
from candlestrats.features.motifs import MotifSpec, discover_motifs, encode_motif_hits, symbolic_state_sequence


def test_discover_and_encode_motifs(random_ohlcv):
    features = compute_morphology_features(random_ohlcv)
    spec = MotifSpec(window=24, n_clusters=3)
    motifs = discover_motifs(features, spec)
    hits = encode_motif_hits(motifs, universe=range(3))
    expected_len = max(len(features) - spec.window + 1, 0)
    assert {"timestamp", "motif_id"}.issubset(motifs.columns)
    assert len(motifs) == expected_len
    assert hits.filter(like="motif_").sum().sum() == len(hits)


def test_symbolic_state_sequence():
    returns = pd.Series([0.002, -0.002, 0.0])
    states = symbolic_state_sequence(returns)
    assert list(states) == ["U", "D", "F"]
