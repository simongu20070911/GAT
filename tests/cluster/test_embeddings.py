import pandas as pd
import numpy as np

from candlestrats.cluster import PatternStats, build_feature_matrix, jaccard_distance_matrix, normalize_metrics, cluster_patterns


def test_build_feature_matrix_and_jaccard():
    patterns = [
        PatternStats(name="rule1", items=["A", "B"], metrics={"dsr": 0.5}),
        PatternStats(name="rule2", items=["A", "C"], metrics={"dsr": 0.3}),
    ]
    matrix = build_feature_matrix(patterns)
    assert set(matrix.columns) >= {"A", "B", "C", "dsr"}
    distances = jaccard_distance_matrix(matrix[["A", "B", "C"]])
    assert distances.shape == (2, 2)
    assert np.isfinite(distances).all()
    clusters = cluster_patterns(matrix, ["dsr"])
    assert len(clusters) == 2


def test_normalize_metrics():
    frame = pd.DataFrame({"dsr": [0.5, 0.3], "breadth": [0.7, 0.4]})
    normed = normalize_metrics(frame, ["dsr", "breadth"])
    assert np.isfinite(normed.values).all()
