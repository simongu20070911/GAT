"""Pattern embedding and clustering utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize


@dataclass
class PatternStats:
    name: str
    items: Sequence[str]
    metrics: Dict[str, float]


def build_feature_matrix(patterns: Iterable[PatternStats]) -> pd.DataFrame:
    pattern_list = list(patterns)
    all_items = sorted({item for pattern in pattern_list for item in pattern.items})
    data = []
    for pattern in pattern_list:
        row = {item: 1 for item in pattern.items}
        row.update(pattern.metrics)
        data.append(row)
    frame = pd.DataFrame(data).fillna(0.0)
    frame.index = [pattern.name for pattern in pattern_list]
    return frame


def jaccard_distance_matrix(binary_matrix: pd.DataFrame) -> np.ndarray:
    binary = binary_matrix.astype(bool).astype(int)
    return pairwise_distances(binary.values, metric="jaccard")


def normalize_metrics(frame: pd.DataFrame, metric_cols: Iterable[str]) -> pd.DataFrame:
    normalized = frame.copy()
    cols = list(metric_cols)
    if cols:
        normalized[cols] = normalize(frame[cols], axis=0)
    return normalized


def cluster_patterns(matrix: pd.DataFrame, metric_cols: Iterable[str]) -> Dict[str, Dict[str, float]]:
    normalized = normalize_metrics(matrix, metric_cols)
    return normalized.to_dict(orient="index")
