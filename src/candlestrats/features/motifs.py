"""Motif discovery and encoding."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


@dataclass
class MotifSpec:
    """Configuration for morphology motif extraction."""

    window: int
    n_clusters: int
    feature_columns: Sequence[str] | None = None


def discover_motifs(feature_frame: pd.DataFrame, spec: MotifSpec) -> pd.DataFrame:
    """Cluster morphology vectors to discover frequent motifs."""
    if feature_frame.empty:
        raise ValueError("Cannot build motifs from empty feature frame")
    features = feature_frame.copy()
    if "timestamp" not in features.columns:
        raise KeyError("Feature frame must include a 'timestamp' column")
    features["timestamp"] = pd.to_datetime(features["timestamp"], utc=True)
    features = features.set_index("timestamp").sort_index()

    cols = spec.feature_columns or list(features.columns)
    selected = features[cols].apply(pd.to_numeric, errors="coerce")
    window = max(int(spec.window), 1)
    if window > len(selected):
        raise ValueError("Motif window exceeds available observations")

    rolling = selected.rolling(window=window, min_periods=window)
    agg_frames = [
        rolling.mean().add_suffix("_mean"),
        rolling.std(ddof=0).add_suffix("_std"),
        rolling.min().add_suffix("_min"),
        rolling.max().add_suffix("_max"),
        selected.add_suffix("_last"),
    ]
    aggregated = pd.concat(agg_frames, axis=1).dropna()
    if aggregated.empty:
        raise ValueError("Insufficient data after applying motif window")

    matrix = aggregated.to_numpy()
    if matrix.ndim != 2:
        raise ValueError("Feature matrix must be 2D")
    km = KMeans(n_clusters=spec.n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(matrix)
    motifs = pd.DataFrame({"timestamp": aggregated.index, "motif_id": labels}).reset_index(drop=True)
    return motifs


def encode_motif_hits(motifs: pd.DataFrame, universe: Iterable[int]) -> pd.DataFrame:
    """Convert motif assignments into binary hits."""
    hits = pd.DataFrame({"timestamp": motifs["timestamp"]})
    for motif_id in universe:
        hits[f"motif_{motif_id}"] = (motifs["motif_id"] == motif_id).astype(int)
    return hits


def symbolic_state_sequence(returns: pd.Series, thresholds: tuple[float, float] = (-0.001, 0.001)) -> pd.Series:
    """Discretize returns into symbolic U/D/F states."""
    lower, upper = thresholds
    states = np.where(returns > upper, "U", np.where(returns < lower, "D", "F"))
    return pd.Series(states, index=returns.index, name="state")
