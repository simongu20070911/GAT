"""Meta-labeling helpers."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class MetaLabelConfig:
    threshold: float = 0.5


def apply_meta_label(signals: pd.DataFrame, features: pd.DataFrame, config: MetaLabelConfig) -> pd.Series:
    """Return a binary take/skip meta-label."""
    score = signals["score"].fillna(0)
    return (score.abs() > config.threshold).astype(int)
