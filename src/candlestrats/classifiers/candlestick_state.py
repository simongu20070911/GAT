"""Candlestick state classifier."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier


@dataclass
class CandlestickStateClassifier:
    """Wrap a scikit-learn classifier for candlestick state labeling."""

    model: BaseEstimator = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> None:
        train_features = features.drop(columns=["timestamp"], errors="ignore")
        self.model.fit(train_features, labels)

    def predict(self, features: pd.DataFrame) -> pd.Series:
        eval_features = features.drop(columns=["timestamp"], errors="ignore")
        preds = self.model.predict(eval_features)
        return pd.Series(preds, index=features.index, name="state")

    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        eval_features = features.drop(columns=["timestamp"], errors="ignore")
        probs = self.model.predict_proba(eval_features)
        classes = getattr(self.model, "classes_", [])
        return pd.DataFrame(probs, index=features.index, columns=classes)

    def to_signal(self, states: pd.Series, mapping: Dict[str, float] | None = None) -> pd.Series:
        mapping = mapping or {"upturn": 1.0, "downturn": -1.0}
        return states.map(mapping).fillna(0.0)
