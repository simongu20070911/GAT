"""Feature registry for reproducibility."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import pandas as pd


@dataclass
class FeatureRecord:
    feature_id: str
    params: Dict[str, object]
    graph_hash: str


class FeatureRegistry:
    """In-memory registry to track feature provenance."""

    def __init__(self) -> None:
        self._records: Dict[str, FeatureRecord] = {}

    @staticmethod
    def _hash_params(params: Dict[str, object]) -> str:
        payload = json.dumps(params, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def register(self, feature_id: str, params: Dict[str, object]) -> None:
        if feature_id in self._records:
            raise KeyError(f"Feature {feature_id} already registered")
        graph_hash = self._hash_params(params)
        self._records[feature_id] = FeatureRecord(feature_id=feature_id, params=params, graph_hash=graph_hash)

    def get(self, feature_id: str) -> Optional[FeatureRecord]:
        return self._records.get(feature_id)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                feature_id: {
                    "params": record.params,
                    "graph_hash": record.graph_hash,
                }
                for feature_id, record in self._records.items()
            }
        ).T.reset_index().rename(columns={"index": "feature_id"})

    def list_ids(self) -> Iterable[str]:
        return list(self._records.keys())
