"""Prototype clustering utilities for GP rules."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

from candlestrats.cluster.embeddings import PatternStats, build_feature_matrix, normalize_metrics


TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def tokenize_expression(expression: str) -> List[str]:
    tokens = set(TOKEN_PATTERN.findall(expression))
    return sorted(tokens)


@dataclass
class Prototype:
    cluster_id: int
    rule_name: str
    expression: str
    metrics: Dict[str, float]


def load_rule_records(json_path: Path) -> List[Dict[str, object]]:
    import json

    with json_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def build_pattern_stats(rule_records: Iterable[Dict[str, object]]) -> tuple[List[PatternStats], Dict[str, str]]:
    stats: List[PatternStats] = []
    expressions: Dict[str, str] = {}
    for record in rule_records:
        name = str(record["name"])
        expression = str(record["expression"])
        metadata = record.get("metadata", {}) or {}
        metrics = {
            key: float(value)
            for key, value in metadata.items()
            if isinstance(value, (int, float)) and not np.isnan(value)
        }
        tokens = tokenize_expression(expression)
        stats.append(PatternStats(name=name, items=tuple(tokens + list(metadata.keys())), metrics=metrics))
        expressions[name] = expression
    return stats, expressions


def cluster_rules(rule_records: Iterable[Dict[str, object]], metric_columns: Iterable[str]) -> List[Prototype]:
    stats, expressions = build_pattern_stats(rule_records)
    if not stats:
        return []
    matrix = build_feature_matrix(stats)
    metric_columns = [col for col in metric_columns if col in matrix.columns]
    matrix = normalize_metrics(matrix, metric_columns)
    embedding = matrix.fillna(0.0).to_numpy()
    if embedding.shape[0] <= 1:
        cluster_labels = np.zeros(embedding.shape[0], dtype=int)
    else:
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=1.5)
        cluster_labels = clustering.fit_predict(embedding)
    prototypes: List[Prototype] = []
    for label, row in zip(cluster_labels, matrix.itertuples()):
        rule_name = row.Index
        expression = expressions.get(rule_name, "")
        metrics = matrix.loc[rule_name, metric_columns].to_dict() if metric_columns else {}
        prototypes.append(Prototype(cluster_id=int(label), rule_name=rule_name, expression=expression, metrics=metrics))
    return prototypes
