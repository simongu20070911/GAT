"""Sequence and quantifier operators."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class SequenceOperator:
    window: int
    tolerance: int = 0

    def evaluate(self, series_list: Iterable[pd.Series]) -> pd.Series:
        masks = [series.astype(bool) for series in series_list]
        if not masks:
            raise ValueError("No predicates provided for sequence evaluation")

        index = masks[0].index
        masks = [mask.reindex(index, fill_value=False) for mask in masks]
        arrays = [mask.to_numpy(dtype=bool) for mask in masks]
        length = len(index)
        window = max(self.window, len(masks))
        tolerance = max(self.tolerance, 0)

        # Pre-compute next true indices for each predicate to avoid inner scans.
        next_true = []
        for arr in arrays:
            nxt = np.full(length + 1, length, dtype=np.int32)
            for i in range(length - 1, -1, -1):
                nxt[i] = i if arr[i] else nxt[i + 1]
            next_true.append(nxt)

        result = np.zeros(length, dtype=bool)
        for t in range(length):
            start = max(0, t - window + 1)
            end = t + 1
            current = start
            misses = 0
            satisfied = 0
            for nxt in next_true:
                pos = nxt[current]
                if pos < end:
                    current = pos + 1
                    satisfied += 1
                else:
                    misses += 1
                    if misses > tolerance:
                        break
            if misses <= tolerance and (satisfied + misses) == len(arrays):
                result[t] = True

        return pd.Series(result, index=index)


@dataclass
class QuantifierOperator:
    window: int
    threshold: int

    def evaluate(self, series: pd.Series) -> pd.Series:
        counts = series.astype(int).rolling(window=self.window, min_periods=1).sum()
        return counts >= self.threshold
