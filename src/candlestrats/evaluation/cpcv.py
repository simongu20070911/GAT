"""Combinatorial purged cross-validation harness."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, List

import numpy as np
import pandas as pd


@dataclass
class FoldResult:
    fold_id: int
    sharpe: float
    abs_return_mean: float
    pnl: pd.Series


@dataclass
class CombinatorialPurgedCV:
    n_splits: int
    embargo_minutes: int
    test_blocks: int = 1

    def split(self, timestamps: pd.Series) -> Iterable[tuple[np.ndarray, np.ndarray]]:
        ts = pd.to_datetime(timestamps, utc=True)
        indices = np.arange(len(ts))
        folds = [fold for fold in np.array_split(indices, self.n_splits) if len(fold) > 0]
        if not folds:
            return

        embargo = pd.Timedelta(minutes=self.embargo_minutes)
        test_blocks = max(1, min(self.test_blocks, len(folds)))

        for combo in combinations(range(len(folds)), test_blocks):
            test_idx = np.concatenate([folds[i] for i in combo])
            test_idx.sort()
            if test_idx.size == 0:
                continue

            train_mask = np.ones(len(ts), dtype=bool)
            for block_index in combo:
                fold = folds[block_index]
                fold_start = ts.iloc[fold[0]]
                fold_end = ts.iloc[fold[-1]]
                left_cut = fold_start - embargo
                right_cut = fold_end + embargo
                train_mask &= ((ts < left_cut) | (ts > right_cut)).to_numpy()

            train_idx = indices[train_mask]
            yield train_idx, test_idx


def evaluate_strategy(returns: pd.Series, cv: CombinatorialPurgedCV) -> List[FoldResult]:
    """Evaluate a return series under CPCV."""
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("Returns must be indexed by timestamps for CPCV evaluation")
    timestamps = returns.index
    fold_results = []
    for fold_id, (train_idx, test_idx) in enumerate(cv.split(pd.Series(timestamps))):
        test_returns = returns.iloc[test_idx]
        if test_returns.std(ddof=0) == 0:
            sharpe = 0.0
        else:
            diffs = test_returns.index.to_series().diff().dropna().dt.total_seconds()
            if diffs.empty:
                periods_per_year = 252.0
            else:
                median_seconds = max(float(diffs.median()), 1.0)
                periods_per_year = 365.25 * 24 * 3600 / median_seconds
            sharpe = test_returns.mean() / test_returns.std(ddof=0) * np.sqrt(periods_per_year)
        abs_mean = test_returns.abs().mean()
        fold_results.append(
            FoldResult(
                fold_id=fold_id,
                sharpe=float(sharpe),
                abs_return_mean=float(abs_mean),
                pnl=test_returns,
            )
        )
    return fold_results
