"""Event-aware purged cross-validation that respects exit horizons."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import pandas as pd


@dataclass
class PurgedEventKFold:
    """K-fold splitter that purges overlapping events and applies embargo."""

    n_splits: int
    embargo: pd.Timedelta

    def split(self, events: pd.DataFrame) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        if "ts" not in events.columns or "t1" not in events.columns:
            raise KeyError("Events frame must contain 'ts' and 't1' columns")

        ts = pd.to_datetime(events["ts"], utc=True).to_numpy()
        t1 = pd.to_datetime(events["t1"], utc=True).to_numpy()
        order = np.argsort(ts)
        ts = ts[order]
        t1 = t1[order]
        indices = order

        folds = [fold for fold in np.array_split(indices, self.n_splits) if len(fold) > 0]
        if not folds:
            return

        for fold_id in range(len(folds)):
            test_idx = folds[fold_id]
            test_ts = ts[test_idx]
            test_t1 = t1[test_idx]

            min_test_ts = test_ts.min()
            max_test_t1 = test_t1.max()

            left_embargo = min_test_ts - self.embargo
            right_embargo = max_test_t1 + self.embargo

            train_mask = np.ones(len(indices), dtype=bool)
            for idx, start in enumerate(ts):
                end = t1[idx]
                if idx in test_idx:
                    train_mask[idx] = False
                    continue
                if (start <= test_t1.max()) and (end >= test_ts.min()):
                    train_mask[idx] = False
                    continue
                if start >= left_embargo and start <= right_embargo:
                    train_mask[idx] = False

            train_indices = indices[train_mask]
            yield train_indices, test_idx
