"""Frequent motif mining over predicate truth tables."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class Motif:
    items: Tuple[str, ...]
    support: float
    lift: float
    purity: float
    direction: int


class FrequentMotifMiner:
    """Simple Apriori-style miner over binary predicate tables."""

    def __init__(self, min_support: float = 0.01, min_lift: float = 1.1) -> None:
        self.min_support = min_support
        self.min_lift = min_lift

    def mine(self, predicates: pd.DataFrame, labels: pd.Series, max_size: int = 3) -> List[Motif]:
        binary = predicates.astype(bool)
        total = len(binary)
        label_positive = labels == 1
        label_negative = labels == -1
        base_positive = float(label_positive.mean()) or 1e-6
        base_negative = float(label_negative.mean()) or 1e-6

        candidates: Dict[Tuple[str, ...], float] = {}
        supports: Dict[Tuple[str, ...], np.ndarray] = {}
        motifs: List[Motif] = []

        for column in binary.columns:
            mask = binary[column].to_numpy(dtype=bool)
            support = mask.mean()
            if support >= self.min_support:
                key = (column,)
                candidates[key] = support
                supports[key] = mask

        size = 1
        while candidates and size <= max_size:
            new_candidates: Dict[Tuple[str, ...], float] = {}
            for items, support in candidates.items():
                mask = supports.get(items)
                if mask is None:
                    mask_df = binary[list(items)].all(axis=1)
                    mask = mask_df.to_numpy(dtype=bool)
                    supports[items] = mask
                occ = int(mask.sum())
                if occ == 0:
                    continue
                support = occ / total
                if support < self.min_support:
                    continue
                if base_positive > 0:
                    pos_purity = float((labels[mask] == 1).mean()) if occ else 0.0
                    pos_lift = pos_purity / base_positive if base_positive else 0.0
                    if pos_lift >= self.min_lift and pos_purity > 0:
                        motifs.append(
                            Motif(items=items, support=support, lift=pos_lift, purity=pos_purity, direction=1)
                        )
                if base_negative > 0:
                    neg_purity = float((labels[mask] == -1).mean()) if occ else 0.0
                    neg_lift = neg_purity / base_negative if base_negative else 0.0
                    if neg_lift >= self.min_lift and neg_purity > 0:
                        motifs.append(
                            Motif(items=items, support=support, lift=neg_lift, purity=neg_purity, direction=-1)
                        )
                if size < max_size:
                    last_item = items[-1]
                    for next_item in binary.columns:
                        if next_item <= last_item or next_item in items:
                            continue
                        base_mask = supports[items]
                        next_mask = supports.get((next_item,))
                        if next_mask is None:
                            next_mask = binary[next_item].to_numpy(dtype=bool)
                            supports[(next_item,)] = next_mask
                        combined_mask = base_mask & next_mask
                        combined_support = combined_mask.mean()
                        if combined_support < self.min_support:
                            continue
                        new_key = tuple(sorted(items + (next_item,)))
                        if new_key in new_candidates:
                            continue
                        supports[new_key] = combined_mask
                        new_candidates[new_key] = combined_support
            candidates = new_candidates
            size += 1

        return motifs
