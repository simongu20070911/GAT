"""Strategy ensemble scaffolding."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from candlestrats.evaluation.costs import FeeModel


@dataclass
class EnsembleConfig:
    target_vol: float = 0.1
    turnover_cap: float = 0.5
    fee_model: FeeModel | None = None


def build_ensemble(signal_panels: Dict[str, pd.DataFrame], config: EnsembleConfig) -> pd.DataFrame:
    """Combine candidate signals by inverse-vol weighting with turnover control."""
    if not signal_panels:
        raise ValueError("No signal panels supplied")

    scores = {}
    weights = {}
    for name, panel in signal_panels.items():
        series = panel["score"].astype(float)
        scores[name] = series
        vol = series.std(ddof=0)
        weights[name] = 0.0 if vol == 0 else 1.0 / vol

    total_weight = sum(weights.values()) or 1.0
    for key in weights:
        weights[key] /= total_weight

    combined = sum(weights[name] * scores[name] for name in scores)
    combined = combined.fillna(0.0)
    turnover = combined.diff().abs().fillna(0.0)
    turnover = turnover.clip(upper=config.turnover_cap)

    fee_model = config.fee_model or FeeModel()
    participation = turnover.clip(lower=0).values if hasattr(turnover, "values") else turnover
    cost_array = fee_model.cost_bps(maker=False, participation=float(participation.mean())) / 10000
    fee_adjusted = combined - np.sign(combined) * cost_array

    return pd.DataFrame(
        {
            "ensemble_score": combined,
            "turnover": turnover,
            "fee_adjusted_score": fee_adjusted,
            "weight_sum": sum(weights.values()),
        }
    )
