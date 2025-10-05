"""Evaluation reporting utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd

from candlestrats.evaluation import CombinatorialPurgedCV, monte_carlo_null_distribution, permutation_mcpt


@dataclass
class MCPTSummaryInput:
    returns: pd.Series
    symbol: str
    splits: int = 6
    embargo_minutes: int = 60
    runs: int = 100
    permutation_block: int = 24


def mcpt_summary(config: MCPTSummaryInput) -> pd.DataFrame:
    cv = CombinatorialPurgedCV(n_splits=config.splits, embargo_minutes=config.embargo_minutes)
    random = monte_carlo_null_distribution(config.returns, cv, runs=config.runs)
    perm = permutation_mcpt(config.returns, cv, block_size=config.permutation_block, runs=config.runs)
    return pd.DataFrame(
        [
            {
                "symbol": config.symbol,
                "real_sharpe": random.real_metric,
                "random_null_mean": (sum(random.synthetic_metrics) / len(random.synthetic_metrics)) if random.synthetic_metrics else float("nan"),
                "random_p": random.p_value,
                "perm_null_mean": (sum(perm.synthetic_metrics) / len(perm.synthetic_metrics)) if perm.synthetic_metrics else float("nan"),
                "perm_p": perm.p_value,
                "runs": config.runs,
                "block_size": config.permutation_block,
            }
        ]
    )


def write_mcpt_summary(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_csv(path)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(path, index=False)
