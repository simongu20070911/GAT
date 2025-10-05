"""Grammar utilities for DSL-based GP."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from candlestrats.dsl import PatternExpression, parse_pattern


@dataclass
class SeedPattern:
    name: str
    definition: dict

    def expression(self) -> PatternExpression:
        return parse_pattern(self.definition)


STOP_RUN_RECLAIM = SeedPattern(
    name="stop_run_reclaim",
    definition={
        "name": "AND",
        "children": [
            {"name": "LONG_LOWER_WICK", "params": {"threshold": 0.4}},
            {"name": "VOLUME_SPIKE", "params": {"window": 20, "quantile": 0.8}},
            {"name": "CLOSE_ABOVE_PRIOR_LOW", "params": {"lag": 1}},
        ],
    },
)

TREND_PULLBACK = SeedPattern(
    name="trend_pullback",
    definition={
        "name": "AND",
        "children": [
            {"name": "HIGHER_LOW", "params": {"depth": 2}},
            {"name": "RANGE_COMPRESSION", "params": {"window": 5, "quantile": 0.2}},
        ],
    },
)


PSYCHOLOGY_SEEDS: List[SeedPattern] = [STOP_RUN_RECLAIM, TREND_PULLBACK]
