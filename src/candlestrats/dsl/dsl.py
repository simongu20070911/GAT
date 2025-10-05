"""Pattern DSL parsing and evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd

from candlestrats.dsl.predicates import Predicate, registry
from candlestrats.dsl.sequences import QuantifierOperator, SequenceOperator


@dataclass
class PatternExpression:
    name: str
    params: Dict[str, Any]
    children: List["PatternExpression"]

    def evaluate(self, data: pd.DataFrame, context: Dict[str, pd.Series]) -> pd.Series:
        if self.name == "AND":
            masks = [child.evaluate(data, context) for child in self.children]
            return pd.concat(masks, axis=1).all(axis=1)
        if self.name == "OR":
            masks = [child.evaluate(data, context) for child in self.children]
            return pd.concat(masks, axis=1).any(axis=1)
        if self.name == "NOT":
            return ~self.children[0].evaluate(data, context)
        if self.name == "SEQUENCE":
            seq = SequenceOperator(window=self.params.get("window", len(self.children)), tolerance=self.params.get("tolerance", 0))
            masks = [child.evaluate(data, context) for child in self.children]
            return seq.evaluate(masks)
        if self.name == "COUNT":
            child_mask = self.children[0].evaluate(data, context)
            quantifier = QuantifierOperator(window=self.params.get("window", 3), threshold=self.params.get("threshold", 1))
            return quantifier.evaluate(child_mask)
        return evaluate_atomic(self, data, context)


SUPPORTED = {"AND", "OR", "NOT", "SEQUENCE", "COUNT"}


def evaluate_atomic(expr: PatternExpression, data: pd.DataFrame, context: Dict[str, pd.Series]) -> pd.Series:
    if expr.name in context:
        return context[expr.name]
    params = expr.params.copy()
    predicate_fn = registry.get(expr.name)
    predicate = Predicate(name=expr.name, params=params, evaluator=predicate_fn)
    return predicate.evaluate(data)


def parse_pattern(definition: Dict[str, Any]) -> PatternExpression:
    name = definition.get("name")
    params = definition.get("params", {})
    children_defs = definition.get("children", [])
    children = [parse_pattern(child) for child in children_defs]
    return PatternExpression(name=name, params=params, children=children)


def evaluate_pattern(pattern: PatternExpression, data: pd.DataFrame, context: Dict[str, pd.Series] | None = None) -> pd.Series:
    context = context or {}
    return pattern.evaluate(data, context)
