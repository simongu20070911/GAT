"""Genetic programming miner skeleton."""
from __future__ import annotations

import ast
import copy
import operator
import random
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

from candlestrats.dsl import PatternExpression, evaluate_pattern, parse_pattern
from candlestrats.dsl.predicates import Predicate, registry as predicate_registry
from candlestrats.gp.grammar import PSYCHOLOGY_SEEDS

SAFE_OPERATORS = {
    "+": operator.add,
        "*": operator.mul,
}


def safe_div(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    numerator = np.asarray(x, dtype=float)
    denominator = np.asarray(y, dtype=float)
    adjusted = np.where(np.abs(denominator) < 1e-6, np.sign(denominator) * 1e-6 + 1e-6, denominator)
    return np.divide(numerator, adjusted)


def logical_and(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.logical_and(_to_bool(x), _to_bool(y))


def logical_or(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.logical_or(_to_bool(x), _to_bool(y))


def greater(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.greater(x, y)


def greater_equal(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.greater_equal(x, y)


def less(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.less(x, y)


def less_equal(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.less_equal(x, y)


def equal(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.equal(x, y)


SAFE_FUNCTIONS = {
    "div": safe_div,
    "tanh": np.tanh,
    "AND": logical_and,
    "OR": logical_or,
    "GT": greater,
    "GE": greater_equal,
    "LT": less,
    "LE": less_equal,
    "EQ": equal,
}


COMPARATOR_NAMES = ("GT", "GE", "LT", "LE", "EQ")


def _to_bool(arr: np.ndarray) -> np.ndarray:
    data = np.asarray(arr)
    if data.dtype == bool:
        return data
    return np.where(np.isnan(data), False, data != 0)


_ALLOWED_SUBTREE_TYPES = (
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.Call,
    ast.Name,
    ast.Constant,
)


def _parse_expression_tree(expression: str) -> ast.Expression | None:
    try:
        return ast.parse(expression, mode="eval")
    except SyntaxError:
        return None


def _safe_unparse(tree: ast.Expression) -> str:
    expr = copy.deepcopy(tree)
    ast.fix_missing_locations(expr)
    return ast.unparse(expr.body)


def _iter_nodes_with_parents(node: ast.AST, parent: ast.AST | None = None):
    yield parent, node
    for child in ast.iter_child_nodes(node):
        yield from _iter_nodes_with_parents(child, node)


def _is_function_name(node: ast.Name, parent: ast.AST | None) -> bool:
    return isinstance(parent, ast.Call) and parent.func is node


def _collect_candidate_subtrees(tree: ast.Expression, feature_names: Tuple[str, ...]) -> List[ast.AST]:
    names = set(feature_names)
    candidates: List[ast.AST] = []
    for parent, node in _iter_nodes_with_parents(tree.body):
        if isinstance(node, ast.Name):
            if _is_function_name(node, parent):
                continue
            if node.id not in names:
                continue
            candidates.append(node)
        elif isinstance(node, _ALLOWED_SUBTREE_TYPES):
            candidates.append(node)
    if not candidates:
        candidates.append(tree.body)
    return candidates


def _replace_subtree(tree: ast.Expression, target: ast.AST, replacement: ast.AST) -> ast.Expression:
    target_id = id(target)

    class _Replacer(ast.NodeTransformer):
        def visit(self, node: ast.AST) -> ast.AST:
            if id(node) == target_id:
                return copy.deepcopy(replacement)
            return super().generic_visit(node)

    tree.body = _Replacer().visit(tree.body)
    return tree


@dataclass
class GeneticProgramConfig:
    population_size: int = 128
    elite_fraction: float = 0.1
    max_depth: int = 5
    n_generations: int = 25
    mutation_rate: float = 0.2
    crossover_rate: float = 0.6
    feature_columns: Tuple[str, ...] = ()
    turnover_weight: float = 1.0
    breadth_weight: float = 1.0


@dataclass
class CandidateRule:
    expression: str
    fitness: float | None = None
    metadata: dict[str, float] | None = None


class GeneticProgramMiner:
    """Simple GP miner for morphology features."""

    def __init__(self, config: GeneticProgramConfig, strategy_evaluator=None) -> None:
        if not config.feature_columns:
            raise ValueError("GeneticProgramConfig requires feature_columns")
        self.config = config
        self.strategy_evaluator = strategy_evaluator
        random.seed(42)

    def seed_population(self) -> List[CandidateRule]:
        seeded: List[CandidateRule] = []
        for seed in PSYCHOLOGY_SEEDS:
            feature_name = f"seed_{seed.name}"
            if feature_name in self.config.feature_columns and feature_name not in {c.expression for c in seeded}:
                seeded.append(CandidateRule(expression=feature_name))

        remaining = self.config.population_size - len(seeded)
        if remaining > 0:
            seeded.extend(
                CandidateRule(expression=self._random_expression(self.config.max_depth))
                for _ in range(remaining)
            )
        return seeded[: self.config.population_size]

    def evolve(self, features: pd.DataFrame, labels: pd.Series) -> CandidateRule:
        population = self.seed_population()
        for _ in range(self.config.n_generations):
            evaluated = self.evaluate(population, features, labels)
            population = self._next_generation(evaluated)
        final = self.evaluate(population, features, labels)
        return max(final, key=lambda rule: rule.fitness or -np.inf)

    def evaluate(self, rules: Iterable[CandidateRule], features: pd.DataFrame, labels: pd.Series | pd.DataFrame) -> List[CandidateRule]:
        evaluated = []
        for rule in rules:
            try:
                signal = self._evaluate_expression(rule.expression, features)
                score, meta = self._assess(signal, labels)
            except Exception as exc:  # pragma: no cover - defensive guard
                score, meta = -np.inf, {"error": str(exc)}
            evaluated.append(
                CandidateRule(
                    expression=rule.expression,
                    fitness=float(score),
                    metadata=meta if isinstance(meta, dict) else None,
                )
            )
        return evaluated

    def _assess(self, signal: pd.Series, labels: pd.Series | pd.DataFrame) -> tuple[float, dict[str, float]]:
        if isinstance(labels, pd.DataFrame):
            if "ts" in labels.columns:
                label_index = pd.to_datetime(labels["ts"], utc=True)
                labels_indexed = labels.set_index(label_index)
            else:
                label_index = labels.index
                labels_indexed = labels
        elif isinstance(labels, pd.Series):
            label_index = labels.index
            labels_indexed = labels
        else:
            label_index = pd.Index([])
            labels_indexed = labels

        if isinstance(signal, pd.Series):
            signal_series = signal.reindex(label_index).fillna(0.0)
        else:
            signal_series = pd.Series(signal, index=label_index).fillna(0.0)

        if self.strategy_evaluator is not None:
            evaluation = self.strategy_evaluator.evaluate_signals(signal_series, labels_indexed)
            dsr = float(evaluation.dsr)
            score = dsr
            if not np.isnan(score):
                score -= self.config.turnover_weight * float(evaluation.turnover)
                score += self.config.breadth_weight * float(evaluation.breadth)
                score -= float(evaluation.pbo)
            meta: dict[str, float] = {
                "dsr": dsr,
                "pbo": float(evaluation.pbo),
                "turnover": float(evaluation.turnover),
                "breadth": float(evaluation.breadth),
            }
            meta.update({k: float(v) for k, v in (evaluation.metrics or {}).items()})
            meta["fitness_score"] = float(score)
            return (float(score) if not np.isnan(score) else -np.inf, meta)
        if isinstance(labels_indexed, pd.DataFrame):
            returns_series = labels_indexed.get("realized_return")
            if returns_series is None:
                returns_series = labels_indexed.get("gross_return")
            if returns_series is None:
                returns_series = labels_indexed.get("y", pd.Series(0.0, index=labels_indexed.index))
            target = returns_series
        else:
            target = labels_indexed

        aligned_signal, aligned_target = signal_series.align(target, join="inner")
        aligned_signal = aligned_signal.fillna(0.0)
        aligned_target = aligned_target.fillna(0.0)
        if aligned_signal.std(ddof=0) == 0:
            return -np.inf, {
                "dsr": float("nan"),
                "sharpe": float("nan"),
                "mean_return": float("nan"),
                "std_return": float(0.0),
                "pbo": float("nan"),
                "turnover": float(signal_series.diff().abs().mean()),
                "breadth": float(((aligned_signal * aligned_target) > 0).mean()),
                "fitness_score": float("nan"),
            }
        pnl = aligned_signal * aligned_target
        mean = pnl.mean()
        std = pnl.std(ddof=0)
        sharpe = mean / std if std > 0 else -np.inf
        fallback_meta = {
            "dsr": float(sharpe),
            "sharpe": float(sharpe),
            "mean_return": float(mean),
            "std_return": float(std),
            "pbo": float("nan"),
            "turnover": float(signal_series.diff().abs().mean()),
            "breadth": float((pnl > 0).mean()),
        }
        fallback_meta["fitness_score"] = float(sharpe)
        return sharpe, fallback_meta

    def _next_generation(self, population: List[CandidateRule]) -> List[CandidateRule]:
        population.sort(key=lambda rule: rule.fitness or -np.inf, reverse=True)
        elites = population[: max(1, int(len(population) * self.config.elite_fraction))]
        new_population = elites.copy()
        while len(new_population) < self.config.population_size:
            if random.random() < self.config.crossover_rate:
                parent_a = self._tournament_selection(population)
                parent_b = self._tournament_selection(population)
                child_expr = self._crossover(parent_a.expression, parent_b.expression)
            else:
                parent = self._tournament_selection(population)
                child_expr = parent.expression
            if random.random() < self.config.mutation_rate:
                child_expr = self._mutate(child_expr)
            new_population.append(CandidateRule(expression=child_expr))
        return new_population[: self.config.population_size]

    def evolve_population(
        self,
        features: pd.DataFrame,
        labels: pd.Series | pd.DataFrame,
    ) -> tuple[List[CandidateRule], CandidateRule | None]:
        """Run GP evolution and return the final evaluated population and best rule."""

        population = self.seed_population()
        evaluated = self.evaluate(population, features, labels)
        best_rule: CandidateRule | None = None
        if evaluated:
            best_rule = max(evaluated, key=lambda rule: rule.fitness or -np.inf)

        for _ in range(1, self.config.n_generations):
            population = self._next_generation(evaluated)
            evaluated = self.evaluate(population, features, labels)
            if evaluated:
                candidate = max(evaluated, key=lambda rule: rule.fitness or -np.inf)
                if best_rule is None or (candidate.fitness or -np.inf) > (best_rule.fitness or -np.inf):
                    best_rule = candidate

        # Re-evaluate the best rule to ensure metadata is aligned with the final generation.
        if best_rule is not None:
            best_rule = self.evaluate([CandidateRule(expression=best_rule.expression)], features, labels)[0]

        return evaluated, best_rule

    def _tournament_selection(self, population: List[CandidateRule], k: int = 3) -> CandidateRule:
        participants = random.sample(population, k)
        return max(participants, key=lambda rule: rule.fitness or -np.inf)

    def _random_expression(self, depth: int) -> str:
        if depth == 0 or (depth > 1 and random.random() < 0.3):
            terminal = random.choice(self.config.feature_columns + tuple(["const"]))
            if terminal == "const":
                return f"{random.uniform(-1, 1):.4f}"
            return terminal
        op = random.choice(list(SAFE_OPERATORS.keys()) + ["div", "tanh", "AND", "OR"] + list(COMPARATOR_NAMES))
        if op in SAFE_OPERATORS:
            left = self._random_expression(depth - 1)
            right = self._random_expression(depth - 1)
            return f"({left} {op} {right})"
        if op == "div":
            left = self._random_expression(depth - 1)
            right = self._random_expression(depth - 1)
            return f"div({left}, {right})"
        if op == "tanh":
            arg = self._random_expression(depth - 1)
            return f"tanh({arg})"
        if op in ("AND", "OR"):
            left = self._random_expression(depth - 1)
            right = self._random_expression(depth - 1)
            return f"{op}({left}, {right})"
        if op in COMPARATOR_NAMES:
            left = self._random_expression(depth - 1)
            right = self._random_expression(depth - 1)
            return f"{op}({left}, {right})"
        raise ValueError("Unsupported operator")

    def _crossover(self, expr_a: str, expr_b: str) -> str:
        tree_a = _parse_expression_tree(expr_a)
        tree_b = _parse_expression_tree(expr_b)
        if tree_a is None or tree_b is None:
            return self._random_expression(self.config.max_depth)

        candidates_a = _collect_candidate_subtrees(tree_a, self.config.feature_columns)
        candidates_b = _collect_candidate_subtrees(tree_b, self.config.feature_columns)
        if not candidates_a or not candidates_b:
            return self._random_expression(self.config.max_depth)

        subtree_a = random.choice(candidates_a)
        subtree_b = random.choice(candidates_b)

        new_tree = _replace_subtree(tree_a, subtree_a, subtree_b)
        try:
            return _safe_unparse(new_tree)
        except Exception:  # pragma: no cover - fallback safety
            return self._random_expression(self.config.max_depth)

    def _mutate(self, expression: str) -> str:
        tree = _parse_expression_tree(expression)
        if tree is None:
            return self._random_expression(self.config.max_depth)

        candidates = _collect_candidate_subtrees(tree, self.config.feature_columns)
        if not candidates:
            return self._random_expression(self.config.max_depth)

        target = random.choice(candidates)
        replacement_expr = self._random_expression(min(2, self.config.max_depth))
        replacement_tree = _parse_expression_tree(replacement_expr)
        if replacement_tree is None:
            return self._random_expression(self.config.max_depth)

        new_tree = _replace_subtree(tree, target, replacement_tree.body)
        try:
            return _safe_unparse(new_tree)
        except Exception:  # pragma: no cover - fallback safety
            return self._random_expression(self.config.max_depth)

    def _evaluate_expression(self, expression: str, features: pd.DataFrame) -> pd.Series:
        local_env = {col: features[col].to_numpy() for col in self.config.feature_columns if col in features}
        local_env.update(SAFE_FUNCTIONS)
        local_env.update({name: func for name, func in SAFE_OPERATORS.items()})
        local_env["div"] = safe_div
        result = eval(expression, {"__builtins__": {}}, local_env)
        if np.isscalar(result):
            result = np.full(len(features), float(result), dtype=float)
        array = np.asarray(result)
        if array.dtype == bool:
            array = array.astype(float)
        array = np.where(np.isnan(array), 0.0, array.astype(float))
        std = np.nanstd(array)
        scale = std if np.isfinite(std) and std > 1e-6 else 1.0
        normalized = array / scale
        bounded = np.tanh(normalized)
        return pd.Series(bounded, index=features.index)
