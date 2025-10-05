"""Motif mining pipeline helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from candlestrats.dsl import evaluate_predicates
import yaml
from candlestrats.dsl.predicates import Predicate, registry as predicate_registry
from candlestrats.motifs.frequent import FrequentMotifMiner, Motif


@dataclass
class PredicateSpec:
    name: str
    params: dict[str, float]


@dataclass
class MotifAtom:
    name: str
    items: tuple[str, ...]
    support: float
    lift: float
    purity: float
    direction: int



DEFAULT_SPEC_PATH = Path(__file__).resolve().parents[3] / "config" / "research" / "predicate_specs.yaml"

DEFAULT_SPECS: List[PredicateSpec] = [
    PredicateSpec("LONG_LOWER_WICK", {"kind": "lower", "threshold": 0.3}),
    PredicateSpec("WICK_DOMINANCE", {"kind": "upper", "ratio": 1.5}),
    PredicateSpec("RANGE_COMPRESSION", {"window": 5, "quantile": 0.2}),
    PredicateSpec("VOLUME_SPIKE", {"window": 20, "quantile": 0.8}),
    PredicateSpec("BREAK_ABOVE_RANGE", {"lookback": 10, "eps": 0.0}),
    PredicateSpec("BREAK_BELOW_RANGE", {"lookback": 10, "eps": 0.0}),
    PredicateSpec("INSIDE_BAR", {}),
    PredicateSpec("OUTSIDE_BAR", {}),
    PredicateSpec("BULLISH_ENGULF", {}),
    PredicateSpec("BEARISH_ENGULF", {}),
    PredicateSpec("EVR_ANOMALY", {"window": 20, "threshold": 1.0}),
    PredicateSpec("ROUND_LEVEL_PROXIMITY", {"step": 0.01, "tolerance": 0.25}),
    PredicateSpec("PIVOT_HIGH", {"left": 2, "right": 2}),
    PredicateSpec("PIVOT_LOW", {"left": 2, "right": 2}),
]



def _load_spec_config(path: Path | None) -> List[PredicateSpec]:
    cfg_path = path or DEFAULT_SPEC_PATH
    if cfg_path.exists():
        with cfg_path.open('r', encoding='utf-8') as fh:
            data = yaml.safe_load(fh) or {}
        specs = data.get('default', data)
        predicate_specs = []
        for entry in specs:
            name = entry.get('name') if isinstance(entry, dict) else entry
            params = entry.get('params', {}) if isinstance(entry, dict) else {}
            predicate_specs.append(PredicateSpec(name=name, params=params))
        return predicate_specs
    return DEFAULT_SPECS

def _build_predicates(specs: Iterable[PredicateSpec]) -> List[Predicate]:
    predicates: List[Predicate] = []
    for idx, spec in enumerate(specs):
        evaluator = predicate_registry.get(spec.name)
        predicates.append(
            Predicate(name=f"{spec.name}__{idx}", params=spec.params.copy(), evaluator=evaluator)
        )
    return predicates


def mine_motifs_from_bars(
    bars: pd.DataFrame,
    labels: pd.Series,
    predicate_specs: Iterable[PredicateSpec] | None = None,
    min_support: float = 0.01,
    min_lift: float = 1.1,
    max_size: int = 3,
    spec_config: Path | None = None,
) -> List[MotifAtom]:
    specs = list(predicate_specs) if predicate_specs else _load_spec_config(spec_config)
    predicates = _build_predicates(specs)

    frame = bars.copy()
    if "ts" in frame.columns and "timestamp" not in frame.columns:
        frame = frame.rename(columns={"ts": "timestamp"})

    if "bar_id" in frame.columns:
        frame = frame.set_index("bar_id")
        frame.index.name = "bar_id"
    elif "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = frame.set_index("timestamp")
        frame.index.name = "timestamp"
    else:
        raise KeyError("Bars must include either 'bar_id' or 'ts'/'timestamp'")

    frame = frame.sort_index()

    predicate_table = evaluate_predicates(frame, predicates)
    label_index = pd.to_datetime(labels.index, utc=True, errors="coerce")
    aligned_labels = pd.Series(labels.values, index=label_index)
    aligned_labels = aligned_labels.reindex(predicate_table.index).fillna(0)

    miner = FrequentMotifMiner(min_support=min_support, min_lift=min_lift)
    motifs = miner.mine(predicate_table, aligned_labels, max_size=max_size)
    atoms: List[MotifAtom] = []
    for motif in motifs:
        suffix = "LONG" if motif.direction > 0 else "SHORT"
        atoms.append(
            MotifAtom(
                name=f"{'_'.join(motif.items)}__{suffix}",
                items=motif.items,
                support=motif.support,
                lift=motif.lift,
                purity=motif.purity,
                direction=motif.direction,
            )
        )
    return atoms


def motifs_to_frame(motifs: Iterable[MotifAtom]) -> pd.DataFrame:
    rows = [
        {
            "name": motif.name,
            "items": ",".join(motif.items),
            "support": motif.support,
            "lift": motif.lift,
            "purity": motif.purity,
            "direction": motif.direction,
        }
        for motif in motifs
    ]
    return pd.DataFrame(rows)
