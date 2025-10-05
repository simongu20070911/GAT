"""End-to-end pipeline orchestration from minute bars to features and labels."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from candlestrats.bars import DecisionBarConfig, build_decision_bars
from candlestrats.data import MinuteBarStore
from candlestrats.dsl import evaluate_pattern
from candlestrats.features import (
    compute_morphology_features,
    compute_morphology_features_on_bars,
    discover_motifs,
)
from candlestrats.features.motifs import MotifSpec
from candlestrats.labeling import TripleBarrierConfig, apply_triple_barrier
from candlestrats.gp.grammar import PSYCHOLOGY_SEEDS


@dataclass
class PipelineConfig:
    symbol: str
    bar: DecisionBarConfig = field(default_factory=DecisionBarConfig)
    triple_barrier: TripleBarrierConfig = field(default_factory=TripleBarrierConfig)
    motif: Optional[MotifSpec] = None
    start: Optional[pd.Timestamp] = None
    end: Optional[pd.Timestamp] = None


@dataclass
class PipelineResult:
    minute_bars: pd.DataFrame
    decision_bars: pd.DataFrame
    mapping: pd.DataFrame
    labels: pd.DataFrame
    features: pd.DataFrame
    motifs: Optional[pd.DataFrame]


def _aggregate_minute_features_to_bars(
    minute_features: pd.DataFrame,
    mapping: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate minute-level features onto decision bar ids using shape-aware reducers."""

    base_cols = [col for col in minute_features.columns if col != "timestamp"]
    if minute_features.empty or mapping.empty:
        empty = pd.DataFrame(columns=base_cols)
        empty.index.name = "bar_id"
        return empty

    feature_frame = minute_features.merge(mapping, on="timestamp", how="inner")
    if feature_frame.empty:
        empty = pd.DataFrame(columns=base_cols)
        empty.index.name = "bar_id"
        return empty

    feature_cols = [col for col in feature_frame.columns if col not in {"timestamp", "bar_id"}]

    agg_spec: dict[str, list[str]] = {}

    for col in feature_cols:
        funcs: list[str] = []
        if col.endswith("nr_flag"):
            funcs.append("sum")
        if any(token in col for token in ["upper", "lower", "wick", "body", "range_var", "vol_z", "ret_z", "drawup", "drawdown"]):
            funcs.append("max")
        if "ret_sum" in col:
            funcs.append("sum")
        if col.endswith("direction") or col.endswith("cum_return"):
            funcs.append("last")
        if not funcs:
            funcs.append("mean")
        agg_spec[col] = list(dict.fromkeys(funcs))

    grouped = feature_frame.groupby("bar_id").agg(agg_spec)
    grouped.columns = [f"{col}_{func}" for col, func in grouped.columns]
    grouped = grouped.sort_index()
    grouped.index = grouped.index.astype(int)
    grouped.index.name = "bar_id"
    return grouped


def run_pipeline(store: MinuteBarStore, config: PipelineConfig) -> PipelineResult:
    """Run the research pipeline for a single symbol."""
    minute_bars = store.load(config.symbol, start=config.start, end=config.end)
    decision = build_decision_bars(minute_bars, config.bar)
    minute_features = compute_morphology_features(minute_bars)
    labels = apply_triple_barrier(decision.bars, config.triple_barrier)
    labels["ts"] = pd.to_datetime(labels["ts"], utc=True)
    labels["gross_return"] = labels["realized_return"]
    bar_ts = (
        decision.bars.set_index("bar_id")["ts"].astype("datetime64[ns, UTC]")
        if not decision.bars.empty
        else pd.Series(dtype="datetime64[ns, UTC]")
    )
    features = _aggregate_minute_features_to_bars(minute_features, decision.mapping)
    if not features.empty:
        features["ts"] = pd.to_datetime(features.index.to_series().map(bar_ts), utc=True)

        # Join decision-bar morphology features
        bar_geom = compute_morphology_features_on_bars(decision.bars)
        if not bar_geom.empty:
            bar_geom["ts"] = pd.to_datetime(bar_geom["ts"], utc=True)
            bar_geom = bar_geom.set_index("ts")
            features = features.join(bar_geom, on="ts", how="left")

        if PSYCHOLOGY_SEEDS:
            decision_bars = decision.bars.copy()
            decision_bars["ts"] = pd.to_datetime(decision_bars["ts"], utc=True)
            decision_frame = decision_bars.set_index("ts").sort_index()
            for seed in PSYCHOLOGY_SEEDS:
                column_name = f"seed_{seed.name}"
                try:
                    mask = evaluate_pattern(seed.expression(), decision_frame)
                    mask = mask.reindex(decision_frame.index).fillna(False)
                    bar_series = pd.Series(mask.astype(float).values, index=decision_bars["bar_id"].values)
                    features[column_name] = features.index.to_series().map(bar_series).fillna(0.0)
                except Exception:  # pragma: no cover - defensive guard
                    features[column_name] = 0.0
    motifs = None
    if config.motif:
        motif_features = minute_features.copy()
        motifs = discover_motifs(motif_features, config.motif)

    return PipelineResult(
        minute_bars=minute_bars,
        decision_bars=decision.bars,
        mapping=decision.mapping,
        labels=labels,
        features=features,
        motifs=motifs,
    )
