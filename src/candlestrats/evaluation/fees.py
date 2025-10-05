"""Fee configuration loaders."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

from candlestrats.evaluation.costs import FeeModel

logger = logging.getLogger(__name__)

def _default_config_path(*parts: str) -> Path:
    roots = [
        Path(__file__).resolve().parents[3],  # project root (e.g. GAT)
        Path(__file__).resolve().parents[4],  # monorepo root (fallback)
    ]
    for root in roots:
        candidate = root.joinpath("config", *parts)
        if candidate.exists():
            return candidate
    # Fall back to the first root even if the file is missing; callers will raise later.
    return roots[0].joinpath("config", *parts)


FEES_CONFIG_PATH = _default_config_path("fees.yaml")
STRATEGY_FEES_PATH = _default_config_path("research", "strategy_fees.yaml")


@dataclass
class FeeConfig:
    venue: str
    market: str
    tier: str
    config_path: Path = FEES_CONFIG_PATH


@dataclass
class StrategyFeeOverrides:
    path: Path = STRATEGY_FEES_PATH


def load_fee_model_from_config(options: FeeConfig) -> FeeModel:
    data = _load_yaml(options.config_path)
    venues = data.get("venues", data)
    venue_data = venues.get(options.venue, {})
    market_data = venue_data.get(options.market, {})
    tier_data = market_data.get(options.tier, {})
    if not tier_data:
        msg = (
            f"No fee data for {options.venue}/{options.market}/{options.tier} "
            f"in {options.config_path}"
        )
        logger.error(msg)
        raise KeyError(msg)

    maker_bps = float(tier_data.get("maker_bps", tier_data.get("maker", 0)))
    taker_bps = float(tier_data.get("taker_bps", tier_data.get("taker", 0)))
    half_spread = float(tier_data.get("half_spread_bps", tier_data.get("half_spread", 0)))
    impact_coeff = float(tier_data.get("impact_perc_coeff", tier_data.get("impact", 0)))
    return FeeModel(
        maker_bps=maker_bps,
        taker_bps=taker_bps,
        half_spread_bps=half_spread,
        impact_perc_coeff=impact_coeff,
    )


def resolve_fee_tier(
    symbol: Optional[str],
    default_tier: Optional[str] = None,
    overrides: StrategyFeeOverrides = StrategyFeeOverrides(),
) -> str:
    mapping: Dict[str, str] = {}
    default = default_tier or "vip0"
    path = overrides.path
    if path and path.exists():
        data = _load_yaml(path)
        default = str(data.get("default_tier", default))
        mapping = {str(k): str(v) for k, v in (data.get("overrides", {}) or {}).items()}
    if symbol and symbol in mapping:
        return mapping[symbol]
    return default


def resolve_fee_model(
    venue: str,
    market: str,
    symbol: Optional[str] = None,
    tier: Optional[str] = None,
    fees_config: Path = FEES_CONFIG_PATH,
    strategy_config: Path = STRATEGY_FEES_PATH,
) -> Tuple[FeeModel, str]:
    strategy_path = strategy_config or STRATEGY_FEES_PATH
    fees_path = fees_config or FEES_CONFIG_PATH
    overrides = StrategyFeeOverrides(path=strategy_path)
    final_tier = tier or resolve_fee_tier(symbol, overrides=overrides)
    model = load_fee_model_from_config(
        FeeConfig(venue, market, final_tier, config_path=fees_path)
    )
    return model, final_tier


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path or not path.exists():
        msg = f"Fee configuration {path} not found"
        logger.error(msg)
        raise FileNotFoundError(msg)
    with path.open("r", encoding="utf-8") as fh:
        content = fh.read()
        try:
            return yaml.safe_load(content) or {}
        except yaml.YAMLError:
            return json.loads(content)
