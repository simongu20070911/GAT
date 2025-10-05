"""Execution simulator."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from candlestrats.evaluation.costs import FeeModel


@dataclass
class ExecutionSimulator:
    queue_penalty_bps: float = 1.0
    impact_coeff: float = 0.1
    fee_model: FeeModel = field(default_factory=FeeModel)

    def simulate(self, orders: pd.DataFrame) -> pd.DataFrame:
        required = {"timestamp", "side", "price", "quantity"}
        if not required.issubset(orders.columns):
            missing = required.difference(orders.columns)
            raise KeyError(f"Orders missing columns: {missing}")
        fills = orders.copy()
        side_sign = fills["side"].map({"buy": 1, "sell": -1}).astype(float)
        queue_penalty = (self.queue_penalty_bps / 10000.0) * side_sign
        abs_quantity = fills["quantity"].abs()
        max_qty = abs_quantity.max()
        if max_qty and max_qty > 0:
            size_fraction = abs_quantity / max_qty
        else:
            size_fraction = pd.Series(0.0, index=fills.index)
        impact = (self.impact_coeff / 10000.0) * size_fraction
        impact = impact.fillna(0.0) * side_sign
        fills["fill_price"] = fills["price"] * (1.0 + queue_penalty + impact)
        maker = fills.get("maker", pd.Series(False, index=fills.index)).astype(bool)
        participation = fills.get("participation", pd.Series(0.0, index=fills.index)).astype(float)
        fee_bps = [
            self.fee_model.cost_bps(maker=bool(m), participation=float(p))
            for m, p in zip(maker, participation)
        ]
        fills["fee_bps"] = fee_bps
        fills["fee_paid"] = fills["quantity"].abs() * fills["fill_price"] * (fills["fee_bps"] / 10000)
        fills["slippage_bps"] = (fills["fill_price"] - fills["price"]) / fills["price"] * 10000
        return fills
