"""Fee and slippage modelling."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FeeModel:
    maker_bps: float = 1.0
    taker_bps: float = 4.0
    half_spread_bps: float = 1.5
    impact_perc_coeff: float = 0.5  # bps per percent of participation

    def cost_bps(self, *, maker: bool, participation: float = 0.0) -> float:
        """Return total expected cost in basis points."""
        fee = self.maker_bps if maker else self.taker_bps
        spread = 0.0 if maker else self.half_spread_bps * 2
        impact = max(participation, 0.0) * self.impact_perc_coeff
        return fee + spread + impact

    def net_edge(self, gross_bps: float, *, maker: bool, participation: float = 0.0) -> float:
        """Convert gross signal edge (bps) to net after costs."""
        return gross_bps - self.cost_bps(maker=maker, participation=participation)
