import pandas as pd

from candlestrats.simulation import ExecutionSimulator


def test_execution_simulator_outputs_slippage():
    orders = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="h"),
            "side": ["buy", "sell", "buy"],
            "price": [100.0, 101.0, 102.0],
            "quantity": [1.0, -1.5, 0.5],
            "maker": [True, False, True],
            "participation": [0.1, 0.5, 0.2],
        }
    )
    simulator = ExecutionSimulator()
    fills = simulator.simulate(orders)
    assert {"fill_price", "fee_paid", "slippage_bps", "fee_bps"}.issubset(fills.columns)
    assert (fills["fee_bps"] > 0).all()
