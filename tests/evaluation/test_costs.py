from candlestrats.evaluation import FeeModel


def test_fee_model_net_edge():
    model = FeeModel(maker_bps=0.5, taker_bps=5.0, half_spread_bps=1.0, impact_perc_coeff=0.2)
    net = model.net_edge(10.0, maker=False, participation=0.5)
    assert net < 10.0
    cost = model.cost_bps(maker=True)
    assert cost == 0.5
