from candlestrats.evaluation.fees import (
    FeeConfig,
    StrategyFeeOverrides,
    load_fee_model_from_config,
    resolve_fee_model,
    resolve_fee_tier,
    FEES_CONFIG_PATH,
)


def test_load_fee_model_from_config(tmp_path):
    yaml_text = (
        "venues:\n"
        "  binance:\n"
        "    futures_usdm:\n"
        "      vip0:\n"
        "        maker_bps: 1.0\n"
        "        taker_bps: 4.0\n"
        "        half_spread_bps: 1.5\n"
        "        impact_perc_coeff: 0.4\n"
    )
    config_path = tmp_path / "fees.yaml"
    config_path.write_text(yaml_text)
    fee_model = load_fee_model_from_config(
        FeeConfig("binance", "futures_usdm", "vip0", config_path=config_path)
    )
    assert fee_model.taker_bps == 4.0
    assert fee_model.half_spread_bps == 1.5


def test_resolve_fee_model_with_overrides(tmp_path):
    fees_yaml = (
        "venues:\n"
        "  binance:\n"
        "    futures_usdm:\n"
        "      vip0:\n"
        "        maker_bps: 2\n"
        "        taker_bps: 5\n"
        "      vip1:\n"
        "        maker_bps: 1.5\n"
        "        taker_bps: 4\n"
    )
    overrides_yaml = (
        "default_tier: vip0\n"
        "overrides:\n"
        "  BTCUSDT: vip1\n"
    )
    fees_path = tmp_path / "fees.yaml"
    fees_path.write_text(fees_yaml)
    strat_path = tmp_path / "strategy_fees.yaml"
    strat_path.write_text(overrides_yaml)

    model, tier = resolve_fee_model(
        "binance",
        "futures_usdm",
        symbol="BTCUSDT",
        fees_config=fees_path,
        strategy_config=strat_path,
    )
    assert tier == "vip1"
    assert model.taker_bps == 4

    fallback = resolve_fee_tier("ETHUSDT", overrides=StrategyFeeOverrides(path=strat_path))
    assert fallback == "vip0"


def test_default_fee_config_available():
    model = load_fee_model_from_config(
        FeeConfig("binance", "futures_usdm", "vip0", config_path=FEES_CONFIG_PATH)
    )
    assert model.taker_bps == 4.0
