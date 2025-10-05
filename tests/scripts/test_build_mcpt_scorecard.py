import pandas as pd

from candlestrats.evaluation import FeeConfig, load_fee_model_from_config
from candlestrats.evaluation.reporting import MCPTSummaryInput, mcpt_summary
from scripts.analysis.build_mcpt_scorecard import derive_symbol


def test_derive_symbol():
    assert derive_symbol(__import__('pathlib').Path('BTCUSDT_returns.parquet')) == 'BTCUSDT'
