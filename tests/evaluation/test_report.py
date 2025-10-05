import pandas as pd

from candlestrats.evaluation.reporting import MCPTSummaryInput, mcpt_summary


def test_mcpt_summary_produces_dataframe():
    index = pd.date_range('2024-01-01', periods=60, freq='min')
    returns = pd.Series(0.001, index=index)
    df = mcpt_summary(MCPTSummaryInput(returns=returns, symbol='TEST', runs=5, permutation_block=12))
    assert not df.empty
    assert set(['symbol', 'real_sharpe', 'random_p', 'perm_p']).issubset(df.columns)
