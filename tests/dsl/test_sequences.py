import pandas as pd

from candlestrats.dsl.sequences import QuantifierOperator, SequenceOperator


def test_sequence_operator_tolerance():
    s1 = pd.Series([True, True, False, True])
    s2 = pd.Series([True, False, False, True])
    op = SequenceOperator(window=2, tolerance=1)
    result = op.evaluate([s1, s2])
    assert bool(result.iloc[-1])


def test_quantifier_operator():
    mask = pd.Series([True, False, True, True])
    quantifier = QuantifierOperator(window=3, threshold=2)
    result = quantifier.evaluate(mask)
    assert bool(result.iloc[-1])
