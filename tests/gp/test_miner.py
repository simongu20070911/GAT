import pandas as pd

from candlestrats.gp import GeneticProgramConfig, GeneticProgramMiner


def test_gp_miner_evolves_rule():
    index = pd.date_range("2024-01-01", periods=4, freq="H", tz="UTC")
    features = pd.DataFrame({
        "feat_a": [0.1, -0.2, 0.3, -0.4],
        "feat_b": [0.5, 0.4, -0.1, -0.2],
    }, index=index)
    labels = pd.Series([1, -1, 1, -1], index=index)
    config = GeneticProgramConfig(population_size=8, n_generations=2, feature_columns=("feat_a", "feat_b"))
    miner = GeneticProgramMiner(config)
    best = miner.evolve(features, labels)
    assert best.expression
    assert best.metadata is None or isinstance(best.metadata, dict)
    if best.metadata is not None:
        assert "dsr" in best.metadata
    assert best.fitness is not None


from candlestrats.evaluation import CombinatorialPurgedCV, FeeModel
from candlestrats.gp.evaluation import StrategyEvaluator


def test_gp_with_strategy_evaluator():
    index = pd.date_range("2024-01-01", periods=4, freq="H", tz="UTC")
    features = pd.DataFrame({
        "feat_a": [0.1, -0.2, 0.3, -0.4],
        "feat_b": [0.5, 0.4, -0.1, -0.2],
    }, index=index)
    labels = pd.Series([1, -1, 1, -1], index=index)
    config = GeneticProgramConfig(population_size=6, n_generations=1, feature_columns=("feat_a", "feat_b"))
    cv = CombinatorialPurgedCV(n_splits=2, embargo_minutes=1)
    evaluator = StrategyEvaluator(cv=cv, fee_model=FeeModel(), min_trades=1)
    miner = GeneticProgramMiner(config, strategy_evaluator=evaluator)
    best = miner.evolve(features, labels)
    assert best.expression
    assert best.metadata is None or isinstance(best.metadata, dict)
    if best.metadata is not None:
        assert "dsr" in best.metadata


def test_gp_signal_bounded_and_comparator(features=None):
    index = pd.date_range("2024-01-01", periods=4, freq="H", tz="UTC")
    feature_frame = pd.DataFrame({
        "feat_a": [0.1, -0.2, 0.3, -0.4],
        "feat_b": [0.5, 0.4, -0.1, -0.2],
    }, index=index)
    config = GeneticProgramConfig(population_size=4, n_generations=1, feature_columns=("feat_a", "feat_b"))
    miner = GeneticProgramMiner(config)
    series = miner._evaluate_expression("GT(feat_a, 0.0)", feature_frame)
    assert ((series.values <= 1.0) & (series.values >= -1.0)).all()
