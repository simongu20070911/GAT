import pandas as pd

from candlestrats.gp.miner import GeneticProgramConfig, GeneticProgramMiner
from candlestrats.dsl import PatternExpression, evaluate_pattern


def test_gp_with_dsl_context():
    features = pd.DataFrame({
        "feat_a": [0.1, 0.2, -0.1, -0.3],
        "feat_b": [0.4, 0.3, 0.2, -0.2],
    })
    labels = pd.Series([1, 1, -1, -1])
    config = GeneticProgramConfig(population_size=6, n_generations=1, feature_columns=("feat_a", "feat_b"))
    miner = GeneticProgramMiner(config)
    best = miner.evolve(features, labels)
    assert best.expression
