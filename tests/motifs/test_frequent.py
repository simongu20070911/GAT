import pandas as pd

from candlestrats.motifs.frequent import FrequentMotifMiner


def test_frequent_motif_miner_basic():
    preds = pd.DataFrame(
        {
            "A": [1, 1, 0, 1, 0],
            "B": [0, 1, 0, 1, 0],
            "C": [1, 0, 1, 0, 1],
        }
    )
    labels = pd.Series([1, 1, -1, 1, -1])
    miner = FrequentMotifMiner(min_support=0.2, min_lift=1.0)
    motifs = miner.mine(preds, labels, max_size=2)
    assert motifs
    assert any("A" in motif.items for motif in motifs)
