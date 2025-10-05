from candlestrats.features import FeatureRegistry


def test_feature_registry_hashing():
    registry = FeatureRegistry()
    registry.register("feat_a", {"window": 60, "fn": "body_atr"})
    record = registry.get("feat_a")
    assert record is not None
    assert len(record.graph_hash) == 64
