import pandas as pd

from candlestrats.motifs import mine_motifs_from_bars, motifs_to_frame


def test_mine_motifs_from_bars_basic():
    bars = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="h"),
            "open": [1, 1.1, 1.2, 1.15, 1.3, 1.25, 1.2, 1.22, 1.3, 1.28],
            "close": [1.05, 1.15, 1.25, 1.1, 1.35, 1.2, 1.22, 1.3, 1.27, 1.26],
            "high": [1.1, 1.2, 1.3, 1.2, 1.4, 1.28, 1.25, 1.32, 1.35, 1.31],
            "low": [0.95, 1.05, 1.1, 1.05, 1.2, 1.18, 1.15, 1.18, 1.22, 1.2],
            "volume": [100, 120, 150, 130, 200, 180, 170, 160, 190, 175],
        }
    )
    labels = pd.Series([1, 1, -1, 1, 1, -1, 0, 1, 1, -1], index=bars["timestamp"])
    motifs = mine_motifs_from_bars(bars, labels, min_support=0.2, min_lift=0.5, max_size=2)
    df = motifs_to_frame(motifs)
    assert not df.empty
    assert {"name", "support", "lift", "direction"}.issubset(df.columns)


def test_mine_motifs_with_yaml(tmp_path):
    yaml_path = tmp_path / "predicate_specs.yaml"
    yaml_path.write_text("""default:
  - name: LONG_LOWER_WICK
""")
    bars = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=5, freq="h"),
        "open": [1,1,1,1,1],
        "close": [1.1,1.2,1.05,1.15,1.2],
        "high": [1.2,1.25,1.1,1.2,1.25],
        "low": [0.9,0.95,0.92,0.93,0.94],
        "volume": [100,120,110,130,140],
    })
    labels = pd.Series([1,1,-1,1,1], index=bars["timestamp"])
    motifs = mine_motifs_from_bars(bars, labels, min_support=0.2, min_lift=0.5, max_size=1, spec_config=yaml_path)
    assert isinstance(motifs, list)
