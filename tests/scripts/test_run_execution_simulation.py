import pandas as pd

from scripts.run_execution_simulation import read_orders


def test_read_orders_csv(tmp_path):
    path = tmp_path / 'orders.csv'
    pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=2, freq='h'),
        'side': ['buy', 'sell'],
        'price': [100, 101],
        'quantity': [1, -1],
    }).to_csv(path, index=False)
    df = read_orders(path)
    assert len(df) == 2
