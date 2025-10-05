"""Shared pytest fixtures."""
from __future__ import annotations

import sys
from pathlib import Path

PATH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PATH_ROOT / "src"))
sys.path.insert(0, str(PATH_ROOT))



import pytest

from candlestrats.utils import generate_random_walk_ohlcv


@pytest.fixture
def random_ohlcv():
    return generate_random_walk_ohlcv(length=120)
