"""Configuration helpers for global paths and defaults."""
from __future__ import annotations

from pathlib import Path

DATA_ROOT = Path("/mnt/timemachine/binance")


def ensure_data_root() -> Path:
    """Return the canonical data root and verify it exists."""
    if not DATA_ROOT.exists():  # pragma: no cover - depends on host env
        raise FileNotFoundError(
            f"Expected historical data under {DATA_ROOT}; adjust config if layout shifts."
        )
    return DATA_ROOT
