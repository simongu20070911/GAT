"""Entry point for validating Binance ingestion specs."""
from __future__ import annotations

import argparse

from candlestrats.data import build_default_store, validate_store


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate ingestion store layout")
    parser.add_argument("symbols", nargs="+", help="Symbols to validate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = build_default_store(args.symbols)
    validate_store(specs)
    print(f"Validated {len(specs)} specs")


if __name__ == "__main__":
    main()
