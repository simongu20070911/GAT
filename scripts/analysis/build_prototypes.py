"""Cluster GP rules into psychology prototypes."""
from __future__ import annotations

import argparse
from pathlib import Path
import glob

import pandas as pd

from candlestrats.cluster.prototypes import cluster_rules, load_rule_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build prototype clusters from GP rules")
    parser.add_argument("rules_glob", type=str, help="Glob pattern for gp_rules_*.json files")
    parser.add_argument("--metric-cols", nargs="*", default=["dsr", "turnover", "breadth"], help="Metric columns to normalize")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV for prototypes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = sorted(glob.glob(args.rules_glob, recursive=True))
    rows = []
    for file in files:
        records = load_rule_records(Path(file))
        prototypes = cluster_rules(records, args.metric_cols)
        for proto in prototypes:
            rows.append(
                {
                    "cluster_id": proto.cluster_id,
                    "rule_name": proto.rule_name,
                    "expression": proto.expression,
                    "metrics": proto.metrics,
                    "source": str(file),
                }
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        metrics_expanded = df["metrics"].apply(pd.Series)
        df = pd.concat([df.drop(columns=["metrics"]), metrics_expanded], axis=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} prototypes to {args.output}")


if __name__ == "__main__":
    main()
