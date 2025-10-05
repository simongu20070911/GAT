"""End-to-end batch runner for the candlestick research pipeline."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from glob import glob
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from candlestrats.cluster.prototypes import cluster_rules, load_rule_records

DEFAULT_OUTPUT_ROOT = Path("/mnt/timemachine/binance/features/full_run")
DEFAULT_LOG_DIR = Path("/mnt/timemachine/binance/features/full_run/logs")


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def run_command(cmd: List[str], log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as log:
        log.write(f"\n[{timestamp()}] Running: {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            raise RuntimeError(f"Command {' '.join(cmd)} failed with code {proc.returncode}")
        log.write(f"[{timestamp()}] Completed.\n")


def aggregate_motifs(output_root: Path, destination: Path) -> int:
    frames = []
    for motif_path in output_root.glob("*/**/motifs_*.csv"):
        try:
            df = pd.read_csv(motif_path)
        except pd.errors.EmptyDataError:
            continue
        if df.empty:
            continue
        parts = motif_path.parts
        symbol = parts[-2]
        month = parts[-1].split("_")[3].replace(".csv", "")
        df["symbol"] = symbol
        df["month"] = month
        frames.append(df)
    if not frames:
        destination.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_csv(destination, index=False)
        return 0
    combined = pd.concat(frames, ignore_index=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(destination, index=False)
    return len(combined)


def aggregate_gp(output_root: Path, destination: Path) -> int:
    frames = []
    for gp_path in output_root.glob("*/**/gp_matrix_*.csv"):
        try:
            df = pd.read_csv(gp_path)
        except pd.errors.EmptyDataError:
            continue
        if df.empty:
            continue
        parts = gp_path.parts
        symbol = parts[-2]
        month = parts[-1].split("_")[3].replace(".csv", "")
        df["symbol"] = symbol
        df["month"] = month
        frames.append(df)
    if not frames:
        destination.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_csv(destination, index=False)
        return 0
    combined = pd.concat(frames, ignore_index=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(destination, index=False)
    return len(combined)


def aggregate_synthetic_checks(output_root: Path, destination: Path) -> dict[str, int | float]:
    records = []
    for check_path in output_root.glob("*/**/synthetic_check_*.json"):
        try:
            text_data = check_path.read_text()
        try:
            payload = json.loads(text_data, parse_constant=lambda _k: None)
        except json.JSONDecodeError:
            text_data = text_data.replace("Infinity", "null").replace("-Infinity", "null").replace("NaN", "null")
            payload = json.loads(text_data)
        except json.JSONDecodeError:
            continue
        payload["path"] = str(check_path)
        records.append(payload)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        destination.write_text(json.dumps([], indent=2))
        return {
            "count": 0,
            "initial_flagged": 0,
            "final_flagged": 0,
            "post_guard_passed": 0,
            "post_guard_failed": 0,
            "post_guard_pending": 0,
            "max_positive_fitness": 0.0,
            "max_overall_fitness": 0.0,
            "post_guard_max_positive_fitness": 0.0,
        }

    df = pd.DataFrame(records)
    destination.write_text(df.to_json(orient="records", indent=2))

    def _series_or_none(name: str) -> pd.Series | None:
        return df.get(name)

    def _count_true(series: pd.Series | None) -> int:
        if series is None:
            return 0
        return int(series.fillna(False).astype(bool).sum())

    flagged_initial_series = _series_or_none("flagged_initial")
    if flagged_initial_series is None:
        flagged_initial_series = _series_or_none("flagged")
    flagged_initial = _count_true(flagged_initial_series)
    flagged_final = _count_true(_series_or_none("flagged"))

    post_guard_series = _series_or_none("post_guard_passed")
    post_guard_passed = _count_true(post_guard_series)
    if post_guard_series is not None:
        post_guard_pending = int(post_guard_series.isna().sum())
        post_guard_failed = int(post_guard_series.fillna(False).eq(False).sum()) - post_guard_pending
    else:
        post_guard_failed = 0
        post_guard_pending = 0

    pos_series = _series_or_none("synthetic_max_positive_fitness")
    overall_series = _series_or_none("synthetic_max_overall_fitness")
    post_guard_positive = _series_or_none("post_guard_max_positive_fitness")

    def _max_or_zero(series: pd.Series | None) -> float:
        if series is None or series.empty:
            return 0.0
        return float(series.fillna(0.0).max())

    return {
        "count": int(len(df)),
        "initial_flagged": flagged_initial,
        "final_flagged": flagged_final,
        "post_guard_passed": post_guard_passed,
        "post_guard_failed": post_guard_failed,
        "post_guard_pending": post_guard_pending,
        "max_positive_fitness": _max_or_zero(pos_series),
        "max_overall_fitness": _max_or_zero(overall_series),
        "post_guard_max_positive_fitness": _max_or_zero(post_guard_positive),
    }


def build_prototypes(output_root: Path, destination: Path, metric_cols: Iterable[str]) -> int:
    rows = []
    for json_path_str in glob(str(output_root / "*" / "*" / "gp_rules_*.json")):
        json_path = Path(json_path_str)
        records = load_rule_records(json_path)
        prototypes = cluster_rules(records, metric_cols)
        for proto in prototypes:
            row = {
                "cluster_id": proto.cluster_id,
                "rule_name": proto.rule_name,
                "expression": proto.expression,
                "source": str(json_path),
            }
            row.update(proto.metrics)
            rows.append(row)
    df = pd.DataFrame(rows)
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination, index=False)
    return len(df)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full pipeline batches and aggregate outputs")
    parser.add_argument("--months", nargs="+", required=True, help="Months to process (YYYY-MM)")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--force", action="store_true", help="Reprocess even if _SUCCESS exists")
    parser.add_argument("--aggregate-only", action="store_true", help="Skip processing, just aggregate results")
    parser.add_argument("--metric-cols", nargs="*", default=["dsr", "turnover", "breadth"], help="Prototype metric columns")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_dir = args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    if not args.aggregate_only:
        for month in args.months:
            cmd = [
                "python",
                "scripts/full_run_pipeline.py",
                "--months",
                month,
                "--batch-size",
                str(args.batch_size),
                "--output-root",
                str(args.output_root),
            ]
            if args.force:
                cmd.append("--force")
            run_command(cmd, log_dir / f"pipeline_{month}.log")

    motifs_written = aggregate_motifs(args.output_root, args.output_root / "motifs_latest.csv")
    gp_written = aggregate_gp(args.output_root, args.output_root / "gp_matrix_latest.csv")
    prototypes_written = build_prototypes(args.output_root, args.output_root / "prototypes_latest.csv", args.metric_cols)
    synthetic_meta = aggregate_synthetic_checks(args.output_root, args.output_root / "synthetic_checks_latest.json")

    summary = {
        "output_root": str(args.output_root),
        "months": args.months,
        "motifs_rows": motifs_written,
        "gp_rows": gp_written,
        "prototypes_rows": prototypes_written,
        "synthetic_checks": synthetic_meta,
        "timestamp": timestamp(),
    }
    summary_path = args.output_root / "summary_latest.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
