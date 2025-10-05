# Candlestick Strategy Research Pipeline

This repository scaffolds the mid-frequency candlestick research initiative described in `AGENTS.md`. It wires up the core packages, scripts, and test harnesses required for agents to start implementing ingestion, feature engineering, evolutionary mining, and evaluation.

## Key Components
- `src/candlestrats/`: Python package containing data access, bar construction, labeling, feature, modelling, evaluation, and portfolio modules.
- `scripts/`: Thin orchestration entrypoints for each stage of the research workflow.
- `tests/`: Pytest suites aligned with each module family to keep regression coverage tight.
- `docs/`: Living design notes, module ownership, and runbooks.

Follow the sprint sequencing from `AGENTS.md` to populate each stub with full logic. All filesystem access to historical market data must read from `/mnt/timemachine/binance` per the operations guide.

## Validation
- Run `pytest` for unit coverage.
- Guard against leakage with `python scripts/run_monte_carlo.py <returns.parquet>`; review `docs/LEAKAGE_GUARDRAILS.md` for the full workflow.

- Use `PYTHONPATH=src python scripts/run_research_pipeline.py BTCUSDT --start 2024-01-01T00:00:00Z --end 2024-01-01T12:00:00Z --output /mnt/timemachine/binance/features/pipeline_demo` to materialise decision bars, labels, and features.
- Summarise leakage checks with `PYTHONPATH=src python scripts/analysis/aggregate_mcpt.py <returns.parquet> --symbol BTCUSDT`.

- Nightly leakage summaries: `PYTHONPATH=src python scripts/analysis/build_mcpt_scorecard.py reports/stage_02_priors/returns --runs 200 --block-size 12`.
- Execution check: `PYTHONPATH=src python scripts/run_execution_simulation.py orders.parquet --fee-tier vip0`.

- Fee overrides: edit `config/research/strategy_fees.yaml` to map symbols to tiers; scripts default to its `default_tier` unless `--fee-tier` overrides them.

- Nightly cron-friendly target: `make mcpt-scorecard` (reads `reports/stage_02_priors/returns`).

- Candlestick DSL + GP: modules under `candlestrats/dsl/`, `candlestrats/motifs/`, and `candlestrats/gp/grammar.py` seed psychology patterns for the evolutionary miner.

- Motif mining: `PYTHONPATH=src python scripts/run_motif_mining.py <bars.parquet> <labels.parquet> --output reports/stage_02_priors/motifs.csv` generates frequent predicate combinations for GP seeds.

- Configure motif predicates via `config/research/predicate_specs.yaml`; `scripts/run_motif_mining.py` automatically loads overrides with `--spec-config`.
