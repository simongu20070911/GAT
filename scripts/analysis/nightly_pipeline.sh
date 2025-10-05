#!/usr/bin/env bash
set -euo pipefail
nightly_dir=/mnt/timemachine/binance/features/nightly_motifs
mkdir -p "$nightly_dir"
bash scripts/nightly_motif_commands.sh
PYTHONPATH=src python - <<'PY'
from pathlib import Path
import pandas as pd

root = Path('/mnt/timemachine/binance/features/nightly_motifs')
frames = []
for cluster_dir in root.iterdir():
    for csv_path in cluster_dir.glob('*.csv'):
        try:
            df = pd.read_csv(csv_path)
        except pd.errors.EmptyDataError:
            continue
        if df.empty:
            continue
        df['cluster'] = cluster_dir.name
        frames.append(df)
combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=['name','items','support','lift','purity','cluster'])
output = Path('reports/stage_02_priors/motifs/motifs_latest.csv')
output.parent.mkdir(parents=True, exist_ok=True)
combined.to_csv(output, index=False)
PY
