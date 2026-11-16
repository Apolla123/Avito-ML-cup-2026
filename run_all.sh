#!/usr/bin/env bash
# Полный прогон: признаки -> CatBoost (3 фолда) -> NN (3 фолда) -> ансамбль -> submission.parquet
# Использование: DATA=/path/to/competition CACHE=cache RUNS=runs bash run_all.sh
set -euo pipefail
DATA=${DATA:?укажите DATA=/path/to/competition}
CACHE=${CACHE:-cache}
RUNS=${RUNS:-runs}
NFOLDS=${NFOLDS:-3}
CB_DEVICE=${CB_DEVICE:-gpu}     # gpu | cpu
NN_DEVICE=${NN_DEVICE:-cuda}    # cuda | cpu
WORKERS=${WORKERS:-8}

python -m avito_rank features --data "$DATA" --cache "$CACHE" --workers "$WORKERS"

for k in $(seq 0 $((NFOLDS-1))); do
  python -m avito_rank train catboost --cache "$CACHE" --out "$RUNS/cb" --fold "$k" --nfolds "$NFOLDS" --device "$CB_DEVICE"
  python -m avito_rank train nn       --cache "$CACHE" --out "$RUNS/nn" --fold "$k" --nfolds "$NFOLDS" --device "$NN_DEVICE"
done

python -m avito_rank blend "$RUNS/cb" "$RUNS/nn" --sample "$DATA/sample_submission.parquet" --out "$RUNS/submission.parquet"
