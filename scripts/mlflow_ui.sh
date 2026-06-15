#!/usr/bin/env bash
# Abre a UI do MLflow com o mesmo backend SQLite usado por run_educational_ml_suite().
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-sqlite:///${ROOT}/mlruns/mlflow.db}"
mkdir -p mlruns

if [[ -x "${ROOT}/.venv/bin/mlflow" ]]; then
  MLFLOW_BIN="${ROOT}/.venv/bin/mlflow"
elif command -v mlflow >/dev/null 2>&1; then
  MLFLOW_BIN="mlflow"
else
  echo "mlflow não encontrado. Ative o venv ou rode: pip install -r requirements.txt"
  exit 1
fi

echo "MLflow UI → ${MLFLOW_TRACKING_URI}"
echo "Acesse http://localhost:5000"
exec "${MLFLOW_BIN}" ui --backend-store-uri "${MLFLOW_TRACKING_URI}" --host 0.0.0.0 --port 5000
