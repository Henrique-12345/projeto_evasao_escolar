#!/bin/bash
set -euo pipefail

cd /app
export PYTHONPATH="/app:${PYTHONPATH:-}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-sqlite:////app/mlruns/mlflow.db}"

run_etl_if_raw_present() {
  if [[ -f "data/raw/dados_educacionais_recife.csv" && -f "data/raw/dados_socioeconomicos_recife.csv" ]]; then
    echo "[entrypoint] Executando ETL..."
    python etl/etl_pipeline.py
  else
    echo "[entrypoint] CSVs brutos ausentes em data/raw/ — pulando ETL."
  fi
}

run_train_if_needed() {
  if [[ ! -f "outputs/ml/final_model_bundle.pkl" ]]; then
    echo "[entrypoint] Bundle ML ausente — executando run_educational_ml_suite()..."
    python -c "from ml.educational_ml import run_educational_ml_suite; run_educational_ml_suite()"
  fi
}

case "${1:-dashboard}" in
  dashboard)
    run_etl_if_raw_present
    run_train_if_needed
    exec streamlit run dashboard/app.py \
      --server.address=0.0.0.0 \
      --server.port=8501 \
      --browser.gatherUsageStats=false
    ;;
  train)
    run_etl_if_raw_present
    exec python -c "from ml.educational_ml import run_educational_ml_suite; run_educational_ml_suite()"
    ;;
  etl)
    run_etl_if_raw_present
    ;;
  mlflow-ui)
    exec mlflow ui \
      --host 0.0.0.0 \
      --port 5000 \
      --backend-store-uri "${MLFLOW_TRACKING_URI}"
    ;;
  bash)
    exec bash
    ;;
  *)
    exec "$@"
    ;;
esac
