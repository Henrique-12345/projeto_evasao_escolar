# Imagem reprodutível — Aprendizado de Máquina (CESAR School)
# Dashboard Streamlit + pipeline ML + MLflow (tracking local em mlruns/)

FROM python:3.11-slim

LABEL maintainer="projeto_evasao_escolar"
LABEL description="Evasao escolar Recife — ETL, ML, MLflow e dashboard Streamlit"

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    MLFLOW_TRACKING_URI=sqlite:////app/mlruns/mlflow.db \
    MPLBACKEND=Agg

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x docker/entrypoint.sh \
    && mkdir -p data/raw data/processed outputs/ml outputs/figures mlruns

EXPOSE 8501 5000

ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["dashboard"]
