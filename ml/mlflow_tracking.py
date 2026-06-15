"""
Integração MLflow para a suite educacional de evasão escolar.

Usa backend **SQLite** em ``mlruns/mlflow.db`` (compatível com MLflow 3.x sem
``MLFLOW_ALLOW_FILE_STORE``). Artefatos ficam em ``mlruns/``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MLRUNS_DIR = ROOT / "mlruns"
MLFLOW_DB_PATH = MLRUNS_DIR / "mlflow.db"
EXPERIMENT_NAME = "evasao_escolar_escola_ano"


def default_tracking_uri() -> str:
    """
    URI padrão: SQLite local (recomendado no MLflow 3+).

    Pode ser sobrescrita por ``MLFLOW_TRACKING_URI``.
    """
    env = os.environ.get("MLFLOW_TRACKING_URI")
    if env:
        return env
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{MLFLOW_DB_PATH.resolve()}"


def mlflow_ui_command() -> str:
    """Comando documentado para abrir a UI com o mesmo backend da suite."""
    return f'mlflow ui --backend-store-uri "{default_tracking_uri()}"'


def mlflow_enabled() -> bool:
    """Permite desligar MLflow via variável de ambiente (ex.: testes)."""
    return os.environ.get("MLFLOW_DISABLED", "").lower() not in {"1", "true", "yes"}


def _import_mlflow() -> Any:
    """
    Importa o pacote PyPI ``mlflow``, evitando conflito com pastas locais
    chamadas ``mlflow/`` na raiz do projeto (namespace package sem __init__.py).
    """
    import importlib
    import sys

    root = str(ROOT.resolve())
    blocked = {root, str((ROOT / "mlflow").resolve())}
    saved_path = list(sys.path)
    try:
        sys.path = [p for p in sys.path if p not in blocked]
        cached = sys.modules.get("mlflow")
        if cached is not None and not hasattr(cached, "set_tracking_uri"):
            del sys.modules["mlflow"]
        return importlib.import_module("mlflow")
    finally:
        sys.path = saved_path


def configure_mlflow(
    tracking_uri: Path | str | None = None,
    experiment_name: str = EXPERIMENT_NAME,
) -> Any:
    """
    Configura URI local e experimento. Retorna o módulo ``mlflow`` ou ``None`` se desabilitado.
    """
    if not mlflow_enabled():
        return None
    mlflow = _import_mlflow()

    uri = str(tracking_uri) if tracking_uri else default_tracking_uri()
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)
    return mlflow


def _flatten_params(params: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, val in params.items():
        name = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if isinstance(val, dict):
            flat.update(_flatten_params(val, name))
        else:
            flat[name] = val
    return flat


def log_educational_ml_suite_to_mlflow(
    *,
    suite_params: dict[str, Any],
    metrics_by_model: dict[str, dict[str, float]],
    final_test_metrics: dict[str, float],
    final_best_params: dict[str, Any],
    final_best_cv_mae: float,
    cv_summary: dict[str, float],
    fitted_pipelines: dict[str, Any],
    final_model_allfit: Any,
    X_train: pd.DataFrame,
    y_train: Any,
    figure_paths: list[str],
    artifact_json_path: Path | str | None,
    kmeans_k: int,
    tracking_uri: Path | str | None = None,
) -> dict[str, Any] | None:
    """
    Registra múltiplos runs aninhados:
      - run pai da suite (parâmetros globais)
      - um run por modelo comparado (KNN, árvore, HGB baseline)
      - run do modelo final ajustado (métricas de teste + CV + model salvo)
    """
    mlflow = configure_mlflow(tracking_uri=tracking_uri)
    if mlflow is None:
        return None

    from mlflow.models import infer_signature

    uri = mlflow.get_tracking_uri()
    info: dict[str, Any] = {
        "experiment_name": EXPERIMENT_NAME,
        "tracking_uri": uri,
        "parent_run_id": None,
        "child_run_ids": {},
        "final_run_id": None,
        "registered_model_name": None,
        "ui_command": mlflow_ui_command(),
    }

    sig_sample = X_train.iloc[: min(5, len(X_train))]
    signature = infer_signature(sig_sample, final_model_allfit.predict(sig_sample))

    with mlflow.start_run(run_name="educational_ml_suite") as parent:
        info["parent_run_id"] = parent.info.run_id
        mlflow.log_params(_flatten_params(suite_params))
        mlflow.set_tags(
            {
                "projeto": "projeto_evasao_escolar",
                "disciplina": "Aprendizado de Maquina + Projeto 6",
                "target": "taxa_abandono_em",
                "granularidade": "escola-ano",
            }
        )

        for model_name, pipe in fitted_pipelines.items():
            m = metrics_by_model.get(model_name, {})
            with mlflow.start_run(run_name=f"compare_{model_name}", nested=True):
                run_id = mlflow.active_run().info.run_id
                info["child_run_ids"][model_name] = run_id
                mlflow.log_params({"modelo": model_name})
                if m:
                    mlflow.log_metrics(
                        {f"test_{k}": float(v) for k, v in m.items() if v is not None}
                    )
                mlflow.sklearn.log_model(
                    sk_model=pipe,
                    artifact_path=f"model_{model_name.lower()}",
                    input_example=sig_sample,
                    signature=signature,
                )

        with mlflow.start_run(run_name="final_hist_gradient_boosting", nested=True):
            info["final_run_id"] = mlflow.active_run().info.run_id
            mlflow.log_params(_flatten_params(final_best_params, prefix="best"))
            mlflow.log_metric("best_cv_mae", float(final_best_cv_mae))
            mlflow.log_metrics(
                {f"test_{k}": float(v) for k, v in final_test_metrics.items()}
            )
            for key, val in cv_summary.items():
                if val is not None and key != "n_folds":
                    try:
                        mlflow.log_metric(f"cv_{key}", float(val))
                    except (TypeError, ValueError):
                        pass
            if cv_summary.get("n_folds") is not None:
                mlflow.log_param("cv_n_folds", int(cv_summary["n_folds"]))
            mlflow.log_param("kmeans_k", int(kmeans_k))

            mlflow.sklearn.log_model(
                sk_model=final_model_allfit,
                artifact_path="model_final",
                registered_model_name="evasao_abandono_em_final",
                input_example=sig_sample,
                signature=signature,
            )
            info["registered_model_name"] = "evasao_abandono_em_final"

            for fig in figure_paths[:12]:
                p = Path(fig)
                if p.is_file():
                    mlflow.log_artifact(str(p), artifact_path="figures")

            if artifact_json_path and Path(artifact_json_path).is_file():
                mlflow.log_artifact(str(artifact_json_path), artifact_path="storytelling")

    return info
