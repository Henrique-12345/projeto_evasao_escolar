"""
Campanha de experimentos MLflow — três regressores com hiperparâmetros variados.

Cada rodada do plano ``EXPERIMENT_PLAN`` treina **um** algoritmo (HGB, árvore ou KNN),
avalia no holdout temporal e deve ser registrada no MLflow pelo notebook
``mlruns/experimentos_mlflow_parametros.ipynb``.
"""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from sklearn.pipeline import Pipeline

from ml.baseline_municipio import (
    evaluate_regression,
    make_decision_tree_pipeline,
    make_hist_gradient_boosting_pipeline,
    make_knn_pipeline,
    prepare_temporal_supervised_split,
)
from ml.mlflow_tracking import configure_mlflow, default_tracking_uri, mlflow_ui_command

AlgorithmName = Literal[
    "HistGradientBoosting",
    "DecisionTree",
    "KNeighbors",
]

# Experimento principal da campanha manual (notebook mlruns/)
MLFLOW_TUNING_EXPERIMENT = "evasao_treino_parametros"

# Mantido por compatibilidade com execuções anteriores
HGB_TUNING_EXPERIMENT = MLFLOW_TUNING_EXPERIMENT

# ---------------------------------------------------------------------------
# Plano de 30 rodadas: 10 por algoritmo (HGB, árvore, KNN)
# Cada entrada declara explicitamente o algoritmo e os hiperparâmetros da rodada.
# ---------------------------------------------------------------------------
EXPERIMENT_PLAN: list[dict[str, Any]] = [
    # --- HistGradientBoosting (10 rodadas) ---
    {
        "run_name": "hgb_01_baseline",
        "algorithm": "HistGradientBoosting",
        "params": {
            "learning_rate": 0.15,
            "max_depth": 3,
            "max_iter": 80,
            "min_samples_leaf": 24,
            "l2_regularization": 0.0,
            "max_leaf_nodes": 15,
        },
        "nota": "HGB baseline — poucas iterações, folhas grandes.",
    },
    {
        "run_name": "hgb_02_mais_iter_l2",
        "algorithm": "HistGradientBoosting",
        "params": {
            "learning_rate": 0.10,
            "max_depth": 4,
            "max_iter": 180,
            "min_samples_leaf": 16,
            "l2_regularization": 0.1,
            "max_leaf_nodes": 31,
        },
        "nota": "HGB — mais iterações + regularização L2 leve.",
    },
    {
        "run_name": "hgb_03_arvore_rasa",
        "algorithm": "HistGradientBoosting",
        "params": {
            "learning_rate": 0.10,
            "max_depth": 2,
            "max_iter": 220,
            "min_samples_leaf": 20,
            "l2_regularization": 0.05,
            "max_leaf_nodes": 7,
        },
        "nota": "HGB — boosting com árvores rasas (max_depth=2).",
    },
    {
        "run_name": "hgb_04_balanceado",
        "algorithm": "HistGradientBoosting",
        "params": {
            "learning_rate": 0.10,
            "max_depth": 4,
            "max_iter": 220,
            "min_samples_leaf": 12,
            "l2_regularization": 0.3,
            "max_leaf_nodes": 15,
        },
        "nota": "HGB — combinação balanceada (candidato a modelo final).",
    },
    {
        "run_name": "hgb_05_lr_alto_curto",
        "algorithm": "HistGradientBoosting",
        "params": {
            "learning_rate": 0.20,
            "max_depth": 3,
            "max_iter": 60,
            "min_samples_leaf": 30,
            "l2_regularization": 0.0,
            "max_leaf_nodes": 15,
        },
        "nota": "HGB — learning rate alto, poucas iterações.",
    },
    {
        "run_name": "hgb_06_lr_baixo_longo",
        "algorithm": "HistGradientBoosting",
        "params": {
            "learning_rate": 0.05,
            "max_depth": 3,
            "max_iter": 300,
            "min_samples_leaf": 18,
            "l2_regularization": 0.1,
            "max_leaf_nodes": 31,
        },
        "nota": "HGB — learning rate baixo, treino longo.",
    },
    {
        "run_name": "hgb_07_profundo_moderado",
        "algorithm": "HistGradientBoosting",
        "params": {
            "learning_rate": 0.08,
            "max_depth": 5,
            "max_iter": 200,
            "min_samples_leaf": 14,
            "l2_regularization": 0.15,
            "max_leaf_nodes": 31,
        },
        "nota": "HGB — árvores mais profundas (max_depth=5).",
    },
    {
        "run_name": "hgb_08_folhas_pequenas",
        "algorithm": "HistGradientBoosting",
        "params": {
            "learning_rate": 0.12,
            "max_depth": 3,
            "max_iter": 150,
            "min_samples_leaf": 8,
            "l2_regularization": 0.2,
            "max_leaf_nodes": 63,
        },
        "nota": "HGB — folhas pequenas, mais folhas por árvore.",
    },
    {
        "run_name": "hgb_09_l2_forte",
        "algorithm": "HistGradientBoosting",
        "params": {
            "learning_rate": 0.10,
            "max_depth": 3,
            "max_iter": 250,
            "min_samples_leaf": 22,
            "l2_regularization": 0.5,
            "max_leaf_nodes": 15,
        },
        "nota": "HGB — regularização L2 forte contra overfitting.",
    },
    {
        "run_name": "hgb_10_muito_raso",
        "algorithm": "HistGradientBoosting",
        "params": {
            "learning_rate": 0.08,
            "max_depth": 2,
            "max_iter": 280,
            "min_samples_leaf": 28,
            "l2_regularization": 0.1,
            "max_leaf_nodes": 5,
        },
        "nota": "HGB — variante extra-rasa (vizinha da melhor rodada 03).",
    },
    # --- DecisionTree (10 rodadas) ---
    {
        "run_name": "tree_01_rasa",
        "algorithm": "DecisionTree",
        "params": {"max_depth": 3, "min_samples_leaf": 12},
        "nota": "Árvore rasa — regras simples, menos overfitting.",
    },
    {
        "run_name": "tree_02_media",
        "algorithm": "DecisionTree",
        "params": {"max_depth": 4, "min_samples_leaf": 8},
        "nota": "Árvore profundidade média (padrão do dashboard).",
    },
    {
        "run_name": "tree_03_profunda",
        "algorithm": "DecisionTree",
        "params": {"max_depth": 6, "min_samples_leaf": 4},
        "nota": "Árvore mais profunda — captura interações não lineares.",
    },
    {
        "run_name": "tree_04_folhas_grandes",
        "algorithm": "DecisionTree",
        "params": {"max_depth": 5, "min_samples_leaf": 16},
        "nota": "Árvore com min_samples_leaf alto — mais estável.",
    },
    {
        "run_name": "tree_05_muito_rasa",
        "algorithm": "DecisionTree",
        "params": {"max_depth": 2, "min_samples_leaf": 20},
        "nota": "Árvore muito rasa — máxima interpretabilidade.",
    },
    {
        "run_name": "tree_06_media_folhas_grandes",
        "algorithm": "DecisionTree",
        "params": {"max_depth": 4, "min_samples_leaf": 20},
        "nota": "Árvore média com folhas grandes (regularização).",
    },
    {
        "run_name": "tree_07_profunda_restrita",
        "algorithm": "DecisionTree",
        "params": {"max_depth": 7, "min_samples_leaf": 8},
        "nota": "Árvore profunda com folhas moderadas.",
    },
    {
        "run_name": "tree_08_intermediaria",
        "algorithm": "DecisionTree",
        "params": {"max_depth": 5, "min_samples_leaf": 6},
        "nota": "Árvore intermediária — equilíbrio profundidade/folhas.",
    },
    {
        "run_name": "tree_09_super_rasa",
        "algorithm": "DecisionTree",
        "params": {"max_depth": 2, "min_samples_leaf": 30},
        "nota": "Árvore super rasa — alta estabilidade.",
    },
    {
        "run_name": "tree_10_muito_profunda",
        "algorithm": "DecisionTree",
        "params": {"max_depth": 8, "min_samples_leaf": 5},
        "nota": "Árvore muito profunda — maior capacidade, mais risco de overfit.",
    },
    # --- KNeighbors (10 rodadas) ---
    {
        "run_name": "knn_01_poucos_vizinhos",
        "algorithm": "KNeighbors",
        "params": {"n_neighbors": 3, "weights": "uniform"},
        "nota": "KNN — 3 vizinhos, pesos uniformes.",
    },
    {
        "run_name": "knn_02_padrao",
        "algorithm": "KNeighbors",
        "params": {"n_neighbors": 5, "weights": "distance"},
        "nota": "KNN — 5 vizinhos, pesos por distância (padrão do projeto).",
    },
    {
        "run_name": "knn_03_mais_vizinhos",
        "algorithm": "KNeighbors",
        "params": {"n_neighbors": 7, "weights": "distance"},
        "nota": "KNN — 7 vizinhos para suavizar previsões.",
    },
    {
        "run_name": "knn_04_muitos_vizinhos",
        "algorithm": "KNeighbors",
        "params": {"n_neighbors": 11, "weights": "distance"},
        "nota": "KNN — 11 vizinhos — maior viés, menor variância.",
    },
    {
        "run_name": "knn_05_k4_uniform",
        "algorithm": "KNeighbors",
        "params": {"n_neighbors": 4, "weights": "uniform"},
        "nota": "KNN — 4 vizinhos, pesos uniformes.",
    },
    {
        "run_name": "knn_06_k6_distance",
        "algorithm": "KNeighbors",
        "params": {"n_neighbors": 6, "weights": "distance"},
        "nota": "KNN — 6 vizinhos, pesos por distância.",
    },
    {
        "run_name": "knn_07_k9_distance",
        "algorithm": "KNeighbors",
        "params": {"n_neighbors": 9, "weights": "distance"},
        "nota": "KNN — 9 vizinhos para suavização intermediária.",
    },
    {
        "run_name": "knn_08_k13_distance",
        "algorithm": "KNeighbors",
        "params": {"n_neighbors": 13, "weights": "distance"},
        "nota": "KNN — 13 vizinhos, alta suavização.",
    },
    {
        "run_name": "knn_09_k15_uniform",
        "algorithm": "KNeighbors",
        "params": {"n_neighbors": 15, "weights": "uniform"},
        "nota": "KNN — 15 vizinhos, pesos uniformes.",
    },
    {
        "run_name": "knn_10_k5_uniform",
        "algorithm": "KNeighbors",
        "params": {"n_neighbors": 5, "weights": "uniform"},
        "nota": "KNN — 5 vizinhos uniformes (contraste com knn_02 distance).",
    },
]

# Alias legado (só HGB) — preferir EXPERIMENT_PLAN
HGB_EXPERIMENT_GRID = [
    {**row, **row["params"]}
    for row in EXPERIMENT_PLAN
    if row["algorithm"] == "HistGradientBoosting"
]


def algorithm_display_name(algorithm: str) -> str:
    return {
        "HistGradientBoosting": "HistGradientBoostingRegressor",
        "DecisionTree": "DecisionTreeRegressor",
        "KNeighbors": "KNeighborsRegressor",
    }.get(algorithm, algorithm)


def build_pipeline_from_config(
    algorithm: AlgorithmName | str,
    numeric_features: list[str],
    categorical_features: list[str],
    params: dict[str, Any],
    *,
    random_state: int = 42,
) -> Pipeline:
    """Monta o pipeline sklearn conforme algoritmo e hiperparâmetros da rodada."""
    if algorithm == "HistGradientBoosting":
        return make_hist_gradient_boosting_pipeline(
            numeric_features,
            categorical_features,
            random_state=random_state,
            **params,
        )
    if algorithm == "DecisionTree":
        return make_decision_tree_pipeline(
            numeric_features,
            categorical_features,
            random_state=random_state,
            **params,
        )
    if algorithm == "KNeighbors":
        return make_knn_pipeline(
            numeric_features,
            categorical_features,
            **params,
        )
    raise ValueError(f"Algoritmo não suportado: {algorithm}")


def prepare_split(year_cutoff: int = 2017) -> dict[str, Any]:
    """Atalho documentado para o notebook."""
    return prepare_temporal_supervised_split(year_cutoff=year_cutoff)


def log_training_run_to_mlflow(
    mlflow: Any,
    *,
    run_name: str,
    algorithm: str,
    params: dict[str, Any],
    pipe: Pipeline,
    metrics: dict[str, float],
    rodada: int,
    year_cutoff: int,
    random_state: int,
    sig_sample: pd.DataFrame,
    nota: str = "",
) -> str:
    """
    Registra um training run no MLflow (parâmetros + métricas + modelo).
    Retorna o run_id.
    """
    from mlflow.models import infer_signature

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(
            {
                "campanha": "treino_parametros_notebook",
                "rodada": str(rodada),
                "algorithm": algorithm,
                "modelo_sklearn": algorithm_display_name(algorithm),
                "nota": nota,
            }
        )
        mlflow.log_params(
            {
                "algorithm": algorithm,
                "year_cutoff": year_cutoff,
                "random_state": random_state,
                **params,
            }
        )
        mlflow.log_metrics(
            {
                "test_mae": float(metrics["mae"]),
                "test_rmse": float(metrics["rmse"]),
                "test_r2": float(metrics["r2"]),
            }
        )
        signature = infer_signature(sig_sample, pipe.predict(sig_sample))
        mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path="model",
            input_example=sig_sample,
            signature=signature,
        )
        return mlflow.active_run().info.run_id


def run_hgb_mlflow_training_campaign(**_kwargs: Any) -> pd.DataFrame:
    """Descontinuado — use o notebook ``mlruns/experimentos_mlflow_parametros.ipynb``."""
    raise RuntimeError(
        "run_hgb_mlflow_training_campaign foi substituído pelo loop explícito no notebook "
        "mlruns/experimentos_mlflow_parametros.ipynb (EXPERIMENT_PLAN + log_training_run_to_mlflow)."
    )


def campaign_info() -> dict[str, str]:
    n_hgb = sum(1 for r in EXPERIMENT_PLAN if r["algorithm"] == "HistGradientBoosting")
    n_tree = sum(1 for r in EXPERIMENT_PLAN if r["algorithm"] == "DecisionTree")
    n_knn = sum(1 for r in EXPERIMENT_PLAN if r["algorithm"] == "KNeighbors")
    return {
        "experiment_name": MLFLOW_TUNING_EXPERIMENT,
        "tracking_uri": default_tracking_uri(),
        "ui_command": mlflow_ui_command(),
        "n_configs": str(len(EXPERIMENT_PLAN)),
        "n_hgb": str(n_hgb),
        "n_decision_tree": str(n_tree),
        "n_knn": str(n_knn),
    }
