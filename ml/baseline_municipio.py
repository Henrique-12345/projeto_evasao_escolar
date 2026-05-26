"""
Baseline de regressão — granularidade **escola–ano** (`fato_integrado`).

Enunciado metodológico: estimar **taxa_abandono_em** como indicador associado ao risco de
evasão escolar; **`taxa_evasao_em`** permanece como **contexto municipal** (replicado por ano).

Carrega `fato_integrado`, monta pré-processamento reprodutível e avalia
modelos de regressão (HistGradientBoosting, árvore, KNN) com split temporal;
análise de ausentes ainda usa Ridge como referência linear.
Linhas sem valor para o alvo (`taxa_abandono_em`) são excluídas do treino/teste
(regressão supervisionada); covariáveis ausentes são tratadas pelo `SimpleImputer`
no Pipeline (ajuste apenas no treino).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
INTEGRATED_PATH = PROC / "fato_integrado.csv"

TARGET = "taxa_abandono_em"
EXCLUDE_FROM_FEATURES = frozenset({TARGET, "indice_risco_evasao", "id_linha_educacional"})
DEFAULT_YEAR_CUTOFF = 2017


def ensure_processed_data() -> None:
    """Gera CSVs processados via ETL se `fato_integrado.csv` não existir."""
    if INTEGRATED_PATH.exists():
        return
    sys.path.insert(0, str(ROOT))
    from etl.etl_pipeline import run_etl

    run_etl()


def load_fato_integrado() -> pd.DataFrame:
    """Lê a tabela integrada gerada pelo ETL."""
    ensure_processed_data()
    return pd.read_csv(INTEGRATED_PATH)


def infer_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Separa colunas numéricas e categóricas para predição do alvo `TARGET`,
    excluindo vazamento (`indice_risco_evasao`), o identificador de linha e o próprio alvo.
    """
    numeric: list[str] = []
    categorical: list[str] = []
    for col in df.columns:
        if col in EXCLUDE_FROM_FEATURES:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric.append(col)
        else:
            categorical.append(col)
    return numeric, categorical


def build_preprocess_transformer(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    try:
        _num_imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    except TypeError:
        _num_imputer = SimpleImputer(strategy="median")
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", _num_imputer),
            ("scaler", StandardScaler()),
        ]
    )
    # sklearn >= 1.2: sparse_output; versões antigas: sparse
    try:
        _ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        _ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    try:
        _cat_imputer = SimpleImputer(strategy="most_frequent", keep_empty_features=True)
    except TypeError:
        _cat_imputer = SimpleImputer(strategy="most_frequent")
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", _cat_imputer),
            ("onehot", _ohe),
        ]
    )
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_features:
        transformers.append(("num", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_pipeline, categorical_features))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def temporal_split_by_year(
    df: pd.DataFrame,
    year_cutoff: int = DEFAULT_YEAR_CUTOFF,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Treino: ano <= cutoff; validação/teste: ano > cutoff (validação temporal)."""
    train = df[df["ano"] <= year_cutoff].copy()
    test = df[df["ano"] > year_cutoff].copy()
    return train, test


def make_ridge_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    alpha: float = 1.0,
) -> Pipeline:
    """Pipeline completo: pré-processamento + Ridge."""
    prep = build_preprocess_transformer(numeric_features, categorical_features)
    return Pipeline(
        steps=[
            ("prep", prep),
            ("model", Ridge(alpha=alpha)),
        ]
    )


def make_hist_gradient_boosting_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    *,
    max_depth: int | None = 5,
    max_iter: int = 200,
    learning_rate: float = 0.08,
    min_samples_leaf: int = 20,
    l2_regularization: float = 0.0,
    max_leaf_nodes: int = 31,
    random_state: int = 42,
) -> Pipeline:
    """Pré-processamento + HistGradientBoosting (ensemble baseado em árvores)."""
    prep = build_preprocess_transformer(numeric_features, categorical_features)
    return Pipeline(
        steps=[
            ("prep", prep),
            (
                "model",
                HistGradientBoostingRegressor(
                    max_depth=max_depth,
                    max_iter=max_iter,
                    learning_rate=learning_rate,
                    min_samples_leaf=min_samples_leaf,
                    l2_regularization=l2_regularization,
                    max_leaf_nodes=max_leaf_nodes,
                    random_state=random_state,
                    early_stopping=False,
                ),
            ),
        ]
    )


def make_decision_tree_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    *,
    max_depth: int = 4,
    min_samples_leaf: int = 2,
    random_state: int = 42,
) -> Pipeline:
    """Pré-processamento + árvore de decisão (interpretável; profundidade limitada)."""
    prep = build_preprocess_transformer(numeric_features, categorical_features)
    return Pipeline(
        steps=[
            ("prep", prep),
            (
                "model",
                DecisionTreeRegressor(
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    random_state=random_state,
                ),
            ),
        ]
    )


def make_knn_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    *,
    n_neighbors: int = 5,
    weights: str = "distance",
) -> Pipeline:
    """Pré-processamento + KNN (escalas já normalizadas no bloco numérico)."""
    prep = build_preprocess_transformer(numeric_features, categorical_features)
    return Pipeline(
        steps=[
            ("prep", prep),
            ("model", KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)),
        ]
    )


def prepare_temporal_supervised_split(
    year_cutoff: int = DEFAULT_YEAR_CUTOFF,
) -> dict[str, Any]:
    """
    Carrega `fato_integrado`, aplica split temporal e remove linhas sem alvo.
    Retorna matrizes e metadados para treino de regressão supervisionada.
    """
    df = load_fato_integrado()
    if "ano" not in df.columns:
        raise ValueError("Coluna 'ano' obrigatória para split temporal.")

    numeric_features, categorical_features = infer_feature_columns(df)
    train_df, test_df = temporal_split_by_year(df, year_cutoff=year_cutoff)

    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError(
            "Split temporal vazio: ajuste `year_cutoff` ou verifique os anos em `fato_integrado`."
        )

    train_df = train_df.dropna(subset=[TARGET])
    test_df = test_df.dropna(subset=[TARGET])
    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError(
            "Após remover linhas sem alvo (%s), treino ou teste ficou vazio." % TARGET
        )

    sort_cols = [c for c in ["ano", "id_linha_educacional"] if c in train_df.columns]
    if sort_cols:
        train_df = train_df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
        test_df = test_df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    feat_cols = numeric_features + categorical_features
    X_train = train_df[feat_cols]
    y_train = train_df[TARGET].values
    X_test = test_df[feat_cols]
    y_test = test_df[TARGET].values

    return {
        "train_df": train_df,
        "test_df": test_df,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "train_years": sorted(train_df["ano"].unique().tolist()),
        "test_years": sorted(test_df["ano"].unique().tolist()),
    }


def evaluate_regression(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    """MAE, RMSE e R² (métricas acordadas na definição formal do problema)."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"mae": mae, "rmse": rmse, "r2": r2}


def run_baseline_experiment(
    year_cutoff: int = DEFAULT_YEAR_CUTOFF,
    ridge_alpha: float = 1.0,
    hgb_random_state: int = 42,
) -> dict[str, Any]:
    """
    Baseline rápido: **HistGradientBoosting** como modelo principal preditivo
    (mesmo split temporal). Mantém ``ridge_alpha`` apenas por compatibilidade de assinatura;
    o Ridge não é mais o baseline principal.

    Retorna métricas e previsões do HGB para notebooks que só precisam de um ajuste único.
    """
    S = prepare_temporal_supervised_split(year_cutoff=year_cutoff)
    X_train, y_train = S["X_train"], S["y_train"]
    X_test, y_test = S["X_test"], S["y_test"]
    numeric_features = S["numeric_features"]
    categorical_features = S["categorical_features"]
    train_df, test_df = S["train_df"], S["test_df"]

    hgb = make_hist_gradient_boosting_pipeline(
        numeric_features, categorical_features, random_state=hgb_random_state
    )
    hgb.fit(X_train, y_train)
    y_pred_hgb = hgb.predict(X_test)
    metrics_hgb = evaluate_regression(y_test, y_pred_hgb)

    return {
        "target": TARGET,
        "year_cutoff": year_cutoff,
        "train_years": S["train_years"],
        "test_years": S["test_years"],
        "n_train": len(train_df),
        "n_test": len(test_df),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "metrics_hgb": metrics_hgb,
        "metrics_ridge": metrics_hgb,
        "pipeline_hgb": hgb,
        "pipeline_ridge": hgb,
        "y_test": y_test,
        "y_pred_hgb": y_pred_hgb,
        "y_pred_ridge": y_pred_hgb,
    }


def run_model_comparison_experiment(
    year_cutoff: int = DEFAULT_YEAR_CUTOFF,
    ridge_alpha: float = 1.0,
    tree_max_depth: int = 4,
    knn_neighbors: int = 5,
    hgb_max_depth: int | None = 5,
    hgb_max_iter: int = 200,
    hgb_learning_rate: float = 0.08,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Compara **três** regressores no mesmo conjunto de teste temporal:

    - ``HistGradientBoostingRegressor`` — modelo principal preditivo
    - ``DecisionTreeRegressor`` — interpretabilidade / regras
    - ``KNeighborsRegressor`` — vizinhança / escolas semelhantes

    Todas as métricas são MAE, RMSE e R² em ``y_test``.
    O parâmetro ``ridge_alpha`` é ignorado (mantido só por compatibilidade com chamadas antigas).
    Para KMeans e artefatos completos use ``ml.educational_ml.run_educational_ml_suite``.
    """
    S = prepare_temporal_supervised_split(year_cutoff=year_cutoff)
    X_train, y_train = S["X_train"], S["y_train"]
    X_test, y_test = S["X_test"], S["y_test"]
    num_f, cat_f = S["numeric_features"], S["categorical_features"]

    n_tr = len(X_train)
    k_nn = min(max(2, knn_neighbors), max(2, n_tr - 1))

    pipelines: dict[str, Pipeline] = {
        "HistGradientBoosting": make_hist_gradient_boosting_pipeline(
            num_f,
            cat_f,
            max_depth=hgb_max_depth,
            max_iter=hgb_max_iter,
            learning_rate=hgb_learning_rate,
            random_state=random_state,
        ),
        "DecisionTreeRegressor": make_decision_tree_pipeline(
            num_f, cat_f, max_depth=tree_max_depth, random_state=random_state
        ),
        "KNeighborsRegressor": make_knn_pipeline(num_f, cat_f, n_neighbors=k_nn),
    }

    metrics_by_model: dict[str, dict[str, float]] = {}
    predictions_by_model: dict[str, np.ndarray] = {}
    fitted: dict[str, Pipeline] = {}

    for name, est in pipelines.items():
        est.fit(X_train, y_train)
        y_pred = est.predict(X_test)
        predictions_by_model[name] = y_pred
        metrics_by_model[name] = evaluate_regression(y_test, y_pred)
        fitted[name] = est

    return {
        "target": TARGET,
        "year_cutoff": year_cutoff,
        "train_years": S["train_years"],
        "test_years": S["test_years"],
        "n_train": len(S["train_df"]),
        "n_test": len(S["test_df"]),
        "numeric_features": num_f,
        "categorical_features": cat_f,
        "metrics_by_model": metrics_by_model,
        "predictions_by_model": predictions_by_model,
        "pipelines": fitted,
        "y_test": y_test,
    }


def metrics_comparison_dataframe(metrics_by_model: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Transforma o dicionário de métricas em tabela ordenada por MAE."""
    rows = [{"modelo": k, **v} for k, v in metrics_by_model.items()]
    df = pd.DataFrame(rows)
    return df.sort_values("mae").reset_index(drop=True)


def plot_model_comparison_figures(
    y_test: np.ndarray,
    metrics_by_model: dict[str, dict[str, float]],
    predictions_by_model: dict[str, np.ndarray],
    save_dir: Path | None = None,
    dpi: int = 120,
    show: bool = False,
) -> list[Path]:
    """
    Gera figuras para análise comparativa: barras de métricas, observado vs previsto,
    boxplot de erros (resíduos). Se ``save_dir`` for informado, grava PNGs e retorna caminhos.
    Com ``show=True``, exibe as figuras (útil no Jupyter) antes de fechar.

    Requer matplotlib (já em ``requirements.txt``).
    """
    import matplotlib.pyplot as plt

    out_paths: list[Path] = []
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    order = sorted(metrics_by_model.keys(), key=lambda k: metrics_by_model[k]["mae"])
    mae_v = [metrics_by_model[k]["mae"] for k in order]
    rmse_v = [metrics_by_model[k]["rmse"] for k in order]
    r2_v = [metrics_by_model[k]["r2"] for k in order]

    def _finish(fig: Any, fname: str | None) -> None:
        if save_dir is not None and fname:
            p = save_dir / fname
            fig.savefig(p, dpi=dpi, bbox_inches="tight")
            out_paths.append(p)
        if show:
            plt.show()
        plt.close(fig)

    # --- Figura 1: barras MAE / RMSE ---
    fig1, ax1 = plt.subplots(figsize=(9, 4))
    x = np.arange(len(order))
    w = 0.35
    ax1.bar(x - w / 2, mae_v, width=w, label="MAE", color="#475569")
    ax1.bar(x + w / 2, rmse_v, width=w, label="RMSE", color="#94A3B8")
    ax1.set_xticks(x)
    ax1.set_xticklabels(order, rotation=15, ha="right")
    ax1.set_ylabel("Pontos percentuais (alvo: abandono EM)")
    ax1.set_title("Comparação de modelos — MAE e RMSE no conjunto de teste temporal")
    ax1.legend()
    fig1.tight_layout()
    _finish(fig1, "model_comparison_mae_rmse.png")

    # --- Figura 2: R² ---
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    colors = ["#1E3A5F" for _ in order]
    ax2.barh(order[::-1], r2_v[::-1], color=colors[::-1])
    ax2.set_xlabel("R²")
    ax2.set_title("R² no conjunto de teste (maior é melhor)")
    fig2.tight_layout()
    _finish(fig2, "model_comparison_r2.png")

    # --- Figura 3: observado vs previsto (grid 2x2) ---
    plot_models = [m for m in order if m in predictions_by_model]
    n_p = len(plot_models)
    ncols = 2
    nrows = int(np.ceil(n_p / ncols))
    fig3, axes = plt.subplots(nrows, ncols, figsize=(10, 4 * nrows), squeeze=False)
    lim = float(y_test.min()), float(y_test.max())
    for idx, name in enumerate(plot_models):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        pred = predictions_by_model[name]
        ax.scatter(y_test, pred, alpha=0.85, edgecolors="k", s=60)
        ax.plot(lim, lim, "r--", lw=1.2)
        ax.set_xlabel("Observado (%)")
        ax.set_ylabel("Previsto (%)")
        ax.set_title(name)
    for j in range(len(plot_models), nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].set_visible(False)
    fig3.suptitle("Abandono EM — observado × previsto (teste temporal)", y=1.02)
    fig3.tight_layout()
    _finish(fig3, "model_comparison_obs_vs_pred.png")

    # --- Figura 4: boxplot dos erros (previsto - observado) ---
    fig4, ax4 = plt.subplots(figsize=(9, 4))
    errors = [predictions_by_model[k] - y_test for k in order]
    bp_kw: dict[str, Any] = {"patch_artist": True}
    try:
        bp = ax4.boxplot(errors, tick_labels=order, **bp_kw)
    except TypeError:
        bp = ax4.boxplot(errors, labels=order, **bp_kw)
    for patch in bp["boxes"]:
        patch.set_facecolor("#CBD5E1")
    ax4.axhline(0, color="#DC2626", linestyle="--", lw=1)
    ax4.set_ylabel("Erro (previsto − observado), p.p.")
    ax4.set_title("Distribuição dos erros no conjunto de teste")
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=15, ha="right")
    fig4.tight_layout()
    _finish(fig4, "model_comparison_error_boxplot.png")

    return out_paths


def missing_fraction_per_column(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Proporção de valores ausentes por coluna no treino (0–1)."""
    return df[columns].isna().mean()


def rows_with_any_missing(df: pd.DataFrame, columns: list[str]) -> int:
    """Quantidade de linhas com pelo menos um ausente nas colunas informadas."""
    return int(df[columns].isna().any(axis=1).sum())


def run_missing_impact_analysis(
    year_cutoff: int = DEFAULT_YEAR_CUTOFF,
    ridge_alpha: float = 1.0,
    sparse_threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Avalia o impacto dos ausentes no baseline: métricas com todas as features
    (imputação via Pipeline no treino) versus modelo **sem** colunas cuja taxa de
    ausência no **treino** supere ``sparse_threshold``.

    As frações de ausência são calculadas apenas em ``train_df`` para evitar vazamento.
    """
    df = load_fato_integrado()
    if "ano" not in df.columns:
        raise ValueError("Coluna 'ano' obrigatória para split temporal.")

    numeric_features, categorical_features = infer_feature_columns(df)
    all_feat = numeric_features + categorical_features
    train_df, test_df = temporal_split_by_year(df, year_cutoff=year_cutoff)

    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError(
            "Split temporal vazio: ajuste `year_cutoff` ou verifique os anos em `fato_integrado`."
        )

    train_df = train_df.dropna(subset=[TARGET])
    test_df = test_df.dropna(subset=[TARGET])
    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError(
            "Após remover linhas sem alvo (%s), treino ou teste ficou vazio." % TARGET
        )

    frac = missing_fraction_per_column(train_df, all_feat)
    dropped = sorted([c for c in all_feat if frac.get(c, 0) > sparse_threshold])
    kept = [c for c in all_feat if c not in dropped]

    num_kept = [c for c in numeric_features if c in kept]
    cat_kept = [c for c in categorical_features if c in kept]

    X_train = train_df[all_feat]
    y_train = train_df[TARGET].values
    y_test = test_df[TARGET].values
    X_test = test_df[all_feat]

    ridge_full = make_ridge_pipeline(numeric_features, categorical_features, alpha=ridge_alpha)
    ridge_full.fit(X_train, y_train)
    metrics_full = evaluate_regression(y_test, ridge_full.predict(X_test))

    if kept:
        ridge_sparse = make_ridge_pipeline(num_kept, cat_kept, alpha=ridge_alpha)
        ridge_sparse.fit(train_df[kept], y_train)
        metrics_sparse = evaluate_regression(y_test, ridge_sparse.predict(test_df[kept]))
    else:
        metrics_sparse = metrics_full

    burden = frac.sort_values(ascending=False).reset_index()
    burden.columns = ["coluna", "frac_ausentes_treino"]

    return {
        "sparse_threshold": sparse_threshold,
        "columns_dropped_as_sparse": dropped,
        "n_features_full": len(all_feat),
        "n_features_after_drop": len(kept),
        "train_rows_with_any_missing_in_X": rows_with_any_missing(train_df, all_feat),
        "test_rows_with_any_missing_in_X": rows_with_any_missing(test_df, all_feat),
        "metrics_ridge_all_features": metrics_full,
        "metrics_ridge_without_sparse_columns": metrics_sparse,
        "top_missing_columns_train": burden.head(20).to_dict("records"),
    }
