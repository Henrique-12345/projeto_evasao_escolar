"""
Suite de Machine Learning orientada a decisão educacional (escola–ano).

Papéis por algoritmo:
  • HistGradientBoostingRegressor — predição principal e ranking de risco
  • DecisionTreeRegressor — regras interpretáveis e thresholds
  • KNeighborsRegressor — escolas semelhantes / benchmarking
  • KMeans — segmentação não supervisionada de perfis

Alvo supervisionado: ``taxa_abandono_em`` (regressão). KMeans atua apenas em covariáveis
transformadas (sem usar o alvo nos centróides), com ``StandardScaler`` + one-hot já no pré-processador.
"""

from __future__ import annotations

import json
import os
import warnings

# Sem interface gráfica: grava só PNG e evita avisos Qt/Wayland ao importar matplotlib mais tarde.
os.environ.setdefault("MPLBACKEND", "Agg")
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.inspection import permutation_importance
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.tree import export_text, plot_tree

from ml.baseline_municipio import (
    TARGET,
    build_preprocess_transformer,
    evaluate_regression,
    make_decision_tree_pipeline,
    make_hist_gradient_boosting_pipeline,
    make_knn_pipeline,
    metrics_comparison_dataframe,
    plot_model_comparison_figures,
    prepare_temporal_supervised_split,
)

ROOT = Path(__file__).resolve().parent.parent
ML_OUT = ROOT / "outputs" / "ml"
FIG_OUT = ROOT / "outputs" / "figures"

# Variáveis educacionais para radar / perfis (devem existir em fato_integrado)
RADAR_NUM_COLS = [
    "tdi_em",
    "taxa_reprovacao_em",
    "taxa_abandono_em",
    "taxa_repetencia_em",
    "atu_em",
]


def _ensure_dirs() -> None:
    ML_OUT.mkdir(parents=True, exist_ok=True)
    FIG_OUT.mkdir(parents=True, exist_ok=True)


def choose_kmeans_k(
    X: np.ndarray,
    k_min: int = 2,
    k_max: int = 8,
) -> tuple[int, dict[str, list]]:
    """Elbow (inércia) + silhueta no treino; escolhe k com maior silhueta (tie-break menor k)."""
    n = X.shape[0]
    k_max = min(k_max, n - 1, 8)
    if k_max < k_min:
        k_max = k_min
    inertias: list[float] = []
    silhouettes: list[float] = []
    ks: list[int] = []
    for k in range(k_min, k_max + 1):
        if k >= n:
            break
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertias.append(float(km.inertia_))
        ks.append(k)
        if k >= 2 and n > k:
            try:
                sil = float(silhouette_score(X, km.labels_))
            except ValueError:
                sil = float("nan")
            silhouettes.append(sil)
        else:
            silhouettes.append(float("nan"))
    if not ks:
        return 2, {"k": [], "inertia": [], "silhouette": []}
    if not silhouettes or np.all(np.isnan(silhouettes)):
        best_k = ks[min(len(ks) - 1, 1)]
    else:
        best_idx = int(np.nanargmax(silhouettes))
        best_k = ks[best_idx]
    return best_k, {"k": ks, "inertia": inertias, "silhouette": silhouettes}


def regression_comparison_narrative(metrics_by_model: dict[str, dict[str, float]]) -> str:
    """Texto curto em linguagem acessível para relatório / dashboard."""
    rows = metrics_comparison_dataframe(metrics_by_model)
    best = rows.iloc[0]["modelo"]
    mae_best = rows.iloc[0]["mae"]
    interpret = rows[rows["modelo"].str.contains("DecisionTree", case=False)]
    tree_mae = float(interpret["mae"].iloc[0]) if len(interpret) else None
    lines = [
        "Comparamos três modelos de regressão no mesmo período de teste (anos mais recentes). "
        "Todos estimam a taxa de abandono no Ensino Médio (%) por escola e ano.",
        f"O menor erro médio absoluto (MAE) ficou com **{best}** (~{mae_best:.2f} pontos percentuais). "
        "Quanto menor o MAE, mais perto as previsões ficam dos valores observados.",
        "**HistGradientBoosting** costuma captar relações não lineares (combinações de indicadores) "
        "e tende a ter melhor desempenho preditivo quando há padrão nos dados.",
        "**Árvore de decisão** privilegia clareza: gera regras do tipo “se indicador X passa de um limite, "
        "o abandono tende a subir”. O erro costuma ser um pouco maior, mas a leitura para gestores é mais direta.",
        "**K vizinhos (KNN)** prevê olhando para escolas “parecidas” no passado recente — útil para comparar "
        "comunidades escolares semelhantes, não só para o número de MAE.",
    ]
    if tree_mae is not None:
        lines.append(
            f"Em números: a árvore teve MAE ~{tree_mae:.2f} p.p.; "
            "use-a para explicar fatores de risco e o boosting para priorização numérica."
        )
    return "\n\n".join(lines)


def tree_interpretation_summary(
    tree_pipeline: Pipeline,
    feature_names: np.ndarray,
    max_depth_display: int = 3,
) -> str:
    """Regras em texto (trecho) + narrativa automática."""
    tree = tree_pipeline.named_steps["model"]
    report = export_text(
        tree,
        feature_names=list(feature_names),
        max_depth=max_depth_display,
    )
    snippet = "\n".join(report.splitlines()[:40])
    if len(report.splitlines()) > 40:
        snippet += "\n... (árvore truncada na visualização; ver PNG completo)"
    n_leaves = tree.get_n_leaves()
    depth = tree.get_depth()
    intro = (
        f"A árvore de regressão foi limitada em profundidade para leitura humana "
        f"(profundidade máxima observada: {depth}, folhas: {n_leaves}). "
        "Cada ramo indica uma condição sobre indicadores escolares (após imputação e escala no pipeline). "
        "Folhas com valor médio de abandono mais alto representam combinações associadas a **maior risco**.\n\n"
        "**Trecho das regras (limite de profundidade na impressão):**\n\n"
    )
    return intro + "```\n" + snippet + "\n```"


def kmeans_cluster_profiles(
    df_rows: pd.DataFrame,
    labels: np.ndarray,
    value_cols: list[str],
) -> tuple[pd.DataFrame, str]:
    """Médias por cluster + texto descritivo."""
    d = df_rows.copy()
    d["_cluster"] = labels
    present = [c for c in value_cols if c in d.columns]
    if not present:
        return pd.DataFrame(), "Colunas numéricas insuficientes para perfil de clusters."
    prof = d.groupby("_cluster")[present].mean().round(2)
    prof.index.name = "cluster"
    lines = []
    for cl, row in prof.iterrows():
        hi = row.nlargest(3)
        lo = row.nsmallest(2)
        lines.append(
            f"**Grupo {int(cl) + 1}** — indicadores mais altos: {', '.join(f'{k} ({v:.1f})' for k, v in hi.items())}; "
            f"entre os mais baixos: {', '.join(f'{k} ({v:.1f})' for k, v in lo.items())}."
        )
    crit = prof.get(TARGET)
    extra = ""
    if crit is not None and len(crit):
        worst = int(crit.idxmax())
        best = int(crit.idxmin())
        extra = (
            f" O grupo com **maior abandono médio** histórico neste recorte é o grupo {worst + 1}; "
            f"o menor abandono médio aparece no grupo {best + 1}."
        )
    return prof.reset_index(), "\n\n".join(lines) + extra


def knn_similar_rows(
    knn_pipe: Pipeline,
    X_train: pd.DataFrame,
    X_query: pd.DataFrame,
    meta_train: pd.DataFrame,
    *,
    n_neighbors: int = 5,
) -> pd.DataFrame:
    """Para cada linha em X_query, devolve vizinhos do treino com distância."""
    prep = knn_pipe.named_steps["prep"]
    knn = knn_pipe.named_steps["model"]
    Xt = prep.transform(X_train)
    Xq = prep.transform(X_query)
    dist, ind = knn.kneighbors(Xq, n_neighbors=min(n_neighbors, len(X_train)))
    rows_out = []
    for qi in range(Xq.shape[0]):
        for rank, (d, ti) in enumerate(zip(dist[qi], ind[qi])):
            m = meta_train.iloc[int(ti)]
            rows_out.append(
                {
                    "query_row": qi,
                    "vizinho_rank": rank + 1,
                    "distancia": float(d),
                    "ano_vizinho": m.get("ano"),
                    "id_linha_educacional": m.get("id_linha_educacional"),
                }
            )
    return pd.DataFrame(rows_out)


def plot_hgb_risk_and_importance(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    importances: pd.Series,
    save_dir: Path | None,
    show: bool = False,
) -> list[Path]:
    import matplotlib.pyplot as plt

    paths: list[Path] = []
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    def _fin(fig: Any, name: str) -> None:
        if save_dir is not None:
            p = save_dir / name
            fig.savefig(p, dpi=120, bbox_inches="tight")
            paths.append(p)
        if show:
            plt.show()
        plt.close(fig)

    fig1, ax = plt.subplots(figsize=(6, 5))
    lo, hi = float(min(y_test.min(), y_pred.min())), float(max(y_test.max(), y_pred.max()))
    ax.scatter(y_test, y_pred, alpha=0.85, edgecolors="k", s=55)
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.2)
    ax.set_xlabel("Abandono EM observado (%)")
    ax.set_ylabel("Abandono EM previsto — modelo principal (%)")
    ax.set_title("Modelo principal (HistGradientBoosting): observado × previsto")
    fig1.tight_layout()
    _fin(fig1, "ml_hgb_obs_vs_pred.png")

    order = importances.sort_values(ascending=True).tail(15)
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    order.plot(kind="barh", ax=ax2, color="#1E3A5F")
    ax2.set_title("Importância por permutação (top 15)")
    ax2.set_xlabel("Importância por permutação (impacto no desempenho ao embaralhar a variável)")
    fig2.tight_layout()
    _fin(fig2, "ml_hgb_permutation_importance.png")

    fig3, ax3 = plt.subplots(figsize=(7, 4))
    ax3.bar(np.arange(len(y_pred)), y_pred[np.argsort(-y_pred)], color="#B45309", alpha=0.85)
    ax3.set_xlabel("Posição no ranking (1 = maior abandono previsto)")
    ax3.set_ylabel("Abandono EM previsto (%)")
    ax3.set_title("Ranking de risco no teste (ordenado por abandono previsto)")
    fig3.tight_layout()
    _fin(fig3, "ml_hgb_risk_ranking.png")

    return paths


def plot_decision_tree_simple(tree_pipe: Pipeline, feature_names: np.ndarray, save_dir: Path | None, show: bool) -> list[Path]:
    import matplotlib.pyplot as plt

    paths: list[Path] = []
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(16, 9))
    plot_tree(
        tree_pipe.named_steps["model"],
        feature_names=list(feature_names),
        filled=True,
        rounded=True,
        fontsize=7,
        max_depth=4,
        ax=ax,
    )
    ax.set_title("Árvore de decisão (simplificada até profundidade 4)")
    fig.tight_layout()
    if save_dir is not None:
        p = save_dir / "ml_decision_tree.png"
        fig.savefig(p, dpi=140, bbox_inches="tight")
        paths.append(p)
    if show:
        plt.show()
    plt.close(fig)
    return paths


def plot_kmeans_figures(
    X_2d: np.ndarray,
    labels: np.ndarray,
    ks: list[int],
    inertias: list[float],
    silhouettes: list[float],
    save_dir: Path | None,
    show: bool,
) -> list[Path]:
    import matplotlib.pyplot as plt

    paths: list[Path] = []
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    def _fin(fig: Any, name: str) -> None:
        if save_dir is not None:
            p = save_dir / name
            fig.savefig(p, dpi=120, bbox_inches="tight")
            paths.append(p)
        if show:
            plt.show()
        plt.close(fig)

    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(ks, inertias, "o-", color="#1E3A5F")
    ax1.set_xlabel("Número de grupos (k)")
    ax1.set_ylabel("Inércia (soma das distâncias ao centro)")
    ax1.set_title("Método do cotovelo — KMeans")
    fig1.tight_layout()
    _fin(fig1, "ml_kmeans_elbow.png")

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(ks, silhouettes, "s-", color="#15803D")
    ax2.set_xlabel("k")
    ax2.set_ylabel("Silhueta média")
    ax2.set_title("Qualidade da separação dos grupos (silhueta)")
    fig2.tight_layout()
    _fin(fig2, "ml_kmeans_silhouette.png")

    fig3, ax3 = plt.subplots(figsize=(7, 5))
    sc = ax3.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10", alpha=0.85, edgecolors="k", s=45)
    ax3.set_xlabel("Componente 1 (PCA)")
    ax3.set_ylabel("Componente 2 (PCA)")
    ax3.set_title("Escolas–ano no plano PCA, cor = grupo KMeans")
    fig3.colorbar(sc, ax=ax3, label="cluster")
    fig3.tight_layout()
    _fin(fig3, "ml_kmeans_scatter_pca.png")

    return paths


def plot_knn_radar_sample(
    df_train: pd.DataFrame,
    neighbor_idx: list[int],
    anchor_idx: int,
    cols: list[str],
    save_dir: Path | None,
    show: bool,
) -> list[Path]:
    """Radar comparando média dos vizinhos vs linha âncora (valores brutos nas colunas)."""
    import matplotlib.pyplot as plt

    paths: list[Path] = []
    use = [c for c in cols if c in df_train.columns]
    if len(use) < 3:
        return paths
    anchor = df_train.iloc[anchor_idx][use].astype(float).values
    neigh = df_train.iloc[neighbor_idx][use].astype(float).mean(axis=0).values
    vals = np.concatenate([anchor, anchor[:1]])
    vals_n = np.concatenate([neigh, neigh[:1]])
    angles = np.linspace(0, 2 * np.pi, len(use), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, vals, "o-", label="Escola âncora (treino)", color="#1E3A5F")
    ax.fill(angles, vals, alpha=0.15, color="#1E3A5F")
    ax.plot(angles, vals_n, "s--", label="Média dos vizinhos mais próximos", color="#DC2626")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(use, size=8)
    ax.set_title("Perfil comparado (indicadores brutos — ilustrativo)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    fig.tight_layout()
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        p = save_dir / "ml_knn_radar_exemplo.png"
        fig.savefig(p, dpi=120, bbox_inches="tight")
        paths.append(p)
    if show:
        plt.show()
    plt.close(fig)
    return paths


def run_educational_ml_suite(
    year_cutoff: int = 2017,
    *,
    tree_max_depth: int = 4,
    knn_neighbors: int = 5,
    random_state: int = 42,
    save_artifacts: bool = True,
    show_plots: bool = False,
    permutation_repeats: int = 8,
) -> dict[str, Any]:
    """
    Executa treino/avaliação dos três regressores, segmentação KMeans e exporta artefatos.

    Grava em ``outputs/ml/`` (CSV + JSON) e figuras em ``outputs/figures/`` quando
    ``save_artifacts=True``.
    """
    _ensure_dirs()
    S = prepare_temporal_supervised_split(year_cutoff=year_cutoff)
    X_train, y_train = S["X_train"], S["y_train"]
    X_test, y_test = S["X_test"], S["y_test"]
    train_df, test_df = S["train_df"], S["test_df"]
    num_f, cat_f = S["numeric_features"], S["categorical_features"]

    models: dict[str, Pipeline] = {
        "HistGradientBoosting": make_hist_gradient_boosting_pipeline(num_f, cat_f, random_state=random_state),
        "DecisionTreeRegressor": make_decision_tree_pipeline(
            num_f, cat_f, max_depth=tree_max_depth, random_state=random_state
        ),
        "KNeighborsRegressor": make_knn_pipeline(num_f, cat_f, n_neighbors=knn_neighbors),
    }

    metrics_by_model: dict[str, dict[str, float]] = {}
    predictions_by_model: dict[str, np.ndarray] = {}
    fitted: dict[str, Pipeline] = {}

    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        y_p = pipe.predict(X_test)
        predictions_by_model[name] = y_p
        metrics_by_model[name] = evaluate_regression(y_test, y_p)
        fitted[name] = pipe

    hgb = fitted["HistGradientBoosting"]
    dt = fitted["DecisionTreeRegressor"]
    knn = fitted["KNeighborsRegressor"]

    prep_fitted = hgb.named_steps["prep"]
    feat_names = prep_fitted.get_feature_names_out()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        perm = permutation_importance(
            hgb,
            X_test,
            y_test,
            n_repeats=max(3, permutation_repeats),
            random_state=random_state,
            n_jobs=-1,
        )
    # permutation_importance atua nas colunas originais de X (não nas transformadas)
    feat_in = np.asarray(hgb.feature_names_in_, dtype=object)
    imp = pd.Series(perm.importances_mean, index=feat_in).sort_values(ascending=False)

    # --- KMeans (só covariáveis transformadas; ajuste no treino) ---
    prep_km = build_preprocess_transformer(num_f, cat_f)
    Xtr_m = prep_km.fit_transform(X_train, y_train)
    Xte_m = prep_km.transform(X_test)
    best_k, curves = choose_kmeans_k(Xtr_m)
    km = KMeans(n_clusters=best_k, random_state=random_state, n_init=10)
    labels_train = km.fit_predict(Xtr_m)
    labels_test = km.predict(Xte_m)

    df_train_full = train_df.reset_index(drop=True)
    df_test_full = test_df.reset_index(drop=True)
    prof_df, prof_text = kmeans_cluster_profiles(
        pd.concat([df_train_full, df_test_full], ignore_index=True),
        np.concatenate([labels_train, labels_test]),
        [TARGET, "tdi_em", "taxa_reprovacao_em", "taxa_repetencia_em", "atu_em"],
    )

    # PCA 2D for scatter
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, random_state=random_state)
    X_all = np.vstack([Xtr_m, Xte_m])
    labels_all = np.concatenate([labels_train, labels_test])
    X2 = pca.fit_transform(X_all)

    # Previsões no treino (in-sample) para enriquecer export
    pred_hgb_tr = hgb.predict(X_train)
    pred_dt_tr = dt.predict(X_train)
    pred_knn_tr = knn.predict(X_train)

    pred_hgb_te = predictions_by_model["HistGradientBoosting"]
    pred_dt_te = predictions_by_model["DecisionTreeRegressor"]
    pred_knn_te = predictions_by_model["KNeighborsRegressor"]

    rank_test = pd.Series(pred_hgb_te).rank(ascending=False, method="dense").astype(int)

    export_train = train_df[[c for c in ("id_linha_educacional", "ano", TARGET) if c in train_df.columns]].copy()
    export_train["split"] = "treino"
    export_train["pred_hgb"] = pred_hgb_tr
    export_train["pred_decision_tree"] = pred_dt_tr
    export_train["pred_knn"] = pred_knn_tr
    export_train["cluster_kmeans"] = labels_train

    export_test = test_df[[c for c in ("id_linha_educacional", "ano", TARGET) if c in test_df.columns]].copy()
    export_test["split"] = "teste"
    export_test["pred_hgb"] = pred_hgb_te
    export_test["pred_decision_tree"] = pred_dt_te
    export_test["pred_knn"] = pred_knn_te
    export_test["cluster_kmeans"] = labels_test
    export_test["rank_risco_abandono_previsto"] = rank_test.values

    export_all = pd.concat([export_train, export_test], ignore_index=True)

    tree_text = tree_interpretation_summary(dt, feat_names, max_depth_display=3)
    narrative_cmp = regression_comparison_narrative(metrics_by_model)

    similar_test0 = knn_similar_rows(
        knn, X_train, X_test.iloc[[0]], train_df.reset_index(drop=True), n_neighbors=min(5, len(X_train))
    )

    fig_paths: list[Path] = []
    if save_artifacts:
        export_all.to_csv(ML_OUT / "escola_ano_ml_enriquecido.csv", index=False)
        similar_test0.to_csv(ML_OUT / "knn_vizinhos_exemplo_primeira_linha_teste.csv", index=False)
        prof_df.to_csv(ML_OUT / "kmeans_perfil_medio_por_cluster.csv", index=False)

        meta = {
            "year_cutoff": year_cutoff,
            "target": TARGET,
            "metrics": metrics_by_model,
            "kmeans_k": best_k,
            "narrativa_comparacao_regressao": narrative_cmp,
            "kmeans_interpretacao": prof_text,
            "arvore_regras_resumo": tree_text,
        }
        (ML_OUT / "ml_storytelling.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        fig_paths.extend(
            plot_hgb_risk_and_importance(y_test, pred_hgb_te, imp, FIG_OUT, show=show_plots)
        )
        fig_paths.extend(plot_decision_tree_simple(dt, feat_names, FIG_OUT, show_plots))
        fig_paths.extend(
            plot_kmeans_figures(
                X2,
                labels_all,
                curves["k"],
                curves["inertia"],
                curves["silhouette"],
                FIG_OUT,
                show_plots,
            )
        )
        # Radar: primeiro índice do treino com vizinhos
        prep_knn = knn.named_steps["prep"]
        knn_m = knn.named_steps["model"]
        Xt0 = prep_knn.transform(X_train.iloc[[0]])
        _, idx0 = knn_m.kneighbors(Xt0, n_neighbors=min(knn_neighbors, len(X_train) - 1))
        neigh_list = [int(i) for i in idx0[0][1:]] if len(idx0[0]) > 1 else [int(idx0[0][0])]
        fig_paths.extend(
            plot_knn_radar_sample(df_train_full, neigh_list, 0, RADAR_NUM_COLS, FIG_OUT, show_plots)
        )
        fig_paths.extend(
            plot_model_comparison_figures(
                y_test,
                metrics_by_model,
                predictions_by_model,
                save_dir=FIG_OUT,
                show=show_plots,
            )
        )

    return {
        "target": TARGET,
        "year_cutoff": year_cutoff,
        "train_years": S["train_years"],
        "test_years": S["test_years"],
        "n_train": len(train_df),
        "n_test": len(test_df),
        "metrics_by_model": metrics_by_model,
        "metrics_table": metrics_comparison_dataframe(metrics_by_model),
        "predictions_by_model": predictions_by_model,
        "pipelines": fitted,
        "y_test": y_test,
        "permutation_importance_mean": imp,
        "kmeans": {
            "n_clusters": best_k,
            "labels_train": labels_train,
            "labels_test": labels_test,
            "curves": curves,
            "profiles": prof_df,
            "profile_text": prof_text,
            "pca_projection": X2,
        },
        "decision_tree_text": tree_text,
        "narrative_comparison": narrative_cmp,
        "dataframe_export": export_all,
        "similar_neighbors_example": similar_test0,
        "artifact_dir": str(ML_OUT) if save_artifacts else None,
        "figure_paths": [str(p) for p in fig_paths] if save_artifacts else [],
    }
