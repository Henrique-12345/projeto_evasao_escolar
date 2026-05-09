"""
Relatório de valores ausentes — rastreabilidade e qualidade de dados.

Gera `outputs/relatorio_missing_values.csv` e figuras em `outputs/figures/`.
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"


def _missing_summary(df: pd.DataFrame, nome_tabela: str, n_linhas: int, n_cols: int) -> pd.DataFrame:
    """Uma linha por coluna com contagens e percentuais."""
    rows = []
    for col in df.columns:
        n_nan = int(df[col].isna().sum())
        pct = round(100 * n_nan / n_linhas, 2) if n_linhas else 0.0
        rows.append(
            {
                "tabela": nome_tabela,
                "n_linhas_referencia": n_linhas,
                "n_colunas_referencia": n_cols,
                "coluna": col,
                "valores_ausentes": n_nan,
                "pct_ausentes": pct,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["rank_no_dataset"] = out["valores_ausentes"].rank(
            ascending=False, method="min"
        ).astype(int)
        out = out.sort_values(["valores_ausentes", "coluna"], ascending=[False, True])
    return out


def build_missing_report(
    df_socio_raw: pd.DataFrame,
    df_educ_raw: pd.DataFrame,
    tabelas_processadas: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Monta relatório consolidado: brutas + tabelas processadas.

    `tabelas_processadas` deve incluir `fato_socioeconomico`, `fato_educacional`,
    `fato_integrado` após o transform.

    Retorna ``(detalhe_por_coluna, resumo_global, resumo_formas)``.
    """
    partes = []

    partes.append(_missing_summary(df_socio_raw, "socioeconomico_bruta", *df_socio_raw.shape))
    partes.append(_missing_summary(df_educ_raw, "educacional_bruta", *df_educ_raw.shape))

    for key in ["fato_socioeconomico", "fato_educacional", "fato_integrado"]:
        if key not in tabelas_processadas:
            continue
        df = tabelas_processadas[key]
        partes.append(_missing_summary(df, f"{key}_processada", *df.shape))

    detalhe = pd.concat(partes, ignore_index=True)

    resumo_rows = []
    for nome, df in [
        ("socioeconomico_bruta", df_socio_raw),
        ("educacional_bruta", df_educ_raw),
        ("fato_integrado_processada", tabelas_processadas.get("fato_integrado")),
    ]:
        if df is None:
            continue
        n_cells = df.shape[0] * df.shape[1]
        n_nan = int(df.isna().sum().sum())
        resumo_rows.append(
            {
                "tabela": nome,
                "total_celulas": n_cells,
                "total_ausentes": n_nan,
                "pct_ausentes_global": round(100 * n_nan / n_cells, 4) if n_cells else 0,
            }
        )
    resumo_df = pd.DataFrame(resumo_rows)

    formas_rows = [
        {
            "fonte": "socioeconomica",
            "etapa": "bruta",
            "n_linhas": df_socio_raw.shape[0],
            "n_colunas": df_socio_raw.shape[1],
        },
        {
            "fonte": "educacional",
            "etapa": "bruta",
            "n_linhas": df_educ_raw.shape[0],
            "n_colunas": df_educ_raw.shape[1],
        },
    ]
    if "fato_socioeconomico" in tabelas_processadas:
        fs = tabelas_processadas["fato_socioeconomico"]
        formas_rows.append(
            {
                "fonte": "socioeconomica",
                "etapa": "processada (fato_socioeconomico)",
                "n_linhas": fs.shape[0],
                "n_colunas": fs.shape[1],
            }
        )
    if "fato_educacional" in tabelas_processadas:
        fe = tabelas_processadas["fato_educacional"]
        formas_rows.append(
            {
                "fonte": "educacional",
                "etapa": "processada (fato_educacional)",
                "n_linhas": fe.shape[0],
                "n_colunas": fe.shape[1],
            }
        )
    fi = tabelas_processadas.get("fato_integrado")
    if fi is not None:
        formas_rows.append(
            {
                "fonte": "integracao_escola_ano",
                "etapa": "processada (fato_integrado)",
                "n_linhas": fi.shape[0],
                "n_colunas": fi.shape[1],
            }
        )
    formas_df = pd.DataFrame(formas_rows)

    return detalhe, resumo_df, formas_df


def save_missing_report(
    df_socio_raw: pd.DataFrame,
    df_educ_raw: pd.DataFrame,
    tabelas_processadas: dict[str, pd.DataFrame],
    logger=None,
) -> Path:
    """Salva CSV e tenta gerar figuras."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    detalhe, resumo, formas = build_missing_report(df_socio_raw, df_educ_raw, tabelas_processadas)
    if logger is not None:
        log_missing_summary(detalhe, resumo, formas, logger)

    path_csv = OUTPUT_DIR / "relatorio_missing_values.csv"
    # CSV em seções comentadas: resumo, formas, detalhe por coluna
    with open(path_csv, "w", encoding="utf-8-sig") as f:
        f.write("# RESUMO_GLOBAL — totais de células e ausentes por tabela\n")
        resumo.to_csv(f, index=False)
        f.write("\n")
        f.write("# RESUMO_FORMAS — linhas × colunas (bruto vs processado)\n")
        formas.to_csv(f, index=False)
        f.write("\n")
        f.write("# DETALHE_POR_COLUNA\n")
        detalhe.to_csv(f, index=False)

    _try_plot_figures(df_socio_raw, df_educ_raw, tabelas_processadas.get("fato_integrado"))

    return path_csv


def load_missing_report_bundle(path: Path | None = None) -> dict[str, pd.DataFrame]:
    """
    Lê `relatorio_missing_values.csv` com seções `# RESUMO_*` e `# DETALHE_*`.

    Retorna chaves: ``resumo_global``, ``formas``, ``detalhe`` (DataFrames).
    """
    path = path or (OUTPUT_DIR / "relatorio_missing_values.csv")
    if not path.exists():
        raise FileNotFoundError(path)
    raw = path.read_text(encoding="utf-8-sig")
    sections: dict[str, pd.DataFrame] = {}
    pattern = r"^#\s+([^\n]+)\s*$"
    matches = list(re.finditer(pattern, raw, flags=re.MULTILINE))
    for i, m in enumerate(matches):
        title = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        block = raw[start:end].strip()
        if not block:
            continue
        key = title.split("—")[0].strip() if "—" in title else title
        if key.startswith("RESUMO_GLOBAL"):
            csv_key = "resumo_global"
        elif key.startswith("RESUMO_FORMAS"):
            csv_key = "formas"
        elif key.startswith("DETALHE_POR_COLUNA"):
            csv_key = "detalhe"
        else:
            csv_key = key
        try:
            sections[csv_key] = pd.read_csv(io.StringIO(block))
        except Exception:
            sections[csv_key] = pd.DataFrame()
    return sections


def _try_plot_figures(
    df_socio: pd.DataFrame,
    df_educ: pd.DataFrame,
    df_fi: pd.DataFrame | None,
) -> None:
    """Heatmap e barras (matplotlib/seaborn); falha silenciosa se não houver deps."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        return

    def plot_bars(df: pd.DataFrame, title: str, fname: str):
        na_pct = (df.isna().mean() * 100).sort_values(ascending=False)
        na_pct = na_pct[na_pct > 0]
        if na_pct.empty:
            return
        fig, ax = plt.subplots(figsize=(10, max(3, len(na_pct) * 0.22)))
        na_pct.plot(kind="barh", ax=ax, color="#64748B")
        ax.set_xlabel("% ausentes")
        ax.set_title(title)
        plt.tight_layout()
        fig.savefig(FIG_DIR / fname, dpi=120, bbox_inches="tight")
        plt.close(fig)

    plot_bars(df_educ, "Educacional (bruta) — % ausentes por coluna", "missing_bars_educacional.png")
    plot_bars(df_socio, "Socioeconômica (bruta) — % ausentes por coluna", "missing_bars_socio.png")

    if df_fi is not None and not df_fi.empty:
        # Heatmap: amostra de até 40 colunas para legibilidade
        num_cols = list(df_fi.columns)[:40]
        if num_cols:
            sub = df_fi[num_cols].isna().astype(int)
            h = min(30, len(sub))
            fig, ax = plt.subplots(figsize=(12, max(4, h * 0.15)))
            sns.heatmap(
                sub.iloc[:h],
                cbar=False,
                cmap="YlOrRd",
                yticklabels=False,
                xticklabels=True,
                ax=ax,
            )
            ax.set_title("fato_integrado — mapa de ausentes (1=ausente), primeiras linhas")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            fig.savefig(FIG_DIR / "missing_heatmap_fato_integrado_amostra.png", dpi=120, bbox_inches="tight")
            plt.close(fig)

        plot_bars(df_fi, "fato_integrado — % ausentes por coluna", "missing_bars_fato_integrado.png")


def log_missing_summary(
    detalhe: pd.DataFrame,
    resumo: pd.DataFrame,
    formas: pd.DataFrame,
    log,
) -> None:
    """Registra no logger do ETL os principais achados."""
    log.info("RELATÓRIO DE AUSENTES — resumo global:")
    for _, r in resumo.iterrows():
        log.info(
            "  %s: %d ausentes em %d células (%.4f%%)",
            r["tabela"],
            r["total_ausentes"],
            r["total_celulas"],
            r["pct_ausentes_global"],
        )
    if not formas.empty:
        log.info("RELATÓRIO DE AUSENTES — formas (bruto vs processado):")
        for _, r in formas.iterrows():
            log.info(
                "  %s | %s → %d × %d",
                r["fonte"],
                r["etapa"],
                int(r["n_linhas"]),
                int(r["n_colunas"]),
            )
    fi = detalhe[detalhe["tabela"] == "fato_integrado_processada"]
    top = fi.sort_values("valores_ausentes", ascending=False).head(5)
    if not top.empty:
        log.info("  Top 5 colunas com mais ausentes em fato_integrado:")
        for _, r in top.iterrows():
            log.info("    %s: %d (%.2f%%)", r["coluna"], r["valores_ausentes"], r["pct_ausentes"])
    log.info(
        "POLÍTICA DE AUSENTES — ETL preserva NaN onde não há valor observado; "
        "imputação sistemática apenas no Pipeline sklearn (treino). Ver docs/politica_dados_ausentes.md"
    )
