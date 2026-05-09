"""
ETL Pipeline — Evasão Escolar em Recife
========================================
Etapas:
  1. EXTRACT  — lê os CSVs brutos
  2. TRANSFORM — limpa, enriquece e integra os dados
  3. LOAD      — salva em CSVs processados + banco SQLite

Integração `fato_integrado`:
  • Granularidade **escola–ano** (1 linha = 1 registro da base educacional em 1 ano).
  • Indicadores socioeconômicos municipais são **replicados** por ano em todas as escolas.
  • A série **município–ano** para gráficos agregados está em `dim_integrado_anual`.
"""

import logging
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
RAW_DIR = BASE_DIR / "data" / "raw"
PROC_DIR = BASE_DIR / "data" / "processed"
DB_PATH = PROC_DIR / "evasao_escolar.db"

PROC_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("etl")


# ===========================================================================
# EXTRACT
# ===========================================================================

def extract() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lê os arquivos CSV brutos e retorna os dois DataFrames."""
    log.info("EXTRACT — lendo arquivos brutos...")

    socio_path = RAW_DIR / "dados_socioeconomicos_recife.csv"
    educ_path = RAW_DIR / "dados_educacionais_recife.csv"

    df_socio = pd.read_csv(socio_path)
    df_educ = pd.read_csv(educ_path)

    log.info("  ✓ Socioeconômico: %d linhas × %d colunas", *df_socio.shape)
    log.info("  ✓ Educacional   : %d linhas × %d colunas", *df_educ.shape)

    return df_socio, df_educ


# ===========================================================================
# TRANSFORM
# ===========================================================================

def _limpar_base(df: pd.DataFrame, nome: str) -> pd.DataFrame:
    """Limpeza genérica aplicada a ambas as bases."""
    n_antes = len(df)

    df = df.drop_duplicates().copy()
    log.info("  [%s] Duplicatas removidas: %d", nome, n_antes - len(df))

    df["ano"] = df["ano"].astype(int)
    df["id_municipio"] = df["id_municipio"].astype(str)

    cols_num = df.select_dtypes(include="object").columns.difference(
        ["id_municipio", "id_municipio_nome"]
    )
    for col in cols_num:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    cols_taxa = [c for c in df.columns if c.startswith("taxa_") or c.startswith("tdi_")]
    for col in cols_taxa:
        if col in df.columns:
            df[col] = df[col].clip(lower=0, upper=100)

    return df


def _criar_nivel_risco(taxa: pd.Series, limites=(5, 10, 20)) -> pd.Series:
    """
    Classifica a taxa de abandono em níveis de risco.
    Ausentes não são tratados como zero — recebem rótulo explícito ``Sem dado``.
    """
    out = pd.Series(pd.NA, index=taxa.index, dtype="object")
    mask = taxa.notna()
    if mask.any():
        out.loc[mask] = pd.cut(
            taxa.loc[mask],
            bins=[-np.inf, limites[0], limites[1], limites[2], np.inf],
            labels=["Baixo", "Moderado", "Alto", "Crítico"],
        ).astype(str)
    out.loc[~mask] = "Sem dado"
    return out


def _periodo(ano: int) -> str:
    if ano <= 2010:
        return "2006–2010"
    if ano <= 2015:
        return "2011–2015"
    if ano <= 2019:
        return "2016–2019"
    if ano <= 2022:
        return "2020–2022 (Pandemia)"
    return "2023–2024"


def _indice_risco_evasao(df: pd.DataFrame) -> pd.Series:
    """
    Índice 0–100: média ponderada de evasão EM, TDI EM e repetência EM.
    Componentes ausentes são ignorados e os pesos são **renormalizados** sobre os
    disponíveis (não se assume 0 implícito).
    """
    spec = {
        "taxa_evasao_em": 0.40,
        "tdi_em": 0.30,
        "taxa_repetencia_em": 0.30,
    }
    cols = [c for c in spec if c in df.columns]
    if not cols:
        return pd.Series(np.nan, index=df.index)
    M = df[cols].to_numpy(dtype=float)
    w = np.array([spec[c] for c in cols], dtype=float)
    mask = np.isfinite(M) & ~np.isnan(M)
    numer = (np.where(mask, M, 0.0) * w.reshape(1, -1)).sum(axis=1)
    denom = (w.reshape(1, -1) * mask).sum(axis=1)
    score = np.divide(numer, denom, out=np.full(len(df), np.nan), where=denom > 0)
    return pd.Series(np.clip(score, 0, 100), index=df.index).round(2)


def transform(df_socio: pd.DataFrame, df_educ: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Transforma e enriquece as duas bases.
    Retorna um dicionário com todas as tabelas processadas.
    """
    log.info("TRANSFORM — iniciando...")

    df_socio = _limpar_base(df_socio, "socioeconomico")
    df_educ = _limpar_base(df_educ, "educacional")

    df_socio = df_socio.copy()
    df_educ = df_educ.copy()
    df_socio["periodo"] = df_socio["ano"].map(_periodo)
    df_educ["periodo"] = df_educ["ano"].map(_periodo)

    # Identificador estável por linha da base educacional (extrato sem código INEP de escola)
    df_educ = df_educ.sort_values(["ano", "id_municipio"], kind="mergesort").reset_index(drop=True)
    df_educ["id_linha_educacional"] = np.arange(len(df_educ), dtype=np.int64)

    for col in ["taxa_abandono_ef", "taxa_abandono_em"]:
        if col in df_educ.columns:
            suf = col.replace("taxa_abandono_", "")
            df_educ[f"risco_{suf}"] = _criar_nivel_risco(df_educ[col])

    cols_excluir_agg = ["id_municipio", "id_municipio_nome", "periodo"]

    # --- Série municipal: uma linha por ano (taxas socio + médias educacionais) ---
    socio_anual = (
        df_socio.drop(columns=[c for c in cols_excluir_agg if c in df_socio.columns])
        .groupby("ano", as_index=False)
        .mean(numeric_only=True)
        .round(2)
    )
    socio_anual["periodo"] = socio_anual["ano"].map(_periodo)
    socio_anual["id_municipio"] = socio_anual["ano"].map(
        df_socio.groupby("ano")["id_municipio"].first()
    )

    drop_educ = [
        c
        for c in cols_excluir_agg + ["risco_ef", "risco_em", "id_linha_educacional"]
        if c in df_educ.columns
    ]
    educ_anual = (
        df_educ.drop(columns=drop_educ, errors="ignore")
        .groupby("ano", as_index=False)
        .mean(numeric_only=True)
        .round(2)
    )
    educ_anual["periodo"] = educ_anual["ano"].map(_periodo)

    dim_integrado_anual = pd.merge(
        socio_anual,
        educ_anual,
        on=["ano", "periodo"],
        how="inner",
        suffixes=("_socio", "_educ"),
    )
    dim_integrado_anual["indice_risco_evasao"] = _indice_risco_evasao(dim_integrado_anual)

    n_anos_integrados = len(dim_integrado_anual)
    shape_antigo_integrado = (n_anos_integrados, dim_integrado_anual.shape[1])

    # --- Taxas municipais por ano (uma linha/ano) para join com a base escola–ano ---
    taxas_socio = [c for c in socio_anual.columns if c.startswith("taxa_")]
    socio_por_ano = socio_anual[["ano"] + taxas_socio].drop_duplicates(subset=["ano"])

    # --- Integração escola–ano (sem agregar a base educacional) ---
    fato_integrado = df_educ.merge(
        socio_por_ano,
        on="ano",
        how="inner",
        validate="many_to_one",
        suffixes=("", "_socio"),
    )

    # Remove colunas duplicadas por sufixo se o merge criou
    dup_cols = [c for c in fato_integrado.columns if c.endswith("_socio")]
    if dup_cols:
        fato_integrado = fato_integrado.drop(columns=dup_cols)

    fato_integrado["indice_risco_evasao"] = _indice_risco_evasao(fato_integrado)

    # Consistência e duplicatas
    n_dup_linhas = int(fato_integrado.duplicated().sum())
    if n_dup_linhas:
        log.warning("  [!] Linhas completamente duplicadas em fato_integrado: %d — removendo.", n_dup_linhas)
        fato_integrado = fato_integrado.drop_duplicates().reset_index(drop=True)

    n_unicos_id = fato_integrado["id_linha_educacional"].nunique()
    if n_unicos_id != len(fato_integrado):
        log.warning(
            "  [!] id_linha_educacional não é único por linha — verifique duplicatas na base bruta."
        )

    log.info(
        "  ✓ Comparativo granularidade: integração **antiga** (município–ano) ≈ shape %s → "
        "**nova** `fato_integrado` (escola–ano) shape %s",
        shape_antigo_integrado,
        fato_integrado.shape,
    )
    log.info(
        "  ✓ Amostras ML (`fato_integrado`): %d linhas | anos distintos: %d | ids educacionais únicos: %d",
        len(fato_integrado),
        fato_integrado["ano"].nunique(),
        n_unicos_id,
    )
    log.info("  ✓ dim_integrado_anual (município–ano para séries): %d anos", len(dim_integrado_anual))

    em_crit = df_educ["taxa_abandono_em"].notna() & (df_educ["taxa_abandono_em"] >= 10)
    ef_crit = df_educ["taxa_abandono_ef"].notna() & (df_educ["taxa_abandono_ef"] >= 5)
    escolas_risco = df_educ[em_crit | ef_crit].copy()
    escolas_risco = escolas_risco.sort_values(
        ["taxa_abandono_em", "taxa_abandono_ef"], ascending=False
    )
    log.info("  ✓ Escolas em risco: %d registros", len(escolas_risco))

    tendencia = dim_integrado_anual[["ano"]].copy()
    for col in ["taxa_evasao_em", "taxa_abandono_em", "tdi_em", "indice_risco_evasao"]:
        if col in dim_integrado_anual.columns:
            tendencia[col] = dim_integrado_anual[col].values
            tendencia[f"var_{col}"] = (
                dim_integrado_anual[col].pct_change(fill_method=None).mul(100).round(2)
            )

    log.info("TRANSFORM — concluído.")

    return {
        "fato_socioeconomico": df_socio,
        "fato_educacional": df_educ,
        "dim_socio_anual": socio_anual,
        "dim_educ_anual": educ_anual,
        "dim_integrado_anual": dim_integrado_anual,
        "fato_integrado": fato_integrado,
        "escolas_risco": escolas_risco,
        "tendencia_anual": tendencia,
    }


# ===========================================================================
# LOAD
# ===========================================================================

def _salvar_sqlite(tabelas: dict[str, pd.DataFrame], db_path: Path) -> None:
    """Persiste todas as tabelas no banco SQLite."""
    conn = sqlite3.connect(db_path)
    for nome, df in tabelas.items():
        df_save = df.copy()
        for col in df_save.select_dtypes(include="category").columns:
            df_save[col] = df_save[col].astype(str)
        df_save.to_sql(nome, conn, if_exists="replace", index=False)
        log.info("  ✓ SQLite → tabela '%s' (%d linhas)", nome, len(df_save))
    conn.close()


def load(tabelas: dict[str, pd.DataFrame]) -> None:
    """Salva todas as tabelas em CSVs e no banco SQLite."""
    log.info("LOAD — salvando dados processados...")

    for nome, df in tabelas.items():
        csv_path = PROC_DIR / f"{nome}.csv"
        df_save = df.copy()
        for col in df_save.select_dtypes(include="category").columns:
            df_save[col] = df_save[col].astype(str)
        df_save.to_csv(csv_path, index=False, encoding="utf-8-sig")
        log.info("  ✓ CSV  → %s (%d linhas)", csv_path.name, len(df_save))

    _salvar_sqlite(tabelas, DB_PATH)
    log.info("  ✓ Banco SQLite → %s", DB_PATH)
    log.info("LOAD — concluído.")


# ===========================================================================
# MAIN
# ===========================================================================

def run_etl() -> dict[str, pd.DataFrame]:
    log.info("=" * 55)
    log.info("  INICIANDO ETL — Evasão Escolar Recife")
    log.info("=" * 55)

    df_socio, df_educ = extract()
    socio_bruto = df_socio.copy()
    educ_bruto = df_educ.copy()
    tabelas = transform(df_socio, df_educ)
    load(tabelas)

    try:
        from etl.missing_report import save_missing_report

        path_rep = save_missing_report(socio_bruto, educ_bruto, tabelas, logger=log)
        log.info("  ✓ Relatório de ausentes → %s", path_rep)
    except Exception as exc:
        log.warning("  [!] Relatório de ausentes não gerado: %s", exc)

    log.info("=" * 55)
    log.info("  ETL FINALIZADO COM SUCESSO")
    log.info("  Banco: %s", DB_PATH)
    log.info("  CSVs : %s", PROC_DIR)
    log.info("=" * 55)

    return tabelas


if __name__ == "__main__":
    run_etl()
