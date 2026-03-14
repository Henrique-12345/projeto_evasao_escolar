"""
ETL Pipeline — Evasão Escolar em Recife
========================================
Etapas:
  1. EXTRACT  — lê os CSVs brutos
  2. TRANSFORM — limpa, enriquece e integra os dados
  3. LOAD      — salva em CSVs processados + banco SQLite
"""

import logging
import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR  = BASE_DIR / "data" / "raw"
PROC_DIR = BASE_DIR / "data" / "processed"
DB_PATH  = PROC_DIR / "evasao_escolar.db"

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
    educ_path  = RAW_DIR / "dados_educacionais_recife.csv"

    df_socio = pd.read_csv(socio_path)
    df_educ  = pd.read_csv(educ_path)

    log.info("  ✓ Socioeconômico: %d linhas × %d colunas", *df_socio.shape)
    log.info("  ✓ Educacional   : %d linhas × %d colunas", *df_educ.shape)

    return df_socio, df_educ


# ===========================================================================
# TRANSFORM
# ===========================================================================

def _limpar_base(df: pd.DataFrame, nome: str) -> pd.DataFrame:
    """Limpeza genérica aplicada em ambas as bases."""
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
    """Classifica a taxa de abandono/evasão em níveis de risco."""
    return pd.cut(
        taxa,
        bins=[-np.inf, limites[0], limites[1], limites[2], np.inf],
        labels=["Baixo", "Moderado", "Alto", "Crítico"],
    )


def transform(df_socio: pd.DataFrame, df_educ: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Transforma e enriquece as duas bases.
    Retorna um dicionário com todas as tabelas processadas.
    """
    log.info("TRANSFORM — iniciando...")

    # ------------------------------------------------------------------
    # Limpeza
    # ------------------------------------------------------------------
    df_socio = _limpar_base(df_socio, "socioeconomico")
    df_educ  = _limpar_base(df_educ,  "educacional")

    # ------------------------------------------------------------------
    # Classificação por período histórico
    # ------------------------------------------------------------------
    def periodo(ano):
        if ano <= 2010:   return "2006–2010"
        elif ano <= 2015: return "2011–2015"
        elif ano <= 2019: return "2016–2019"
        elif ano <= 2022: return "2020–2022 (Pandemia)"
        else:             return "2023–2024"

    df_socio = df_socio.copy()
    df_educ  = df_educ.copy()
    df_socio["periodo"] = df_socio["ano"].map(periodo)
    df_educ["periodo"]  = df_educ["ano"].map(periodo)

    # ------------------------------------------------------------------
    # Nível de risco na base educacional
    # ------------------------------------------------------------------
    for col in ["taxa_abandono_ef", "taxa_abandono_em"]:
        if col in df_educ.columns:
            df_educ[f"risco_{col.replace('taxa_abandono_', '')}"] = _criar_nivel_risco(
                df_educ[col].fillna(0)
            )

    # ------------------------------------------------------------------
    # Tabelas agregadas por ano (média entre escolas)
    # ------------------------------------------------------------------
    cols_excluir = ["id_municipio", "id_municipio_nome", "periodo"]

    socio_anual = (
        df_socio.drop(columns=cols_excluir)
        .groupby("ano", as_index=False)
        .mean()
        .round(2)
    )
    socio_anual["periodo"] = socio_anual["ano"].map(periodo)

    educ_anual = (
        df_educ.drop(columns=[c for c in cols_excluir + ["risco_ef", "risco_em"] if c in df_educ.columns])
        .groupby("ano", as_index=False)
        .mean()
        .round(2)
    )
    educ_anual["periodo"] = educ_anual["ano"].map(periodo)

    # ------------------------------------------------------------------
    # Base integrada (JOIN por ano)
    # ------------------------------------------------------------------
    df_merged = pd.merge(
        socio_anual, educ_anual,
        on=["ano", "periodo"],
        how="inner",
        suffixes=("_socio", "_educ"),
    )

    # Indicador composto: Índice de Risco de Evasão (0–100)
    #   Combina evasão EM (peso 40%), TDI EM (peso 30%), repetência EM (30%)
    cols_ire = {
        "taxa_evasao_em":   0.40,
        "tdi_em":           0.30,
        "taxa_repetencia_em": 0.30,
    }
    cols_ire_existentes = {c: w for c, w in cols_ire.items() if c in df_merged.columns}
    if cols_ire_existentes:
        df_merged["indice_risco_evasao"] = sum(
            df_merged[c].fillna(0) * w for c, w in cols_ire_existentes.items()
        ).round(2)

    log.info("  ✓ Base integrada: %d anos × %d variáveis", *df_merged.shape)

    # ------------------------------------------------------------------
    # Tabela de escolas com risco crítico (nível micro)
    # ------------------------------------------------------------------
    escolas_risco = df_educ[
        (df_educ["taxa_abandono_em"].fillna(0) >= 10) |
        (df_educ["taxa_abandono_ef"].fillna(0) >= 5)
    ].copy()
    escolas_risco = escolas_risco.sort_values(
        ["taxa_abandono_em", "taxa_abandono_ef"], ascending=False
    )
    log.info("  ✓ Escolas em risco: %d registros", len(escolas_risco))

    # ------------------------------------------------------------------
    # Tendências: variação percentual ano a ano
    # ------------------------------------------------------------------
    tendencia = df_merged[["ano"]].copy()
    for col in ["taxa_evasao_em", "taxa_abandono_em", "tdi_em", "indice_risco_evasao"]:
        if col in df_merged.columns:
            tendencia[col] = df_merged[col]
            tendencia[f"var_{col}"] = df_merged[col].pct_change(fill_method=None).mul(100).round(2)

    log.info("TRANSFORM — concluído.")

    return {
        "fato_socioeconomico":    df_socio,
        "fato_educacional":       df_educ,
        "dim_socio_anual":        socio_anual,
        "dim_educ_anual":         educ_anual,
        "fato_integrado":         df_merged,
        "escolas_risco":          escolas_risco,
        "tendencia_anual":        tendencia,
    }


# ===========================================================================
# LOAD
# ===========================================================================

def _salvar_sqlite(tabelas: dict[str, pd.DataFrame], db_path: Path) -> None:
    """Persiste todas as tabelas no banco SQLite."""
    conn = sqlite3.connect(db_path)
    for nome, df in tabelas.items():
        df_save = df.copy()
        # Converte colunas categóricas para string antes de salvar
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
    tabelas = transform(df_socio, df_educ)
    load(tabelas)

    log.info("=" * 55)
    log.info("  ETL FINALIZADO COM SUCESSO")
    log.info("  Banco: %s", DB_PATH)
    log.info("  CSVs : %s", PROC_DIR)
    log.info("=" * 55)

    return tabelas


if __name__ == "__main__":
    run_etl()
