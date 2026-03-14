"""
Dashboard — Evasão Escolar em Recife
======================================
Execute com:  streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Configuração da página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Evasão Escolar — Recife",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
ETL  = ROOT / "etl" / "etl_pipeline.py"

# ---------------------------------------------------------------------------
# Carregamento de dados (com cache)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Carregando dados processados...")
def carregar_dados() -> dict[str, pd.DataFrame]:
    tabelas = {}
    for csv in PROC.glob("*.csv"):
        tabelas[csv.stem] = pd.read_csv(csv)
    return tabelas


def garantir_dados():
    """Roda o ETL se os dados ainda não existirem."""
    if not PROC.exists() or not list(PROC.glob("*.csv")):
        with st.spinner("Executando ETL pela primeira vez..."):
            sys.path.insert(0, str(ROOT))
            from etl.etl_pipeline import run_etl
            run_etl()
        st.cache_data.clear()


# ---------------------------------------------------------------------------
# Helpers visuais
# ---------------------------------------------------------------------------
COR_EF   = "#3B82F6"
COR_EM   = "#EF4444"
COR_WARN = "#F59E0B"
COR_OK   = "#10B981"
COR_DARK = "#1E293B"


def hex_rgba(hex_color: str, alpha: float = 0.15) -> str:
    """Converte cor hex (#RRGGBB) para rgba(r,g,b,a) com transparência."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def scatter_com_tendencia(fig: go.Figure, x, y, name: str, cor: str,
                          text_vals=None) -> go.Figure:
    """Adiciona scatter + linha de tendência (regressão linear via numpy)."""
    mask = ~(np.isnan(x) | np.isnan(y))
    xc, yc = np.array(x)[mask], np.array(y)[mask]

    fig.add_trace(go.Scatter(
        x=xc, y=yc, mode="markers+text" if text_vals is not None else "markers",
        name=name, text=np.array(text_vals)[mask] if text_vals is not None else None,
        textposition="top center",
        marker=dict(color=cor, size=10, line=dict(color="white", width=1)),
    ))

    if len(xc) >= 2:
        z = np.polyfit(xc, yc, 1)
        p = np.poly1d(z)
        x_line = np.linspace(xc.min(), xc.max(), 100)
        fig.add_trace(go.Scatter(
            x=x_line, y=p(x_line),
            mode="lines", name=f"Tendência ({name})",
            line=dict(color=cor, width=2, dash="dash"),
            showlegend=False,
        ))
    return fig

PALETA_PERIODO = {
    "2006–2010":             "#94A3B8",
    "2011–2015":             "#60A5FA",
    "2016–2019":             "#34D399",
    "2020–2022 (Pandemia)":  "#FBBF24",
    "2023–2024":             "#F87171",
}


def kpi_card(col, titulo: str, valor: str, delta: str | None = None,
             cor: str = COR_DARK, ajuda: str = ""):
    with col:
        st.metric(label=titulo, value=valor, delta=delta, help=ajuda)


def titulo_secao(texto: str, icone: str = "📊"):
    st.markdown(f"### {icone} {texto}")
    st.markdown("---")


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------

def sidebar(dados: dict) -> dict:
    st.sidebar.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/Brasão_do_Recife.svg/200px-Brasão_do_Recife.svg.png",
        width=70,
    )
    st.sidebar.title("🎓 Evasão Escolar")
    st.sidebar.caption("Recife — Ensino Fundamental e Médio")
    st.sidebar.markdown("---")

    anos_disponiveis = sorted(dados["fato_integrado"]["ano"].unique())
    ano_min, ano_max = int(min(anos_disponiveis)), int(max(anos_disponiveis))

    filtros = {}

    filtros["ano_range"] = st.sidebar.slider(
        "Intervalo de Anos",
        min_value=ano_min, max_value=ano_max,
        value=(ano_min, ano_max), step=1,
    )

    filtros["nivel"] = st.sidebar.multiselect(
        "Nível de Ensino",
        options=["Ensino Fundamental (EF)", "Ensino Médio (EM)"],
        default=["Ensino Fundamental (EF)", "Ensino Médio (EM)"],
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Sobre os dados**")
    st.sidebar.caption(
        "Fonte: INEP / MEC. Município de Recife (código 2611606). "
        "Indicadores anuais de fluxo escolar das redes pública e privada."
    )
    st.sidebar.markdown("---")

    if st.sidebar.button("🔄 Reprocessar ETL"):
        st.cache_data.clear()
        sys.path.insert(0, str(ROOT))
        from etl.etl_pipeline import run_etl
        run_etl()
        st.rerun()

    return filtros


# ---------------------------------------------------------------------------
# PÁGINAS
# ---------------------------------------------------------------------------

def pagina_visao_geral(dados: dict, filtros: dict):
    st.title("📋 Visão Geral — Evasão Escolar em Recife")
    st.caption("Indicadores agregados por ano · Ensino Fundamental e Médio")
    st.markdown("---")

    df = dados["fato_integrado"].copy()
    a1, a2 = filtros["ano_range"]
    df = df[(df["ano"] >= a1) & (df["ano"] <= a2)]

    if df.empty:
        st.warning("Sem dados para o período selecionado.")
        return

    # --- KPIs ---
    ultimo = df.sort_values("ano").iloc[-1]
    penultimo = df.sort_values("ano").iloc[-2] if len(df) > 1 else ultimo

    def delta_str(col):
        if col not in ultimo or col not in penultimo:
            return None
        d = round(float(ultimo[col]) - float(penultimo[col]), 2)
        return f"{d:+.1f}pp vs. {int(penultimo['ano'])}"

    cols_kpi = st.columns(4)
    kpi_card(cols_kpi[0], "Evasão EF (último ano)",
             f"{ultimo.get('taxa_evasao_ef', '–'):.1f}%",
             delta_str("taxa_evasao_ef"),
             ajuda="Taxa média de evasão no Ensino Fundamental")
    kpi_card(cols_kpi[1], "Evasão EM (último ano)",
             f"{ultimo.get('taxa_evasao_em', '–'):.1f}%",
             delta_str("taxa_evasao_em"),
             ajuda="Taxa média de evasão no Ensino Médio")
    kpi_card(cols_kpi[2], "Abandono EM (último ano)",
             f"{ultimo.get('taxa_abandono_em', np.nan):.1f}%" if pd.notna(ultimo.get("taxa_abandono_em")) else "–",
             delta_str("taxa_abandono_em"),
             ajuda="Taxa média de abandono escolar no Ensino Médio")
    kpi_card(cols_kpi[3], "Índice de Risco (EM)",
             f"{ultimo.get('indice_risco_evasao', np.nan):.1f}" if pd.notna(ultimo.get("indice_risco_evasao")) else "–",
             delta_str("indice_risco_evasao"),
             ajuda="Índice composto: evasão EM (40%) + TDI EM (30%) + repetência EM (30%)")

    st.markdown("---")

    # --- Gráfico de linha: evolução temporal ---
    col1, col2 = st.columns(2)

    with col1:
        titulo_secao("Evolução da Evasão Escolar", "📈")
        fig = go.Figure()
        show_ef = "Ensino Fundamental (EF)" in filtros["nivel"]
        show_em = "Ensino Médio (EM)" in filtros["nivel"]

        if show_ef and "taxa_evasao_ef" in df.columns:
            fig.add_trace(go.Scatter(
                x=df["ano"], y=df["taxa_evasao_ef"],
                name="Evasão EF", mode="lines+markers",
                line=dict(color=COR_EF, width=3),
                marker=dict(size=8),
            ))
        if show_em and "taxa_evasao_em" in df.columns:
            fig.add_trace(go.Scatter(
                x=df["ano"], y=df["taxa_evasao_em"],
                name="Evasão EM", mode="lines+markers",
                line=dict(color=COR_EM, width=3),
                marker=dict(size=8),
            ))
        fig.update_layout(
            yaxis_title="Taxa (%)", xaxis_title="Ano",
            legend=dict(orientation="h", y=-0.2),
            hovermode="x unified", height=380,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        titulo_secao("Evolução do Abandono Escolar", "📉")
        df_educ_anual = dados["dim_educ_anual"].copy()
        df_educ_anual = df_educ_anual[(df_educ_anual["ano"] >= a1) & (df_educ_anual["ano"] <= a2)]
        fig2 = go.Figure()
        if show_ef and "taxa_abandono_ef" in df_educ_anual.columns:
            s = df_educ_anual.dropna(subset=["taxa_abandono_ef"])
            fig2.add_trace(go.Scatter(
                x=s["ano"], y=s["taxa_abandono_ef"],
                name="Abandono EF", mode="lines+markers",
                line=dict(color=COR_EF, width=3, dash="dot"),
                marker=dict(size=8),
            ))
        if show_em and "taxa_abandono_em" in df_educ_anual.columns:
            s = df_educ_anual.dropna(subset=["taxa_abandono_em"])
            fig2.add_trace(go.Scatter(
                x=s["ano"], y=s["taxa_abandono_em"],
                name="Abandono EM", mode="lines+markers",
                line=dict(color=COR_EM, width=3, dash="dot"),
                marker=dict(size=8),
            ))
        fig2.update_layout(
            yaxis_title="Taxa (%)", xaxis_title="Ano",
            legend=dict(orientation="h", y=-0.2),
            hovermode="x unified", height=380,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # --- Tabela resumo ---
    titulo_secao("Tabela Resumo por Ano", "📋")
    cols_exibir = [c for c in [
        "ano", "periodo",
        "taxa_evasao_ef", "taxa_evasao_em",
        "taxa_abandono_ef", "taxa_abandono_em",
        "tdi_ef", "tdi_em",
        "indice_risco_evasao",
    ] if c in df.columns]
    st.dataframe(
        df[cols_exibir].sort_values("ano", ascending=False).reset_index(drop=True),
        use_container_width=True,
        height=320,
    )


# ---------------------------------------------------------------------------

def pagina_fluxo_escolar(dados: dict, filtros: dict):
    st.title("🔄 Fluxo Escolar — Promoção, Repetência e Aprovação")
    st.markdown("---")

    a1, a2 = filtros["ano_range"]
    socio = dados["dim_socio_anual"].copy()
    educ  = dados["dim_educ_anual"].copy()
    socio = socio[(socio["ano"] >= a1) & (socio["ano"] <= a2)]
    educ  = educ[(educ["ano"] >= a1) & (educ["ano"] <= a2)]

    show_ef = "Ensino Fundamental (EF)" in filtros["nivel"]
    show_em = "Ensino Médio (EM)" in filtros["nivel"]

    # --- Promoção vs Repetência ---
    col1, col2 = st.columns(2)

    for col, nivel, show, cor_p, cor_r, nome in [
        (col1, "ef", show_ef, COR_EF, "#6366F1", "Ensino Fundamental"),
        (col2, "em", show_em, COR_EM, "#F97316", "Ensino Médio"),
    ]:
        with col:
            titulo_secao(f"{nome} — Promoção e Repetência", "📊")
            if not show:
                st.info(f"{nome} não selecionado no filtro.")
                continue
            fig = go.Figure()
            cp = f"taxa_promocao_{nivel}"
            cr = f"taxa_repetencia_{nivel}"
            s = socio.dropna(subset=[cp, cr])
            fig.add_trace(go.Bar(x=s["ano"], y=s[cp], name="Promoção",
                                 marker_color=cor_p, opacity=0.85))
            fig.add_trace(go.Bar(x=s["ano"], y=s[cr], name="Repetência",
                                 marker_color=cor_r, opacity=0.85))
            fig.update_layout(
                barmode="group", yaxis_title="%",
                xaxis_title="Ano", legend=dict(orientation="h", y=-0.2),
                height=380,
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # --- Aprovação, Reprovação e Abandono (Stacked 100%) ---
    titulo_secao("Composição do Resultado Escolar (Stacked)", "🧩")

    for nivel, show, nome in [("ef", show_ef, "EF"), ("em", show_em, "EM")]:
        if not show:
            continue
        cols_stk = [f"taxa_aprovacao_{nivel}", f"taxa_reprovacao_{nivel}", f"taxa_abandono_{nivel}"]
        cols_ok = [c for c in cols_stk if c in educ.columns]
        s = educ.dropna(subset=cols_ok)
        if s.empty:
            continue

        fig = go.Figure()
        nomes_stk = ["Aprovação", "Reprovação", "Abandono"]
        cores_stk = [COR_OK, COR_WARN, COR_EM]
        for c, n, cr in zip(cols_ok, nomes_stk, cores_stk):
            fig.add_trace(go.Bar(x=s["ano"], y=s[c], name=n,
                                 marker_color=cr, opacity=0.9))
        fig.update_layout(
            barmode="stack",
            title=f"Resultado Escolar — {nome}",
            yaxis_title="%", xaxis_title="Ano",
            legend=dict(orientation="h", y=-0.2),
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------

def pagina_distorcao_risco(dados: dict, filtros: dict):
    st.title("⚠️ Distorção Idade-Série e Análise de Risco")
    st.markdown("---")

    a1, a2 = filtros["ano_range"]
    educ_anual = dados["dim_educ_anual"].copy()
    educ_anual = educ_anual[(educ_anual["ano"] >= a1) & (educ_anual["ano"] <= a2)]
    integrado  = dados["fato_integrado"].copy()
    integrado  = integrado[(integrado["ano"] >= a1) & (integrado["ano"] <= a2)]

    col1, col2 = st.columns(2)

    with col1:
        titulo_secao("Taxa de Distorção Idade-Série (TDI)", "📐")
        fig = go.Figure()
        for nivel, cor, show in [
            ("ef", COR_EF, "Ensino Fundamental (EF)" in filtros["nivel"]),
            ("em", COR_EM, "Ensino Médio (EM)" in filtros["nivel"]),
        ]:
            if not show:
                continue
            s = educ_anual.dropna(subset=[f"tdi_{nivel}"])
            fig.add_trace(go.Scatter(
                x=s["ano"], y=s[f"tdi_{nivel}"],
                name=f"TDI {nivel.upper()}", mode="lines+markers",
                fill="tozeroy", fillcolor=hex_rgba(cor, 0.15),
                line=dict(color=cor, width=2.5),
                marker=dict(size=7),
            ))
        fig.add_hline(y=20, line_dash="dash", line_color="gray",
                      annotation_text="Ref. 20%", annotation_position="right")
        fig.update_layout(yaxis_title="TDI (%)", xaxis_title="Ano",
                          legend=dict(orientation="h", y=-0.2),
                          hovermode="x unified", height=380)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        titulo_secao("Índice de Risco de Evasão (Composto)", "🔴")
        if "indice_risco_evasao" in integrado.columns:
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=integrado["ano"], y=integrado["indice_risco_evasao"],
                marker=dict(
                    color=integrado["indice_risco_evasao"],
                    colorscale="RdYlGn_r",
                    showscale=True,
                    colorbar=dict(title="Risco"),
                ),
                text=integrado["indice_risco_evasao"].round(1),
                textposition="outside",
            ))
            fig2.update_layout(
                yaxis_title="Índice (0–100)", xaxis_title="Ano",
                height=380,
            )
            st.plotly_chart(fig2, use_container_width=True)
            st.caption("Índice = Evasão EM (40%) + TDI EM (30%) + Repetência EM (30%)")
        else:
            st.info("Índice de risco não disponível para o período.")

    st.markdown("---")

    # --- TDI vs. Abandono (scatter) ---
    titulo_secao("TDI × Abandono — Dispersão por Ano", "🔍")

    col3, col4 = st.columns(2)
    for col, nivel, show, cor in [
        (col3, "ef", "Ensino Fundamental (EF)" in filtros["nivel"], COR_EF),
        (col4, "em", "Ensino Médio (EM)" in filtros["nivel"], COR_EM),
    ]:
        with col:
            nome = "EF" if nivel == "ef" else "EM"
            if not show:
                st.info(f"{nome} não selecionado.")
                continue
            cols_req = [f"tdi_{nivel}", f"taxa_abandono_{nivel}"]
            s = integrado.dropna(subset=[c for c in cols_req if c in integrado.columns])
            if s.empty or not all(c in s.columns for c in cols_req):
                st.info(f"Dados insuficientes para {nome}.")
                continue
            fig = go.Figure()
            scatter_com_tendencia(
                fig,
                s[f"tdi_{nivel}"].values,
                s[f"taxa_abandono_{nivel}"].values,
                nome, cor, text_vals=s["ano"].values,
            )
            fig.update_layout(
                title=f"TDI vs. Abandono — {nome}",
                xaxis_title=f"TDI {nome} (%)",
                yaxis_title=f"Abandono {nome} (%)",
                showlegend=False, height=380,
            )
            st.plotly_chart(fig, use_container_width=True)

    # --- Escolas em risco ---
    st.markdown("---")
    titulo_secao("Escolas / Registros em Nível de Risco Elevado", "🏫")
    escolas = dados["escolas_risco"].copy()
    escolas_f = escolas[(escolas["ano"] >= a1) & (escolas["ano"] <= a2)]
    cols_exibir = [c for c in [
        "ano", "taxa_abandono_ef", "taxa_abandono_em",
        "tdi_ef", "tdi_em", "atu_ef", "atu_em",
    ] if c in escolas_f.columns]
    st.dataframe(
        escolas_f[cols_exibir].sort_values(["taxa_abandono_em", "ano"], ascending=[False, True]).reset_index(drop=True),
        use_container_width=True, height=350,
    )
    st.caption(f"Total de registros em risco no período: **{len(escolas_f)}**")


# ---------------------------------------------------------------------------

def pagina_correlacoes(dados: dict, filtros: dict):
    st.title("🔗 Correlações e Fatores de Risco")
    st.markdown("---")

    a1, a2 = filtros["ano_range"]
    df = dados["fato_integrado"].copy()
    df = df[(df["ano"] >= a1) & (df["ano"] <= a2)]

    cols_interesse = [
        "taxa_evasao_ef", "taxa_evasao_em",
        "taxa_abandono_ef", "taxa_abandono_em",
        "taxa_promocao_ef", "taxa_promocao_em",
        "taxa_repetencia_ef", "taxa_repetencia_em",
        "taxa_aprovacao_ef", "taxa_aprovacao_em",
        "taxa_reprovacao_ef", "taxa_reprovacao_em",
        "tdi_ef", "tdi_em",
        "atu_ef", "atu_em",
        "had_ef", "had_em",
    ]
    cols_ok = [c for c in cols_interesse if c in df.columns]
    df_corr = df[cols_ok].dropna(how="all")

    if len(df_corr) < 3:
        st.warning("Dados insuficientes para calcular correlações no período selecionado.")
        return

    corr = df_corr.corr()

    col1, col2 = st.columns([2, 1])

    with col1:
        titulo_secao("Matriz de Correlação (Pearson)", "🗺️")
        fig = px.imshow(
            corr,
            color_continuous_scale="RdYlGn",
            zmin=-1, zmax=1,
            text_auto=".2f",
            aspect="auto",
            title="Correlação entre Indicadores de Evasão",
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        titulo_secao("Top Correlações com Evasão EM", "🏆")
        if "taxa_evasao_em" in corr.columns:
            corr_em = (
                corr["taxa_evasao_em"]
                .drop("taxa_evasao_em")
                .dropna()
                .sort_values()
            )
            cores_bar = [COR_OK if v < 0 else COR_EM for v in corr_em]
            fig_bar = go.Figure(go.Bar(
                x=corr_em.values.round(2),
                y=corr_em.index,
                orientation="h",
                marker_color=cores_bar,
                text=corr_em.values.round(2),
                textposition="outside",
            ))
            fig_bar.add_vline(x=0, line_color="black", line_width=1)
            fig_bar.update_layout(
                xaxis_title="Coef. Pearson",
                yaxis_title="",
                height=600,
                xaxis=dict(range=[-1.2, 1.2]),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # --- Scatter: repetência vs evasão ---
    titulo_secao("Repetência × Evasão por Ano (Nível Macro)", "📌")
    col3, col4 = st.columns(2)
    for col, nivel, cor in [(col3, "ef", COR_EF), (col4, "em", COR_EM)]:
        with col:
            nome = "EF" if nivel == "ef" else "EM"
            s = dados["dim_socio_anual"].copy()
            s = s[(s["ano"] >= a1) & (s["ano"] <= a2)]
            xc, yc = f"taxa_repetencia_{nivel}", f"taxa_evasao_{nivel}"
            s = s.dropna(subset=[xc, yc])
            if s.empty:
                continue
            fig = go.Figure()
            scatter_com_tendencia(
                fig, s[xc].values, s[yc].values,
                nome, cor, text_vals=s["ano"].values,
            )
            fig.update_layout(
                title=f"Repetência vs. Evasão — {nome}",
                xaxis_title=f"Repetência {nome} (%)",
                yaxis_title=f"Evasão {nome} (%)",
                showlegend=False, height=380,
            )
            st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------

def pagina_infraestrutura(dados: dict, filtros: dict):
    st.title("🏗️ Infraestrutura Escolar — ATU e HAD")
    st.markdown("---")

    a1, a2 = filtros["ano_range"]
    educ_anual = dados["dim_educ_anual"].copy()
    educ_anual = educ_anual[(educ_anual["ano"] >= a1) & (educ_anual["ano"] <= a2)]

    show_ef = "Ensino Fundamental (EF)" in filtros["nivel"]
    show_em = "Ensino Médio (EM)" in filtros["nivel"]

    col1, col2 = st.columns(2)

    with col1:
        titulo_secao("Alunos por Turma (ATU)", "👨‍🎓")
        fig = go.Figure()
        if show_ef and "atu_ef" in educ_anual.columns:
            s = educ_anual.dropna(subset=["atu_ef"])
            fig.add_trace(go.Scatter(
                x=s["ano"], y=s["atu_ef"], name="ATU EF",
                mode="lines+markers", line=dict(color=COR_EF, width=3),
                marker=dict(size=8),
            ))
        if show_em and "atu_em" in educ_anual.columns:
            s = educ_anual.dropna(subset=["atu_em"])
            fig.add_trace(go.Scatter(
                x=s["ano"], y=s["atu_em"], name="ATU EM",
                mode="lines+markers", line=dict(color=COR_EM, width=3),
                marker=dict(size=8),
            ))
        fig.add_hline(y=30, line_dash="dash", line_color="gray",
                      annotation_text="Ref. 30 alunos/turma")
        fig.add_hline(y=35, line_dash="dot", line_color=COR_WARN,
                      annotation_text="Alerta 35+")
        fig.update_layout(yaxis_title="Alunos/turma", xaxis_title="Ano",
                          legend=dict(orientation="h", y=-0.2),
                          hovermode="x unified", height=380)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        titulo_secao("Horas-Aula Diárias (HAD)", "⏱️")
        fig2 = go.Figure()
        if show_ef and "had_ef" in educ_anual.columns:
            s = educ_anual.dropna(subset=["had_ef"])
            fig2.add_trace(go.Scatter(
                x=s["ano"], y=s["had_ef"], name="HAD EF",
                mode="lines+markers", line=dict(color=COR_EF, width=3),
                marker=dict(size=8),
            ))
        if show_em and "had_em" in educ_anual.columns:
            s = educ_anual.dropna(subset=["had_em"])
            fig2.add_trace(go.Scatter(
                x=s["ano"], y=s["had_em"], name="HAD EM",
                mode="lines+markers", line=dict(color=COR_EM, width=3),
                marker=dict(size=8),
            ))
        fig2.update_layout(yaxis_title="Horas-aula/dia", xaxis_title="Ano",
                           legend=dict(orientation="h", y=-0.2),
                           hovermode="x unified", height=380)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # --- ATU vs Abandono ---
    titulo_secao("ATU × Abandono — Relação entre Superlotação e Evasão", "🔬")
    integrado = dados["fato_integrado"].copy()
    integrado = integrado[(integrado["ano"] >= a1) & (integrado["ano"] <= a2)]

    col3, col4 = st.columns(2)
    for col, nivel, cor in [(col3, "ef", COR_EF), (col4, "em", COR_EM)]:
        with col:
            nome = "EF" if nivel == "ef" else "EM"
            show = show_ef if nivel == "ef" else show_em
            if not show:
                continue
            xc, yc = f"atu_{nivel}", f"taxa_abandono_{nivel}"
            s = integrado.dropna(subset=[c for c in [xc, yc] if c in integrado.columns])
            if s.empty or xc not in s.columns or yc not in s.columns:
                st.info(f"Dados insuficientes para {nome}.")
                continue
            fig = go.Figure()
            scatter_com_tendencia(
                fig, s[xc].values, s[yc].values,
                nome, cor, text_vals=s["ano"].values,
            )
            fig.update_layout(
                title=f"Superlotação vs. Abandono — {nome}",
                xaxis_title=f"ATU {nome}",
                yaxis_title=f"Abandono {nome} (%)",
                showlegend=False, height=380,
            )
            st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------

def pagina_tendencias(dados: dict, filtros: dict):
    st.title("📅 Tendências e Análise por Período")
    st.markdown("---")

    a1, a2 = filtros["ano_range"]
    tend = dados["tendencia_anual"].copy()
    tend = tend[(tend["ano"] >= a1) & (tend["ano"] <= a2)]
    integrado = dados["fato_integrado"].copy()
    integrado = integrado[(integrado["ano"] >= a1) & (integrado["ano"] <= a2)]

    # --- Variação percentual ano a ano ---
    titulo_secao("Variação Ano a Ano (%) — Evasão EM", "📊")
    if "var_taxa_evasao_em" in tend.columns:
        s = tend.dropna(subset=["var_taxa_evasao_em"])
        fig = go.Figure(go.Bar(
            x=s["ano"],
            y=s["var_taxa_evasao_em"],
            marker_color=[COR_EM if v > 0 else COR_OK for v in s["var_taxa_evasao_em"]],
            text=s["var_taxa_evasao_em"].apply(lambda x: f"{x:+.1f}%"),
            textposition="outside",
        ))
        fig.add_hline(y=0, line_color="black", line_width=1)
        fig.update_layout(
            yaxis_title="Variação (%)", xaxis_title="Ano",
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Vermelho = piora · Verde = melhora na evasão do Ensino Médio.")

    st.markdown("---")

    # --- Boxplot por período ---
    titulo_secao("Distribuição da Evasão por Período Histórico", "📦")
    col1, col2 = st.columns(2)
    for col, base, col_val, nome in [
        (col1, dados["fato_socioeconomico"], "taxa_evasao_em", "Evasão EM (Socio)"),
        (col2, dados["fato_educacional"],    "taxa_abandono_em", "Abandono EM (Educ)"),
    ]:
        with col:
            s = base.dropna(subset=[col_val])
            s = s[(s["ano"] >= a1) & (s["ano"] <= a2)]
            if s.empty:
                continue
            ordem = ["2006–2010", "2011–2015", "2016–2019", "2020–2022 (Pandemia)", "2023–2024"]
            ordem_existente = [p for p in ordem if p in s["periodo"].unique()]
            fig = px.box(
                s, x="periodo", y=col_val,
                category_orders={"periodo": ordem_existente},
                color="periodo",
                color_discrete_map=PALETA_PERIODO,
                title=nome,
                labels={"periodo": "Período", col_val: "%"},
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # --- Radar: perfil por período ---
    titulo_secao("Radar — Perfil de Risco por Período", "🕸️")
    cols_radar = ["taxa_evasao_em", "taxa_repetencia_em", "tdi_em", "atu_em"]
    cols_radar_ok = [c for c in cols_radar if c in integrado.columns]

    if cols_radar_ok and "periodo" in integrado.columns:
        radar_df = integrado.groupby("periodo")[cols_radar_ok].mean().round(2).reset_index()
        categorias = cols_radar_ok + [cols_radar_ok[0]]
        fig_radar = go.Figure()
        for _, row in radar_df.iterrows():
            valores = [row[c] for c in cols_radar_ok] + [row[cols_radar_ok[0]]]
            fig_radar.add_trace(go.Scatterpolar(
                r=valores, theta=categorias,
                fill="toself", name=row["periodo"],
                line=dict(color=PALETA_PERIODO.get(row["periodo"], "#888")),
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            showlegend=True, height=500,
        )
        st.plotly_chart(fig_radar, use_container_width=True)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    garantir_dados()
    dados = carregar_dados()

    if not dados:
        st.error("Não foi possível carregar os dados. Execute o ETL primeiro.")
        st.stop()

    filtros = sidebar(dados)

    paginas = {
        "📋 Visão Geral":              pagina_visao_geral,
        "🔄 Fluxo Escolar":            pagina_fluxo_escolar,
        "⚠️ Distorção e Risco":        pagina_distorcao_risco,
        "🔗 Correlações":              pagina_correlacoes,
        "🏗️ Infraestrutura (ATU/HAD)": pagina_infraestrutura,
        "📅 Tendências":               pagina_tendencias,
    }

    st.sidebar.markdown("---")
    pagina_atual = st.sidebar.radio("Navegação", list(paginas.keys()))

    paginas[pagina_atual](dados, filtros)

    st.sidebar.markdown("---")
    st.sidebar.caption("Projeto: Evasão Escolar · Recife · INEP/MEC")


if __name__ == "__main__":
    main()
