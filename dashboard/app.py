"""
Dashboard — Evasão Escolar em Recife
Execute com:  python -m streamlit run dashboard/app.py
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
# Configuração
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Evasão Escolar — Recife",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
    h1 { font-size: 1.6rem !important; font-weight: 700; color: #1E3A5F; }
    h2 { font-size: 1.2rem !important; font-weight: 600; color: #1E3A5F; }
    h3 { font-size: 1rem !important; font-weight: 600; color: #334155; }
    .stMetric label { font-size: 0.78rem !important; color: #64748B; }
    div[data-testid="stMetricValue"] { font-size: 1.4rem !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"

# ---------------------------------------------------------------------------
# Cores
# ---------------------------------------------------------------------------
COR_PRIMARIA = "#1E3A5F"
COR_EF       = "#2563EB"
COR_EM       = "#DC2626"
COR_ABANDONO = "#EA580C"
COR_OK       = "#15803D"
COR_CINZA    = "#64748B"
COR_CRITICO  = "#991B1B"

CORES_RISCO = {"Critico": "#991B1B", "Alto": "#DC2626", "Moderado": "#B45309", "Baixo": "#15803D"}

PALETA_PERIODO = {
    "2006–2010":             "#94A3B8",
    "2011–2015":             "#60A5FA",
    "2016–2019":             "#34D399",
    "2020–2022 (Pandemia)":  "#FBBF24",
    "2023–2024":             "#F87171",
}

LIMIARES = {"baixo": 20, "moderado": 35, "alto": 50}

# ---------------------------------------------------------------------------
# Glossário
# ---------------------------------------------------------------------------
GLOSSARIO = {
    "Evasão escolar":   "Saída definitiva do aluno do sistema de ensino.",
    "Abandono escolar": "Saída do aluno durante o ano letivo em curso (precursor da evasão).",
    "TDI":              "Taxa de Distorção Idade-Série — percentual de alunos com mais de 2 anos de atraso em relação à série esperada.",
    "p.p.":             "Ponto percentual — diferença direta entre dois percentuais (ex: de 10% para 12% = +2 p.p.).",
    "ATU":              "Média de Alunos por Turma.",
    "HAD":              "Horas-Aula Diárias.",
    "EF":               "Ensino Fundamental (1º ao 9º ano).",
    "EM":               "Ensino Médio (1º ao 3º ano).",
    "Score de Risco":   "Indicador 0–100: Abandono EM (40%) + TDI EM (30%) + Reprovação EM (30%).",
    "EJA":              "Educação de Jovens e Adultos — modalidade alternativa para quem não completou o ensino regular na idade certa.",
}

# ---------------------------------------------------------------------------
# Carregamento
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Carregando dados...")
def carregar_dados() -> dict[str, pd.DataFrame]:
    return {csv.stem: pd.read_csv(csv) for csv in PROC.glob("*.csv")}


def garantir_dados():
    if not PROC.exists() or not list(PROC.glob("*.csv")):
        with st.spinner("Executando ETL pela primeira vez..."):
            sys.path.insert(0, str(ROOT))
            from etl.etl_pipeline import run_etl
            run_etl()
        st.cache_data.clear()

# ---------------------------------------------------------------------------
# Score / risco
# ---------------------------------------------------------------------------
def calcular_score(df: pd.DataFrame) -> pd.Series:
    def col(c): return df[c].fillna(0) if c in df.columns else pd.Series(0.0, index=df.index)
    s = col("taxa_abandono_em") * 0.40 + col("tdi_em") * 0.30 + col("taxa_reprovacao_em") * 0.30
    mask = df["taxa_abandono_em"].notna() if "taxa_abandono_em" in df.columns else pd.Series(False, index=df.index)
    result = pd.Series(np.nan, index=df.index)
    result[mask] = s[mask].clip(0, 100)
    return result.round(1)


def classificar_risco(score: pd.Series) -> pd.Series:
    return pd.cut(score,
        bins=[-np.inf, LIMIARES["baixo"], LIMIARES["moderado"], LIMIARES["alto"], np.inf],
        labels=["Baixo", "Moderado", "Alto", "Critico"]).astype(str)

# ---------------------------------------------------------------------------
# Insights automáticos
# ---------------------------------------------------------------------------
def computar_insights(dados: dict) -> dict:
    """Calcula métricas-chave automaticamente para uso em todos os gráficos."""
    s = dados["dim_socio_anual"].copy().sort_values("ano")
    fi = dados["fato_integrado"].copy().sort_values("ano")
    out = {}

    if "taxa_evasao_em" in s.columns:
        em = s.dropna(subset=["taxa_evasao_em"])
        if not em.empty:
            out["pior_ano_evasao_em"]   = int(em.loc[em["taxa_evasao_em"].idxmax(), "ano"])
            out["pior_val_evasao_em"]   = round(em["taxa_evasao_em"].max(), 1)
            out["melhor_ano_evasao_em"] = int(em.loc[em["taxa_evasao_em"].idxmin(), "ano"])
            out["melhor_val_evasao_em"] = round(em["taxa_evasao_em"].min(), 1)
            if len(em) >= 2:
                out["delta_total_em"] = round(em.iloc[-1]["taxa_evasao_em"] - em.iloc[0]["taxa_evasao_em"], 1)
                out["ano_ini_em"]     = int(em.iloc[0]["ano"])
                out["ano_fim_em"]     = int(em.iloc[-1]["ano"])
                out["val_ini_em"]     = round(em.iloc[0]["taxa_evasao_em"], 1)
                out["val_fim_em"]     = round(em.iloc[-1]["taxa_evasao_em"], 1)

    if "indice_risco_evasao" in fi.columns:
        ri = fi.dropna(subset=["indice_risco_evasao"])
        if not ri.empty:
            out["score_atual"]  = round(ri.iloc[-1]["indice_risco_evasao"], 1)
            out["score_ant"]    = round(ri.iloc[-2]["indice_risco_evasao"], 1) if len(ri) > 1 else out["score_atual"]
            out["ano_score"]    = int(ri.iloc[-1]["ano"])
            out["nivel_score"]  = classificar_risco(pd.Series([out["score_atual"]])).iloc[0]

    tend = dados.get("tendencia_anual", pd.DataFrame())
    if "var_taxa_evasao_em" in tend.columns:
        t = tend.dropna(subset=["var_taxa_evasao_em"])
        if not t.empty:
            out["maior_alta_ano"] = int(t.loc[t["var_taxa_evasao_em"].idxmax(), "ano"])
            out["maior_alta_val"] = round(t["var_taxa_evasao_em"].max(), 1)
            out["maior_queda_ano"] = int(t.loc[t["var_taxa_evasao_em"].idxmin(), "ano"])
            out["maior_queda_val"] = round(t["var_taxa_evasao_em"].min(), 1)

    return out

# ---------------------------------------------------------------------------
# Componentes visuais
# ---------------------------------------------------------------------------
def hex_rgba(hex_color: str, alpha: float = 0.15) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def scatter_tendencia(fig: go.Figure, x, y, name: str, cor: str, texts=None) -> go.Figure:
    mask = ~(np.isnan(x) | np.isnan(y))
    xc, yc = np.array(x)[mask], np.array(y)[mask]
    fig.add_trace(go.Scatter(
        x=xc, y=yc,
        mode="markers+text" if texts is not None else "markers",
        name=name, text=np.array(texts)[mask] if texts is not None else None,
        textposition="top center",
        marker=dict(color=cor, size=10, line=dict(color="white", width=1)),
    ))
    if len(xc) >= 2:
        z = np.polyfit(xc, yc, 1)
        xl = np.linspace(xc.min(), xc.max(), 100)
        fig.add_trace(go.Scatter(
            x=xl, y=np.poly1d(z)(xl), mode="lines",
            line=dict(color=cor, width=2, dash="dash"),
            name="Tendencia", showlegend=False,
        ))
    return fig


def bloco_insight(tipo: str, titulo: str, texto: str):
    """Bloco compacto de insight — usa st nativo quando possivel."""
    emoji_map = {"bom": "✓", "ruim": "!", "info": "i", "acao": "→"}
    st.markdown(
        f"""<div style="padding:10px 16px;border-radius:4px;margin:6px 0 12px 0;
        border-left:4px solid {'#15803D' if tipo=='bom' else '#DC2626' if tipo=='ruim' else '#B45309' if tipo=='acao' else '#2563EB'};
        background:{'#F0FDF4' if tipo=='bom' else '#FEF2F2' if tipo=='ruim' else '#FFFBEB' if tipo=='acao' else '#EFF6FF'}">
        <b style="color:{'#14532D' if tipo=='bom' else '#7F1D1D' if tipo=='ruim' else '#78350F' if tipo=='acao' else '#1E40AF'};
        font-size:0.8rem;text-transform:uppercase;letter-spacing:0.04em">{titulo}</b>
        <p style="color:#1E293B;font-size:0.88rem;margin:4px 0 0 0;line-height:1.55">{texto}</p>
        </div>""",
        unsafe_allow_html=True,
    )


def transicao_pagina(texto: str):
    """Linha de conexão entre páginas."""
    st.markdown(
        f'<p style="color:#64748B;font-size:0.85rem;font-style:italic;'
        f'border-top:1px solid #E2E8F0;margin-top:20px;padding-top:10px">{texto}</p>',
        unsafe_allow_html=True,
    )


def secao(titulo: str, contexto: str = ""):
    st.markdown(f"### {titulo}")
    if contexto:
        st.caption(contexto)


def pandemia_vrect(fig, a1, a2):
    if a1 <= 2020 <= a2:
        fig.add_vrect(x0=2019.5, x1=2022.5, fillcolor="#FEF9C3", opacity=0.3, line_width=0,
                      annotation_text="Pandemia (2020–22)", annotation_position="top left",
                      annotation_font=dict(size=10, color="#92400E"))

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
def sidebar(dados: dict) -> dict:
    st.sidebar.markdown(
        f'<p style="color:{COR_PRIMARIA};font-size:1rem;font-weight:700;margin:0">Evasão Escolar — Recife</p>'
        '<p style="color:#64748B;font-size:0.78rem;margin:2px 0 0 0">INEP / MEC | 2008–2022</p>',
        unsafe_allow_html=True,
    )
    st.sidebar.divider()

    anos = sorted(dados["fato_integrado"]["ano"].unique())
    a_min, a_max = int(min(anos)), int(max(anos))
    default_ini = max(a_min, a_max - 3)

    filtros = {}
    filtros["ano_range"] = st.sidebar.slider(
        "Periodo de analise", min_value=a_min, max_value=a_max,
        value=(default_ini, a_max),
        help="Padrao: ultimos 4 anos. Amplie para ver a serie historica completa.",
    )
    filtros["nivel"] = st.sidebar.multiselect(
        "Nivel de ensino",
        ["Ensino Fundamental (EF)", "Ensino Medio (EM)"],
        default=["Ensino Fundamental (EF)", "Ensino Medio (EM)"],
    )
    filtros["risco_filtro"] = st.sidebar.multiselect(
        "Nivel de risco (ranking)",
        ["Baixo", "Moderado", "Alto", "Critico"],
        default=["Alto", "Critico"],
    )

    st.sidebar.divider()
    with st.sidebar.expander("Escala de risco"):
        st.caption("""
        **Critico** — Score acima de 50  
        **Alto** — Score entre 35 e 50  
        **Moderado** — Score entre 20 e 35  
        **Baixo** — Score abaixo de 20  
        *(Score = Abandono 40% + TDI 30% + Reprovação 30%)*
        """)
    with st.sidebar.expander("Glossario de termos"):
        for termo, defn in GLOSSARIO.items():
            st.markdown(f"**{termo}:** {defn}")

    st.sidebar.divider()
    if st.sidebar.button("Reprocessar ETL"):
        st.cache_data.clear()
        sys.path.insert(0, str(ROOT))
        from etl.etl_pipeline import run_etl
        run_etl()
        st.rerun()

    return filtros

# ===========================================================================
# PÁGINA 1 — VISÃO GERAL: "Qual é o problema?"
# ===========================================================================
def pagina_visao_geral(dados: dict, filtros: dict, insights: dict):
    st.markdown("# Visão Geral — Qual é o problema?")
    st.caption(
        "Ponto de partida do painel. Mostre esta página para qualquer pessoa que precise "
        "entender rapidamente a situação da evasão escolar em Recife."
    )

    a1, a2 = filtros["ano_range"]
    df = dados["fato_integrado"].copy()
    df = df[(df["ano"] >= a1) & (df["ano"] <= a2)].sort_values("ano")
    df_educ = dados["fato_educacional"].copy()
    df_educ = df_educ[(df_educ["ano"] >= a1) & (df_educ["ano"] <= a2)]

    if df.empty:
        st.warning(f"Nenhum dado disponível para {a1}–{a2}. Ajuste o filtro.")
        return

    ultimo = df.iloc[-1]
    ant    = df.iloc[-2] if len(df) > 1 else ultimo

    def sv(c): return float(ultimo.get(c, np.nan) or np.nan)
    def dpp(c):
        v1, v2 = sv(c), float(ant.get(c, np.nan) or np.nan)
        return f"{v1-v2:+.1f} p.p." if pd.notna(v1) and pd.notna(v2) else None

    # ── Diagnóstico automático principal ──────────────────────────────────────
    sc    = insights.get("score_atual", 0)
    nivel = insights.get("nivel_score", "Baixo")
    delta_sc = sc - insights.get("score_ant", sc)
    pior_ano = insights.get("pior_ano_evasao_em")
    melhor_ano = insights.get("melhor_ano_evasao_em")
    delta_total = insights.get("delta_total_em")

    # Mensagem principal automática
    if nivel in ("Critico", "Alto"):
        st.error(
            f"**Situacao critica:** Score de Risco atual = {sc:.0f}/100 (nivel {nivel}). "
            f"Os tres principais indicadores — abandono, defasagem escolar (TDI) e reprovacao — "
            f"estao simultaneamente elevados. Acao imediata e necessaria."
        )
    elif nivel == "Moderado":
        st.warning(
            f"**Atencao:** Score de Risco atual = {sc:.0f}/100 (nivel {nivel}). "
            f"Os indicadores nao estao em crise, mas exigem monitoramento proximo."
        )
    else:
        st.success(
            f"**Situacao sob controle:** Score de Risco atual = {sc:.0f}/100 (nivel {nivel}). "
            f"Continue monitorando para identificar tendencias antes que se agravem."
        )

    if delta_sc > 2:
        st.warning(f"**Piora detectada:** o risco subiu {delta_sc:+.1f} pontos em relacao ao ano anterior.")
    elif delta_sc < -2:
        st.success(f"**Melhora detectada:** o risco caiu {abs(delta_sc):.1f} pontos em relacao ao ano anterior.")

    if pior_ano and 2020 <= pior_ano <= 2022:
        st.info(
            f"**Pior ano da serie:** {pior_ano} ({insights.get('pior_val_evasao_em', '?')}% de evasao no EM) — "
            f"coincide com a pandemia de COVID-19, que fechou escolas e gerou crise economica."
        )
    elif pior_ano:
        st.info(f"**Pior ano da serie:** {pior_ano} ({insights.get('pior_val_evasao_em', '?')}% de evasao no EM).")

    if melhor_ano:
        st.success(
            f"**Melhor ano da serie:** {melhor_ano} ({insights.get('melhor_val_evasao_em', '?')}% de evasao no EM). "
            f"Este e o nivel de referencia a ser recuperado e superado."
        )

    if delta_total is not None:
        if delta_total < -5:
            st.success(
                f"**Tendencia de longo prazo:** queda de {abs(delta_total):.1f} p.p. na evasao do EM "
                f"entre {insights.get('ano_ini_em')} e {insights.get('ano_fim_em')}. "
                "As politicas educacionais produziram avancos reais ao longo dos anos."
            )
        elif delta_total > 2:
            st.warning(
                f"**Tendencia preocupante:** aumento de {delta_total:.1f} p.p. na evasao do EM "
                f"entre {insights.get('ano_ini_em')} e {insights.get('ano_fim_em')}."
            )

    st.divider()

    # ── KPIs ──────────────────────────────────────────────────────────────────
    secao(f"Indicadores do ano de referencia: {int(ultimo['ano'])}",
          "p.p. = ponto percentual. Seta vermelha = piora. Seta verde = melhora.")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Evasao EF (%)", f"{sv('taxa_evasao_ef'):.1f}%" if pd.notna(sv('taxa_evasao_ef')) else "–",
                  dpp("taxa_evasao_ef"), delta_color="inverse",
                  help="Percentual de alunos do EF que abandonaram definitivamente o sistema escolar.")
    with c2:
        st.metric("Evasao EM (%)", f"{sv('taxa_evasao_em'):.1f}%" if pd.notna(sv('taxa_evasao_em')) else "–",
                  dpp("taxa_evasao_em"), delta_color="inverse",
                  help="Evasao no Ensino Medio — historicamente 2 a 3 vezes maior que no EF.")
    with c3:
        st.metric("Abandono EM (%)", f"{sv('taxa_abandono_em'):.1f}%" if pd.notna(sv('taxa_abandono_em')) else "–",
                  dpp("taxa_abandono_em"), delta_color="inverse",
                  help="Saidas durante o ano letivo — sinal mais imediato de evasao futura.")
    with c4:
        st.metric("TDI EM (%)", f"{sv('tdi_em'):.1f}%" if pd.notna(sv('tdi_em')) else "–",
                  dpp("tdi_em"), delta_color="inverse",
                  help="TDI = Distorcao Idade-Serie. Alunos cursando serie muito abaixo da esperada para sua idade.")
    with c5:
        st.metric("Score de Risco (0–100)", f"{sc:.0f}",
                  f"{delta_sc:+.1f} vs. {int(ant['ano'])}" if delta_sc != 0 else None,
                  delta_color="inverse",
                  help="Score composto: quanto maior, maior o risco de evasao.")

    st.divider()

    # ── Gauge + Evolucao do score ──────────────────────────────────────────────
    col1, col2 = st.columns([1, 2])

    with col1:
        secao("Nivel de risco atual", "Verde = baixo | Amarelo = moderado | Vermelho = alto/critico")
        if pd.notna(sc):
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=float(sc),
                delta={"reference": insights.get("score_ant", sc), "valueformat": ".0f",
                       "increasing": {"color": COR_CRITICO}, "decreasing": {"color": COR_OK}},
                number={"suffix": " / 100", "font": {"size": 24}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": CORES_RISCO.get(nivel, COR_CINZA)},
                    "steps": [
                        {"range": [0, 20],  "color": "#DCFCE7"},
                        {"range": [20, 35], "color": "#FEF9C3"},
                        {"range": [35, 50], "color": "#FEE2E2"},
                        {"range": [50, 100],"color": "#FECACA"},
                    ],
                },
                title={"text": nivel, "font": {"size": 14}},
            ))
            fig_g.update_layout(height=250, margin=dict(t=40, b=0, l=20, r=20))
            st.plotly_chart(fig_g, use_container_width=True)

    with col2:
        secao("Como o risco evoluiu ao longo dos anos?")
        if "indice_risco_evasao" in df.columns:
            dt = df.dropna(subset=["indice_risco_evasao"])
            if not dt.empty:
                fig_t = go.Figure()
                for y0, y1, cor_f, lb in [(0,20,"#DCFCE7","Baixo"),(20,35,"#FEF9C3","Moderado"),(35,50,"#FEE2E2","Alto"),(50,100,"#FECACA","Critico")]:
                    fig_t.add_hrect(y0=y0, y1=y1, fillcolor=cor_f, opacity=0.25, line_width=0,
                                    annotation_text=lb, annotation_position="right")
                fig_t.add_trace(go.Scatter(
                    x=dt["ano"], y=dt["indice_risco_evasao"],
                    mode="lines+markers+text",
                    text=dt["indice_risco_evasao"].round(0).astype(int),
                    textposition="top center",
                    line=dict(color=COR_PRIMARIA, width=3),
                    marker=dict(size=9, color=dt["indice_risco_evasao"],
                                colorscale="RdYlGn_r", cmin=0, cmax=60,
                                line=dict(color="white", width=2)),
                ))
                pandemia_vrect(fig_t, a1, a2)
                fig_t.update_layout(yaxis_title="Score (0–100)", xaxis_title="Ano",
                                    hovermode="x unified", height=250, showlegend=False,
                                    margin=dict(t=10, b=10))
                st.plotly_chart(fig_t, use_container_width=True)
                bloco_insight("info", "O que este grafico mostra",
                    "Cada ponto e um ano. Pontos mais altos = maior risco. "
                    "A queda consistente antes de 2020 mostra que as politicas educacionais funcionaram. "
                    "O aumento durante a pandemia foi um choque externo — nao uma falha estrutural do sistema.")

    st.divider()

    # ── Comparação início vs fim do período ────────────────────────────────────
    secao("Quanto os indicadores mudaram no periodo selecionado?",
          f"Comparacao entre {int(df.iloc[0]['ano'])} e {int(ultimo['ano'])}.")

    indicadores_comp = [
        ("taxa_evasao_em",     "Evasao EM",     True),
        ("taxa_abandono_em",   "Abandono EM",   True),
        ("tdi_em",             "TDI EM",        True),
        ("taxa_repetencia_em", "Reprovacao EM", True),
        ("taxa_aprovacao_em",  "Aprovacao EM",  False),
    ]
    primeiro = df.iloc[0]
    colunas = st.columns(len(indicadores_comp))
    for col_ui, (nome, label, inv) in zip(colunas, indicadores_comp):
        v_ini = float(primeiro.get(nome, np.nan) or np.nan)
        v_fim = float(ultimo.get(nome, np.nan) or np.nan)
        if pd.notna(v_ini) and pd.notna(v_fim):
            with col_ui:
                st.metric(label, f"{v_fim:.1f}%",
                          f"{v_fim-v_ini:+.1f} p.p.",
                          delta_color="inverse" if inv else "normal")

    melhoras = sum(1 for nome, _, inv in indicadores_comp
                   if pd.notna(float(primeiro.get(nome, np.nan) or np.nan))
                   and pd.notna(float(ultimo.get(nome, np.nan) or np.nan))
                   and ((inv and float(ultimo.get(nome,0) or 0) < float(primeiro.get(nome,0) or 0))
                        or (not inv and float(ultimo.get(nome,0) or 0) > float(primeiro.get(nome,0) or 0))))
    total_comp = sum(1 for nome, _, _ in indicadores_comp
                     if pd.notna(float(primeiro.get(nome, np.nan) or np.nan)))
    if total_comp > 0:
        if melhoras == total_comp:
            st.success(f"Todos os {total_comp} indicadores melhoraram no periodo {int(df.iloc[0]['ano'])}–{int(ultimo['ano'])}.")
        elif melhoras > total_comp // 2:
            st.info(f"{melhoras} de {total_comp} indicadores melhoraram no periodo.")
        else:
            st.warning(f"Apenas {melhoras} de {total_comp} indicadores melhoraram. O periodo foi de deterioracao.")

    transicao_pagina(
        "Para entender como esses numeros evoluiram ano a ano, acesse 'Evolucao Historica'. "
        "Para saber quais anos e periodos concentram maior risco, acesse 'Identificacao de Risco'."
    )


# ===========================================================================
# PÁGINA 2 — EVOLUÇÃO HISTÓRICA: "Como o problema mudou ao longo do tempo?"
# ===========================================================================
def pagina_evolucao(dados: dict, filtros: dict, insights: dict):
    st.markdown("# Evolucao Historica — Como o problema mudou ao longo do tempo?")
    st.caption(
        "Analise a trajetoria da evasao ao longo dos anos. "
        "Entenda quando melhorou, quando piorou e por que."
    )

    a1, a2 = filtros["ano_range"]
    df_int  = dados["fato_integrado"].copy()
    df_int  = df_int[(df_int["ano"] >= a1) & (df_int["ano"] <= a2)]
    df_educ = dados["dim_educ_anual"].copy()
    df_educ = df_educ[(df_educ["ano"] >= a1) & (df_educ["ano"] <= a2)]
    df_soc  = dados["dim_socio_anual"].copy()
    df_soc  = df_soc[(df_soc["ano"] >= a1) & (df_soc["ano"] <= a2)]
    tend    = dados["tendencia_anual"].copy()
    tend    = tend[(tend["ano"] >= a1) & (tend["ano"] <= a2)]

    show_ef = "Ensino Fundamental (EF)" in filtros["nivel"]
    show_em = "Ensino Medio (EM)"        in filtros["nivel"]

    # ── Narrativa histórica condensada ────────────────────────────────────────
    if a1 <= 2022 and a2 >= 2019:
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.info("**2008–2019**\nQueda consistente. Expansão do acesso à escola e programas sociais funcionaram.")
        with col_b:
            st.warning("**2020**\nPandemia interrompeu o progresso. Fechamento das escolas, ensino remoto desigual e crise econômica.")
        with col_c:
            st.error("**2021**\nPico da crise. Efeitos acumulados de 2020 + alunos que não retornaram + subnotificação anterior.")
        with col_d:
            st.success("**2022 em diante**\nRetorno gradual. Recuperação lenta — o sistema nao volta ao nivel pre-pandemia rapidamente.")
        st.divider()

    # ── Gráfico principal ──────────────────────────────────────────────────────
    secao("A evasao e o abandono aumentaram ou diminuiram?",
          "Linha solida = evasao definitiva. Linha pontilhada = abandono no ano letivo. Azul = EF. Vermelho = EM.")

    fig = go.Figure()
    pandemia_vrect(fig, a1, a2)
    for nivel, show, c_ev, c_ab, nome in [
        ("ef", show_ef, COR_EF, "#93C5FD", "EF"),
        ("em", show_em, COR_EM, "#FCA5A5", "EM"),
    ]:
        if not show: continue
        s_s = df_soc.dropna(subset=[f"taxa_evasao_{nivel}"])
        s_e = df_educ.dropna(subset=[f"taxa_abandono_{nivel}"])
        if not s_s.empty:
            fig.add_trace(go.Scatter(x=s_s["ano"], y=s_s[f"taxa_evasao_{nivel}"],
                name=f"Evasao {nome}", mode="lines+markers",
                line=dict(color=c_ev, width=3), marker=dict(size=8)))
        if not s_e.empty:
            fig.add_trace(go.Scatter(x=s_e["ano"], y=s_e[f"taxa_abandono_{nivel}"],
                name=f"Abandono {nome}", mode="lines+markers",
                line=dict(color=c_ab, width=2, dash="dot"), marker=dict(size=7)))
    fig.update_layout(yaxis_title="Taxa (%)", xaxis_title="Ano",
                      hovermode="x unified", height=380,
                      legend=dict(orientation="h", y=-0.22))
    st.plotly_chart(fig, use_container_width=True)

    # Insight automático
    if "taxa_evasao_em" in df_soc.columns:
        s_em = df_soc.dropna(subset=["taxa_evasao_em"]).sort_values("ano")
        if len(s_em) >= 2:
            v_ini, v_fim = s_em.iloc[0]["taxa_evasao_em"], s_em.iloc[-1]["taxa_evasao_em"]
            delta = round(v_fim - v_ini, 1)
            pico_ano = int(s_em.loc[s_em["taxa_evasao_em"].idxmax(), "ano"])
            pico_val = round(s_em["taxa_evasao_em"].max(), 1)
            if delta < 0:
                bloco_insight("bom", "Tendencia positiva no periodo",
                    f"A evasao no EM caiu {abs(delta)} p.p. no periodo analisado "
                    f"(de {v_ini:.1f}% para {v_fim:.1f}%). "
                    f"O pico foi em {pico_ano} ({pico_val}%), "
                    f"{'durante a pandemia — um choque externo, nao uma falha estrutural.' if 2020 <= pico_ano <= 2022 else 'marcado por condicoes adversas.'} "
                    "Acao recomendada: mantenha as politicas que geraram a queda e monitore os anos recentes.")
            else:
                bloco_insight("ruim", "Atencao: piora no periodo",
                    f"A evasao no EM subiu {delta} p.p. no periodo analisado. "
                    "Acao recomendada: identifique os anos de maior crescimento e investigue suas causas.")

    st.divider()

    # ── Variação YoY ──────────────────────────────────────────────────────────
    secao("Em quais anos a evasao piorou ou melhorou?",
          "Barra vermelha = evasao subiu em relacao ao ano anterior. Barra verde = evasao caiu.")

    if "var_taxa_evasao_em" in tend.columns:
        s = tend.dropna(subset=["var_taxa_evasao_em"])
        if not s.empty:
            pior = s.loc[s["var_taxa_evasao_em"].idxmax()]
            melhor = s.loc[s["var_taxa_evasao_em"].idxmin()]
            fig_y = go.Figure(go.Bar(
                x=s["ano"], y=s["var_taxa_evasao_em"],
                marker_color=[COR_EM if v > 0 else COR_OK for v in s["var_taxa_evasao_em"]],
                text=s["var_taxa_evasao_em"].apply(lambda x: f"{x:+.1f}%"),
                textposition="outside",
            ))
            fig_y.add_hline(y=0, line_color=COR_CINZA, line_width=1.5)
            fig_y.add_annotation(x=int(pior["ano"]), y=float(pior["var_taxa_evasao_em"]),
                text=f"Maior piora: {int(pior['ano'])}", showarrow=True, arrowhead=2,
                ax=0, ay=-30, font=dict(color=COR_EM, size=10))
            fig_y.add_annotation(x=int(melhor["ano"]), y=float(melhor["var_taxa_evasao_em"]),
                text=f"Maior melhora: {int(melhor['ano'])}", showarrow=True, arrowhead=2,
                ax=0, ay=30, font=dict(color=COR_OK, size=10))
            fig_y.update_layout(yaxis_title="Variacao vs. ano anterior (%)",
                                xaxis_title="Ano", height=340)
            st.plotly_chart(fig_y, use_container_width=True)

            pior_pandemia = 2020 <= int(pior["ano"]) <= 2022
            if pior_pandemia:
                bloco_insight("info", f"Por que {int(pior['ano'])} foi o pior ano",
                    f"O aumento de {float(pior['var_taxa_evasao_em']):.1f}% em {int(pior['ano'])} "
                    "foi causado pela pandemia de COVID-19: fechamento das escolas, desigualdade no acesso ao ensino remoto "
                    "e crise economica que levou jovens ao mercado de trabalho. "
                    "Nao foi uma falha do sistema educacional — foi um choque externo de larga escala.")
            else:
                bloco_insight("info", f"Pior ano: {int(pior['ano'])}",
                    f"A evasao subiu {float(pior['var_taxa_evasao_em']):.1f}% em relacao ao ano anterior. "
                    "Investigue se houve mudancas de politica, cortes de orcamento ou fatores externos nesse ano.")
        else:
            st.warning("Dados insuficientes para calcular variacao anual. Amplie o periodo de analise.")

    st.divider()

    # ── EF vs EM ──────────────────────────────────────────────────────────────
    secao("O Ensino Medio e mais critico que o Ensino Fundamental?",
          "Barras = evasao por nivel de ensino. Linha pontilhada = quantas vezes o EM e maior que o EF.")

    if all(c in df_int.columns for c in ["taxa_evasao_ef", "taxa_evasao_em"]):
        dc = df_int.dropna(subset=["taxa_evasao_ef", "taxa_evasao_em"]).sort_values("ano").copy()
        dc = dc[dc["taxa_evasao_ef"] > 0]
        if not dc.empty:
            dc["razao"] = (dc["taxa_evasao_em"] / dc["taxa_evasao_ef"]).round(2)
            fig_c = make_subplots(specs=[[{"secondary_y": True}]])
            fig_c.add_trace(go.Bar(x=dc["ano"], y=dc["taxa_evasao_ef"],
                name="Evasao EF (%)", marker_color=COR_EF, opacity=0.85), secondary_y=False)
            fig_c.add_trace(go.Bar(x=dc["ano"], y=dc["taxa_evasao_em"],
                name="Evasao EM (%)", marker_color=COR_EM, opacity=0.85), secondary_y=False)
            fig_c.add_trace(go.Scatter(x=dc["ano"], y=dc["razao"],
                name="Razao EM / EF", mode="lines+markers",
                line=dict(color=COR_PRIMARIA, width=2, dash="dot"),
                marker=dict(size=7)), secondary_y=True)
            fig_c.update_yaxes(title_text="Evasao (%)", secondary_y=False)
            fig_c.update_yaxes(title_text="Razao EM / EF", secondary_y=True)
            fig_c.update_layout(barmode="group", hovermode="x unified",
                                height=360, legend=dict(orientation="h", y=-0.22))
            st.plotly_chart(fig_c, use_container_width=True)

            mr = round(dc["razao"].mean(), 1)
            bloco_insight("ruim", f"Sim: o EM tem em media {mr}x mais evasao que o EF",
                "Isso ocorre por razoes estruturais: maior pressao para trabalhar, currículo percebido como distante "
                "da realidade, e menor sensação de obrigatoriedade. Durante crises, o EM e sempre o primeiro afetado. "
                "Acao: priorize o 1 ano do EM, onde a transicao do EF gera o maior risco de abandono.")

    st.divider()

    # ── Boxplot ────────────────────────────────────────────────────────────────
    secao("Cada periodo historico foi melhor ou pior?",
          "A caixa mostra onde estao a maioria dos valores. Pontos isolados = anos excepcionais (como a pandemia).")

    st.info(
        "**Como ler este grafico sem estatistica:** imagine os valores do periodo empilhados em ordem. "
        "A caixa e o bloco do meio — 50% dos registros ficam dentro dela. "
        "A linha no centro da caixa e o valor mais tipico. "
        "Pontos acima da caixa = anos com valores muito acima do normal. No caso da pandemia, isso e esperado."
    )

    col1, col2 = st.columns(2)
    for col_ui, base, col_v, titulo_g in [
        (col1, dados["fato_socioeconomico"], "taxa_evasao_em",   "Evasao EM por periodo (%)"),
        (col2, dados["fato_educacional"],    "taxa_abandono_em", "Abandono EM por periodo (%)"),
    ]:
        with col_ui:
            s = base.dropna(subset=[col_v])
            s = s[(s["ano"] >= a1) & (s["ano"] <= a2)]
            if s.empty: continue
            ordem = ["2006–2010", "2011–2015", "2016–2019", "2020–2022 (Pandemia)", "2023–2024"]
            ok = [p for p in ordem if p in s["periodo"].unique()]
            fig_b = px.box(s, x="periodo", y=col_v, category_orders={"periodo": ok},
                color="periodo", color_discrete_map=PALETA_PERIODO,
                labels={"periodo": "Periodo", col_v: "%"}, points="all", title=titulo_g)
            fig_b.update_layout(showlegend=False, height=380)
            st.plotly_chart(fig_b, use_container_width=True)

    bloco_insight("info", "O que comparar",
        "O periodo 2016–2019 (verde) representa o melhor desempenho pre-pandemia — use-o como meta. "
        "O periodo 2020–2022 (amarelo) concentra os valores mais altos — sao outliers causados pela pandemia, nao erros. "
        "Acao: o objetivo atual e voltar ao nivel de 2016–2019 e ir alem.")

    st.warning(
        "Valores extremos nos graficos acima nao sao erros de dados. "
        "Eles refletem o impacto real da pandemia — um evento sem precedentes que nao pode ser ignorado na analise."
    )

    transicao_pagina(
        "Para identificar quais anos e registros concentram maior risco, acesse 'Identificacao de Risco'. "
        "Para entender as causas desses padroes, acesse 'Por que ocorre?'."
    )


# ===========================================================================
# PÁGINA 3 — IDENTIFICAÇÃO DE RISCO: "Onde e quem está em risco?"
# ===========================================================================
def pagina_risco(dados: dict, filtros: dict, insights: dict):
    st.markdown("# Identificacao de Risco — Onde e quem esta em risco?")
    st.caption(
        "Use esta pagina para priorizar: quais anos e registros escolares concentram maior risco. "
        "O Score de Risco combina tres indicadores em um unico numero (0 a 100)."
    )

    a1, a2 = filtros["ano_range"]
    df_educ = dados["fato_educacional"].copy()
    df_educ = df_educ[(df_educ["ano"] >= a1) & (df_educ["ano"] <= a2)]
    df_educ["score_risco"] = calcular_score(df_educ)
    df_educ["nivel_risco"] = classificar_risco(df_educ["score_risco"])

    total = len(df_educ)
    n_alto = df_educ["nivel_risco"].isin(["Alto", "Critico"]).sum()
    n_suficiente = total >= 3

    if not n_suficiente:
        st.warning("Dados insuficientes para conclusoes robustas. Amplie o periodo de analise.")

    # Resumo automático
    if n_alto == 0:
        st.success(f"Nenhum registro em nivel Alto ou Critico no periodo {a1}–{a2}.")
    elif n_alto / total >= 0.5:
        st.error(
            f"{n_alto} de {total} registros ({n_alto/total*100:.0f}%) estao em nivel Alto ou Critico. "
            "Mais da metade dos registros do periodo exigem atencao."
        )
    else:
        st.warning(
            f"{n_alto} de {total} registros ({n_alto/total*100:.0f}%) estao em nivel Alto ou Critico."
        )

    st.divider()

    # ── Distribuição + evolução score ──────────────────────────────────────────
    col1, col2 = st.columns([1, 2])

    with col1:
        secao("Como os registros se distribuem por nivel de risco?")
        contagem = df_educ["nivel_risco"].value_counts().reindex(
            ["Critico", "Alto", "Moderado", "Baixo"], fill_value=0)
        fig_pie = go.Figure(go.Pie(
            labels=contagem.index.tolist(), values=contagem.values.tolist(),
            marker_colors=[CORES_RISCO["Critico"], CORES_RISCO["Alto"], CORES_RISCO["Moderado"], CORES_RISCO["Baixo"]],
            hole=0.5, textinfo="label+percent+value", textfont=dict(size=11),
        ))
        fig_pie.update_layout(height=300, showlegend=False,
            annotations=[dict(text=f"{total}", x=0.5, y=0.5,
                             font=dict(size=14, color="#1E3A5F"), showarrow=False)])
        st.plotly_chart(fig_pie, use_container_width=True)
        perc_crit = contagem.get("Critico", 0) + contagem.get("Alto", 0)
        bloco_insight(
            "ruim" if perc_crit / total >= 0.3 else "info",
            "O que isso significa",
            f"{perc_crit} registros em nivel Alto ou Critico. "
            "Cada registro representa um conjunto de dados de um determinado ano. "
            "Foco de acao: os registros mais vermelhos."
        )

    with col2:
        secao("O score de risco subiu ou caiu?",
              "Linha = media anual. Area sombreada = variacao entre minimo e maximo do ano.")
        score_ano = df_educ.groupby("ano")["score_risco"].agg(["mean", "max", "min"]).reset_index()
        score_ano.columns = ["ano", "medio", "maximo", "minimo"]

        if not score_ano.empty:
            fig_sc = go.Figure()
            for y0, y1, cor_f in [(0,20,"#DCFCE7"),(20,35,"#FEF9C3"),(35,50,"#FEE2E2"),(50,100,"#FECACA")]:
                fig_sc.add_hrect(y0=y0, y1=y1, fillcolor=cor_f, opacity=0.2, line_width=0)
            fig_sc.add_trace(go.Scatter(
                x=pd.concat([score_ano["ano"], score_ano["ano"][::-1]]),
                y=pd.concat([score_ano["maximo"], score_ano["minimo"][::-1]]),
                fill="toself", fillcolor=hex_rgba(COR_EM, 0.1),
                line=dict(color="rgba(0,0,0,0)"), name="Variacao"))
            fig_sc.add_trace(go.Scatter(
                x=score_ano["ano"], y=score_ano["medio"],
                mode="lines+markers", name="Score medio",
                line=dict(color=COR_PRIMARIA, width=3),
                marker=dict(size=9, color=score_ano["medio"],
                            colorscale="RdYlGn_r", cmin=0, cmax=60,
                            line=dict(color="white", width=2)),
                text=score_ano["medio"].round(0).astype(int), textposition="top center",
            ))
            pandemia_vrect(fig_sc, a1, a2)
            fig_sc.update_layout(yaxis_title="Score (0–100)", xaxis_title="Ano",
                                 hovermode="x unified", height=300,
                                 legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig_sc, use_container_width=True)

            pior = score_ano.loc[score_ano["medio"].idxmax()]
            melhor = score_ano.loc[score_ano["medio"].idxmin()]
            bloco_insight("info", "Leitura rapida",
                f"Pior: {int(pior['ano'])} (score medio de {pior['medio']:.0f}). "
                f"Melhor: {int(melhor['ano'])} (score medio de {melhor['medio']:.0f}). "
                "A area sombreada mostra a dispersao — quanto mais larga, maior a diferenca entre os extremos.")

    st.divider()

    # ── Ranking ────────────────────────────────────────────────────────────────
    secao("Quais registros tem maior score de risco?",
          "Ordenado do maior para o menor risco. Use o filtro 'Nivel de risco' na barra lateral.")

    niveis_sel = filtros.get("risco_filtro", ["Alto", "Critico"])
    df_rank = df_educ[df_educ["nivel_risco"].isin(niveis_sel)].copy() if niveis_sel else df_educ.copy()
    df_rank = df_rank.sort_values("score_risco", ascending=False).reset_index(drop=True)
    df_rank.index += 1

    rename = {
        "ano": "Ano", "nivel_risco": "Nivel", "score_risco": "Score (0–100)",
        "taxa_abandono_em": "Abandono EM (%)", "tdi_em": "TDI EM (%)",
        "taxa_reprovacao_em": "Reprovacao EM (%)", "atu_em": "Alunos/Turma EM",
    }
    cols_show = [c for c in rename if c in df_rank.columns]
    df_display = df_rank[cols_show].rename(columns=rename)

    if df_display.empty:
        st.warning("Nenhum registro para os niveis selecionados. Altere o filtro 'Nivel de risco' na barra lateral.")
    else:
        st.dataframe(df_display, use_container_width=True, height=380,
            column_config={"Score (0–100)": st.column_config.ProgressColumn(
                "Score (0–100)", min_value=0, max_value=100, format="%.0f")})
        st.caption(
            "Score = Abandono EM (40%) + TDI — Distorcao Idade-Serie (30%) + Reprovacao EM (30%). "
            "Quanto maior, maior o risco. Foque nos registros com Score acima de 35."
        )

    st.divider()

    # ── Mapa de calor ──────────────────────────────────────────────────────────
    secao("Em quais combinacoes de ano e nivel de risco se concentram os problemas?",
          "Cada celula = numero de registros. Quanto mais escuro, maior a concentracao.")

    pivot = df_educ.pivot_table(index="nivel_risco", columns="ano",
        values="score_risco", aggfunc="count", fill_value=0
    ).reindex(["Critico", "Alto", "Moderado", "Baixo"])

    if not pivot.empty:
        fig_h = px.imshow(pivot, color_continuous_scale=["#DCFCE7", "#FEF9C3", "#FEE2E2", "#991B1B"],
            text_auto=True, labels={"color": "Registros"})
        fig_h.update_layout(height=280, xaxis_title="Ano", yaxis_title="Nivel de Risco")
        st.plotly_chart(fig_h, use_container_width=True)
        bloco_insight("info", "Como usar este mapa",
            "Colunas escuras = anos de maior risco. Linhas vermelhas = registros criticos. "
            "Identifique colunas com muitas celulas vermelhas — esses sao os anos prioritarios para revisao de politicas.")

    st.divider()

    # ── Placeholder modelo preditivo ──────────────────────────────────────────
    st.markdown("### [Em Desenvolvimento] — Modelo Preditivo de Risco")
    st.info(
        "**O que sera adicionado aqui:**\n\n"
        "Esta secao esta sendo preparada para receber um modelo preditivo que vai identificar, "
        "com base nos dados historicos:\n\n"
        "- Probabilidade estimada de evasao nos proximos anos\n"
        "- Grupos de alunos com maior vulnerabilidade\n"
        "- Impacto simulado de intervencoes especificas\n\n"
        "**Por enquanto**, o Score de Risco calculado manualmente (acima) serve como aproximacao. "
        "Registros com Score acima de 35 sao os candidatos prioritarios para intervencao."
    )

    with st.expander("Ver simulacao simples de risco (baseada nos ultimos dados disponiveis)"):
        df_sim = df_educ.sort_values("ano").groupby("ano")[["score_risco"]].mean().reset_index()
        df_sim = df_sim.dropna()
        if len(df_sim) >= 3:
            z = np.polyfit(df_sim["ano"], df_sim["score_risco"], 1)
            anos_proj = [df_sim["ano"].max() + 1, df_sim["ano"].max() + 2]
            proj = [round(np.poly1d(z)(a), 1) for a in anos_proj]
            df_proj = pd.DataFrame({"Ano": anos_proj, "Score projetado (simulacao linear)": proj,
                                    "Nivel estimado": [classificar_risco(pd.Series([p])).iloc[0] for p in proj]})
            st.dataframe(df_proj, use_container_width=True, hide_index=True)
            st.caption(
                "AVISO: esta e uma projecao simplificada baseada apenas na tendencia linear historica. "
                "Nao considera fatores externos, politicas ou mudancas estruturais. "
                "Use apenas como indicativo, nao como previsao."
            )

    transicao_pagina(
        "Para entender por que esses riscos existem, acesse 'Por que ocorre?'. "
        "Para saber o que fazer com essa informacao, acesse 'Plano de Acao'."
    )


# ===========================================================================
# PÁGINA 4 — POR QUE OCORRE: "Quais sao as causas?"
# ===========================================================================
def pagina_causas(dados: dict, filtros: dict, insights: dict):
    st.markdown("# Por que ocorre? — Quais sao as causas da evasao?")
    st.caption(
        "Entender as causas e fundamental para escolher as intervencoes certas. "
        "Esta pagina mostra quais fatores estao mais associados a evasao — com dados."
    )

    a1, a2 = filtros["ano_range"]
    df_int   = dados["fato_integrado"].copy()
    df_int   = df_int[(df_int["ano"] >= a1) & (df_int["ano"] <= a2)]
    df_socio = dados["dim_socio_anual"].copy()
    df_socio = df_socio[(df_socio["ano"] >= a1) & (df_socio["ano"] <= a2)]

    # ── Cadeia causal ──────────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:#F8FAFC;border:1px solid #CBD5E1;padding:16px 20px;border-radius:6px;margin-bottom:16px">
    <p style="color:#1E293B;font-weight:600;font-size:0.9rem;margin:0 0 10px 0">
    A evasao nao acontece de repente — ela segue uma cadeia de causas previsivel:
    </p>
    <div style="display:flex;align-items:center;flex-wrap:wrap;gap:6px;font-size:0.85rem;margin-bottom:12px">
    <div style="background:#FEE2E2;color:#991B1B;padding:6px 12px;border-radius:4px;font-weight:600;text-align:center">
    Reprovacao<br><small style="font-weight:400">aluno fica retido</small></div>
    <span style="color:#94A3B8;font-size:1.2rem">&#8594;</span>
    <div style="background:#FEF9C3;color:#92400E;padding:6px 12px;border-radius:4px;font-weight:600;text-align:center">
    TDI: Defasagem<br><small style="font-weight:400">aluno mais velho que a turma</small></div>
    <span style="color:#94A3B8;font-size:1.2rem">&#8594;</span>
    <div style="background:#FFEDD5;color:#9A3412;padding:6px 12px;border-radius:4px;font-weight:600;text-align:center">
    Desmotivacao<br><small style="font-weight:400">saida no ano letivo</small></div>
    <span style="color:#94A3B8;font-size:1.2rem">&#8594;</span>
    <div style="background:#FEE2E2;color:#991B1B;padding:6px 12px;border-radius:4px;font-weight:600;text-align:center">
    Evasao<br><small style="font-weight:400">saida definitiva</small></div>
    </div>
    <p style="color:#475569;font-size:0.85rem;margin:0">
    <b>Interromper essa cadeia em qualquer etapa reduz a evasao.</b>
    No Ensino Medio, fatores externos agravam o problema: pressao para trabalhar, currículo distante da realidade
    e menor percepcao de obrigatoriedade — especialmente em periodos de crise economica como a pandemia.
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Correlações ────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        secao("A reprovacao causa mais evasao?",
              "Cada ponto = 1 ano. Linha tracejada = tendencia. Pontos mais a direita e mais acima = pior.")
        xc, yc = "taxa_repetencia_em", "taxa_evasao_em"
        if xc in df_socio.columns and yc in df_socio.columns:
            s = df_socio.dropna(subset=[xc, yc])
            if len(s) >= 3:
                fig = go.Figure()
                scatter_tendencia(fig, s[xc].values, s[yc].values, "Ensino Medio", COR_EM, s["ano"].values)
                fig.add_hline(y=s[yc].mean(), line_dash="dot", line_color=COR_CINZA, opacity=0.5,
                              annotation_text=f"Media: {s[yc].mean():.1f}%", annotation_position="right")
                fig.add_vline(x=s[xc].mean(), line_dash="dot", line_color=COR_CINZA, opacity=0.5,
                              annotation_text=f"Media: {s[xc].mean():.1f}%", annotation_position="top")
                fig.update_layout(xaxis_title="Reprovacao EM (%)", yaxis_title="Evasao EM (%)",
                                  showlegend=False, height=340)
                st.plotly_chart(fig, use_container_width=True)
                r = np.corrcoef(s[xc].values, s[yc].values)[0, 1]
                forca = "muito forte" if abs(r) > 0.8 else ("forte" if abs(r) > 0.6 else "moderada")
                bloco_insight(
                    "ruim" if abs(r) > 0.5 else "info",
                    f"Sim: relacao {forca} ({r:.2f})",
                    "A linha tracejada sobe da esquerda para a direita: mais reprovacao = mais evasao. "
                    "Isso acontece porque a reprovacao gera defasagem, que gera desmotivacao, "
                    "que leva ao abandono. A relacao se intensificou com a pandemia. "
                    "Acao: reducao da reprovacao e a intervencao mais eficaz para reduzir a evasao."
                )
            else:
                st.warning("Dados insuficientes para esta analise. Amplie o periodo.")

    with col2:
        secao("A defasagem escolar (TDI) leva ao abandono?",
              "Cada ponto = 1 ano. Quanto mais alto o TDI, maior tende a ser o abandono.")
        if all(c in df_int.columns for c in ["tdi_em", "taxa_abandono_em"]):
            s2 = df_int.dropna(subset=["tdi_em", "taxa_abandono_em"])
            if len(s2) >= 3:
                fig2 = go.Figure()
                scatter_tendencia(fig2, s2["tdi_em"].values, s2["taxa_abandono_em"].values,
                                  "Ensino Medio", COR_ABANDONO, s2["ano"].values)
                fig2.update_layout(xaxis_title="TDI — Defasagem Escolar EM (%)",
                                   yaxis_title="Abandono EM (%)", showlegend=False, height=340)
                st.plotly_chart(fig2, use_container_width=True)
                r2 = np.corrcoef(s2["tdi_em"].values, s2["taxa_abandono_em"].values)[0, 1]
                forca2 = "muito forte" if abs(r2) > 0.8 else ("forte" if abs(r2) > 0.6 else "moderada")
                bloco_insight(
                    "ruim" if abs(r2) > 0.5 else "info",
                    f"Sim: relacao {forca2} ({r2:.2f})",
                    "TDI mede quantos alunos estao cursando uma serie abaixo do esperado para sua idade. "
                    "Aluno mais velho que os colegas sente-se deslocado e desiste mais. "
                    "Aulas de nivelamento reduzem o TDI e, consequentemente, o abandono. "
                    "Acao: identificar alunos com mais de 2 anos de defasagem e oferecer reforco imediato."
                )
            else:
                st.warning("Dados insuficientes. Amplie o periodo.")

    st.divider()

    # ── Diagnóstico fator a fator ──────────────────────────────────────────────
    secao("Quais indicadores estao criticos agora?",
          f"Diagnostico do ultimo ano do periodo selecionado vs. referencias nacionais do INEP.")

    if not df_int.empty:
        ultimo = df_int.sort_values("ano").iloc[-1]
        ano_ref = int(ultimo["ano"])

        fatores = []
        def av(col, label, lim_a, lim_c, ref):
            v = float(ultimo.get(col, np.nan) or np.nan)
            if pd.isna(v): return
            nv = "Critico" if v >= lim_c else ("Atencao" if v >= lim_a else "OK")
            fatores.append(dict(label=label, valor=v, nivel=nv, ref=ref))

        av("taxa_evasao_em",     "Evasao EM",      5,  10, "Aceitavel: ate 5%")
        av("taxa_abandono_em",   "Abandono EM",     5,  10, "Aceitavel: ate 5%")
        av("tdi_em",             "TDI EM",         20,  30, "Aceitavel: ate 20%")
        av("taxa_repetencia_em", "Reprovacao EM",   8,  15, "Aceitavel: ate 8%")
        av("taxa_evasao_ef",     "Evasao EF",       3,   6, "Aceitavel: ate 3%")

        st.caption(f"Ano de referencia: {ano_ref}")
        crit = [f for f in fatores if f["nivel"] == "Critico"]
        atenc = [f for f in fatores if f["nivel"] == "Atencao"]
        ok_f  = [f for f in fatores if f["nivel"] == "OK"]

        if crit:
            st.error("**Indicadores criticos** (exigem acao imediata):\n" +
                     "\n".join(f"- **{f['label']}**: {f['valor']:.1f}% ({f['ref']})" for f in crit))
        if atenc:
            st.warning("**Indicadores em atencao** (monitorar de perto):\n" +
                       "\n".join(f"- **{f['label']}**: {f['valor']:.1f}% ({f['ref']})" for f in atenc))
        if ok_f:
            st.success("**Indicadores dentro do limite aceitavel:**\n" +
                       "\n".join(f"- **{f['label']}**: {f['valor']:.1f}%" for f in ok_f))

    st.divider()

    # ── Matriz de correlação ──────────────────────────────────────────────────
    secao("Quando um indicador piora, quais outros pioram junto?",
          "Verde = relacao inversa (um sobe, o outro cai). Vermelho = relacao direta (ambos sobem juntos).")

    st.info(
        "**Como ler sem estatistica:** olhe apenas as cores. Vermelho escuro = dois problemas andam juntos. "
        "Verde escuro = um indicador positivo esta associado a menor evasao. "
        "Os numeros nao sao essenciais — as cores ja comunicam o suficiente."
    )

    cols_c = [c for c in [
        "taxa_evasao_em", "taxa_abandono_em", "taxa_repetencia_em", "taxa_reprovacao_em",
        "tdi_em", "taxa_aprovacao_em", "taxa_promocao_em", "taxa_evasao_ef", "taxa_abandono_ef", "tdi_ef",
    ] if c in df_int.columns]

    nomes = {
        "taxa_evasao_em": "Evasao EM", "taxa_abandono_em": "Abandono EM",
        "taxa_repetencia_em": "Reprovacao EM", "taxa_reprovacao_em": "Reprovacao EM (b)",
        "tdi_em": "TDI EM", "taxa_aprovacao_em": "Aprovacao EM",
        "taxa_promocao_em": "Promocao EM", "taxa_evasao_ef": "Evasao EF",
        "taxa_abandono_ef": "Abandono EF", "tdi_ef": "TDI EF",
    }

    df_c = df_int[[c for c in cols_c if c in df_int.columns]].dropna(how="all").rename(columns=nomes)
    if len(df_c) >= 3:
        corr = df_c.corr()
        col_c1, col_c2 = st.columns([3, 2])
        with col_c1:
            fig_corr = px.imshow(corr, color_continuous_scale="RdYlGn", zmin=-1, zmax=1,
                                 text_auto=".2f", aspect="auto")
            fig_corr.update_layout(height=420)
            st.plotly_chart(fig_corr, use_container_width=True)
        with col_c2:
            st.markdown("**Principais achados:**")
            achados = []
            for par_a, par_b in [("Aprovacao EM","Evasao EM"), ("TDI EM","Evasao EM"),
                                  ("Reprovacao EM","Evasao EM"), ("Abandono EM","Evasao EM")]:
                if par_a in corr.columns and par_b in corr.columns:
                    rv = round(corr.loc[par_a, par_b], 2)
                    if abs(rv) > 0.4:
                        direcao = "mais aprovacao = menos evasao" if rv < 0 else "mais um = mais o outro"
                        achados.append(f"**{par_a} x {par_b}:** {rv} ({direcao})")
            if achados:
                for a in achados: st.markdown(f"- {a}")
            else:
                st.caption("Amplíe o período para calcular correlações significativas.")
            st.markdown("")
            bloco_insight("bom", "Conclusao principal",
                "Aprovacao e Promocao andam em sentido oposto a evasao — "
                "mais alunos aprovados significa menos evasao. "
                "Acao: qualquer politica que aumente a aprovacao reduz a evasao.")
        st.warning(
            f"A matriz e calculada com {len(df_c)} pontos de dados. "
            "Com menos de 10, os resultados indicam tendencias, mas nao sao estatisticamente definitivos."
        )
    else:
        st.warning("Dados insuficientes para calcular a matriz de correlacao. Amplie o periodo de analise.")

    transicao_pagina(
        "Agora que voce conhece as causas, acesse 'Plano de Acao' para ver as intervencoes recomendadas."
    )


# ===========================================================================
# PÁGINA 5 — PLANO DE AÇÃO: "O que fazer?"
# ===========================================================================
def pagina_acoes(dados: dict, filtros: dict, insights: dict):
    st.markdown("# Plano de Acao — O que fazer para reduzir a evasao?")
    st.caption(
        "Acoes priorizadas por urgencia, baseadas nos fatores de risco identificados nos dados. "
        "Cada acao esta conectada a um indicador especifico."
    )

    a1, a2 = filtros["ano_range"]
    df_int = dados["fato_integrado"].copy()
    df_int = df_int[(df_int["ano"] >= a1) & (df_int["ano"] <= a2)]

    if df_int.empty:
        st.warning("Sem dados para o periodo selecionado.")
        return

    ultimo = df_int.sort_values("ano").iloc[-1]
    def sv(c): return float(ultimo.get(c, 0) or 0)
    sc    = sv("indice_risco_evasao")
    ev    = sv("taxa_evasao_em")
    ab    = sv("taxa_abandono_em")
    tdi   = sv("tdi_em")
    rep   = sv("taxa_repetencia_em")
    atu   = sv("atu_em")
    nivel = classificar_risco(pd.Series([sc])).iloc[0]

    # Diagnóstico atual
    st.info(
        f"**Situacao atual (ano {int(ultimo['ano'])}):** "
        f"Evasao EM = {ev:.1f}% | Abandono EM = {ab:.1f}% | TDI EM = {tdi:.1f}% | "
        f"Reprovacao EM = {rep:.1f}% | Score de Risco = {sc:.0f}/100 ({nivel})"
    )

    st.divider()

    URGENCIAS = {
        "IMEDIATA":    ("#FEF2F2", "#991B1B"),
        "CURTO PRAZO": ("#FFFBEB", "#B45309"),
        "MEDIO PRAZO": ("#EFF6FF", "#1D4ED8"),
        "LONGO PRAZO": ("#F0FDF4", "#15803D"),
    }

    acoes = [
        {
            "urgencia": "IMEDIATA",
            "titulo": "Monitorar frequencia semanalmente",
            "problema": f"Abandono EM: {ab:.1f}% (limite: 5%)",
            "acao": "Alunos com mais de 25% de faltas estao em risco iminente de abandono. "
                    "Implante controle de frequencia semanal e ative a familia quando o limite for atingido.",
            "resultado": "Reduz o abandono em ate 30% (referencia INEP).",
        },
        {
            "urgencia": "IMEDIATA",
            "titulo": "Reforco escolar para alunos com defasagem (TDI)",
            "problema": f"TDI EM: {tdi:.1f}% (limite: 20%)",
            "acao": "Alunos cursando serie abaixo da esperada para sua idade tem chance muito maior de desistir. "
                    "Aulas de nivelamento em contraturno reduzem essa defasagem.",
            "resultado": "Queda no TDI → queda no abandono → queda na evasao.",
        },
        {
            "urgencia": "CURTO PRAZO",
            "titulo": "Revisar politica de reprovacao",
            "problema": f"Reprovacao EM: {rep:.1f}% (limite: 8%)",
            "acao": "A reprovacao e o principal fator de risco de evasao nos dados. "
                    "Substitua a reprovacao por progressao continuada com suporte pedagogico intensivo. "
                    "Isso nao significa aprovar sem criterio — significa dar suporte antes de reprovar.",
            "resultado": "Interrompe a cadeia reprovacao → TDI → abandono → evasao.",
        },
        {
            "urgencia": "CURTO PRAZO",
            "titulo": "Reduzir numero de alunos por turma no EM",
            "problema": f"ATU EM atual: {atu:.0f} alunos/turma (meta: ate 30)",
            "acao": "Turmas superlotadas dificultam o acompanhamento individual. "
                    "Meta: no maximo 30 alunos por turma no EM.",
            "resultado": "Maior vinculo professor-aluno e reducao do abandono.",
        },
        {
            "urgencia": "MEDIO PRAZO",
            "titulo": "Ampliar EJA e Ensino Medio noturno",
            "problema": "Evasao acumulada cria uma populacao fora do sistema.",
            "acao": "Ofereça modalidades flexiveis (EJA, EM noturno) para quem precisou trabalhar e abandonou. "
                    "Facilite a reintegracao com horarios compativeis.",
            "resultado": "Reduz a evasao permanente e amplia a cobertura educacional.",
        },
        {
            "urgencia": "MEDIO PRAZO",
            "titulo": "Apoio socioemocional e reducao de barreiras externas",
            "problema": "Parte da evasao tem causas externas (pobreza, violencia, distancia).",
            "acao": "Bolsas condicionadas a frequencia, auxilio-transporte e apoio psicologico "
                    "reduzem o impacto de fatores externos — especialmente em populacoes vulneraveis.",
            "resultado": "Reduz a evasao causada por fatores socioeconomicos.",
        },
        {
            "urgencia": "LONGO PRAZO",
            "titulo": "Sistema de monitoramento continuo por escola",
            "problema": "Identificacao tardia aumenta o custo de intervencao.",
            "acao": "Dashboard atualizado mensalmente com dados de frequencia, desempenho e perfil por escola. "
                    "Permite agir antes que o problema se agrave.",
            "resultado": "Prevencao e mais barata e mais eficaz que remediacâo.",
        },
        {
            "urgencia": "LONGO PRAZO",
            "titulo": "Analise geografica da evasao por bairro",
            "problema": "Recursos publicos podem ser mal alocados sem mapeamento geografico.",
            "acao": "Cruzar dados educacionais com o IBGE por bairro e regiao (RPA) de Recife "
                    "para identificar concentracoes de risco e direcionar recursos com mais precisao.",
            "resultado": "Alocacao mais eficiente dos investimentos publicos em educacao.",
        },
    ]

    for ac in acoes:
        bg, borda = URGENCIAS[ac["urgencia"]]
        st.markdown(
            f"""<div style="background:{bg};border-left:5px solid {borda};
            padding:14px 18px;border-radius:4px;margin-bottom:12px;">
            <span style="background:{borda};color:white;padding:2px 9px;border-radius:3px;
            font-size:0.72rem;font-weight:700;letter-spacing:0.06em">{ac['urgencia']}</span>
            <b style="display:block;color:#1E293B;font-size:0.95rem;margin:6px 0 4px 0">{ac['titulo']}</b>
            <p style="color:#374151;font-size:0.87rem;margin:0 0 6px 0;line-height:1.55">{ac['acao']}</p>
            <p style="color:#6B7280;font-size:0.8rem;margin:0">
            <b>Indicador:</b> {ac['problema']} &nbsp;|&nbsp; <b>Resultado esperado:</b> {ac['resultado']}
            </p></div>""",
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("#### Resumo consolidado")
    df_res = pd.DataFrame([
        {"Urgencia": a["urgencia"], "Acao": a["titulo"], "Indicador de alerta": a["problema"]}
        for a in acoes
    ])
    st.dataframe(df_res, use_container_width=True, hide_index=True)

    transicao_pagina(
        "Voce percorreu todo o fluxo do painel: problema, evolucao, risco, causas e acoes. "
        "Use as paginas anteriores para apresentar o contexto antes de propor as acoes desta pagina."
    )


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    garantir_dados()
    dados = carregar_dados()

    if not dados:
        st.error("Dados nao encontrados. Execute o ETL primeiro.")
        st.stop()

    # Insights globais calculados uma vez
    insights = computar_insights(dados)

    filtros = sidebar(dados)

    PAGINAS = {
        "Visao Geral — Qual e o problema?":                pagina_visao_geral,
        "Evolucao Historica — Como o problema mudou?":     pagina_evolucao,
        "Identificacao de Risco — Quem esta em risco?":    pagina_risco,
        "Por que ocorre? — Quais sao as causas?":          pagina_causas,
        "Plano de Acao — O que fazer?":                    pagina_acoes,
    }

    st.sidebar.divider()
    pagina_atual = st.sidebar.radio("Navegacao", list(PAGINAS.keys()))
    PAGINAS[pagina_atual](dados, filtros, insights)

    st.sidebar.divider()
    st.sidebar.caption("Analise de Evasao Escolar | Recife | INEP/MEC")


if __name__ == "__main__":
    main()
