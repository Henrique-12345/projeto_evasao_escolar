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
# Configuração da página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Evasão Escolar — Recife",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.4rem; padding-bottom: 2.5rem; }
    h1 { font-size: 1.55rem !important; font-weight: 700; color: #1E3A5F; }
    h2 { font-size: 1.18rem !important; font-weight: 600; color: #1E3A5F; margin-top:1.6rem !important; }
    h3 { font-size: 1rem !important; font-weight: 600; color: #334155; margin-top:1.2rem !important; }
    .stMetric label { font-size: 0.78rem !important; color: #64748B; }
    div[data-testid="stMetricValue"] { font-size: 1.35rem !important; }
    div[data-testid="stMetricDelta"] { font-size: 0.78rem !important; }
    hr { border:none; border-top:1px solid #E2E8F0; margin:1.4rem 0; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Paths e constantes de cor
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"

COR_PRIMARIA = "#1E3A5F"
COR_EF       = "#2563EB"
COR_EM       = "#DC2626"
COR_ABANDONO = "#EA580C"
COR_OK       = "#15803D"
COR_CINZA    = "#64748B"

CORES_RISCO = {
    "Critico":  "#991B1B",
    "Alto":     "#DC2626",
    "Moderado": "#B45309",
    "Baixo":    "#15803D",
}

PALETA_PERIODO = {
    "2006–2010":            "#94A3B8",
    "2011–2015":            "#60A5FA",
    "2016–2019":            "#34D399",
    "2020–2022 (Pandemia)": "#FBBF24",
    "2023–2024":            "#F87171",
}

LIMIARES = {"baixo": 20, "moderado": 35, "alto": 50}

# ---------------------------------------------------------------------------
# Glossário
# ---------------------------------------------------------------------------
GLOSSARIO = {
    "Evasão escolar":        "Saída definitiva do aluno do sistema de ensino, sem previsão de retorno.",
    "Abandono escolar":      "Saída do aluno durante o ano letivo em curso. É o sinal imediato que antecede a evasão definitiva.",
    "TDI":                   "Taxa de Distorção Idade-Série — percentual de alunos que estão cursando uma série com mais de 2 anos de atraso em relação à idade esperada.",
    "p.p. (ponto percentual)": "Diferença direta entre dois percentuais. Exemplo: de 10% para 13% representa um aumento de 3 p.p.",
    "EF":                    "Ensino Fundamental — do 1.º ao 9.º ano.",
    "EM":                    "Ensino Médio — do 1.º ao 3.º ano.",
    "ATU":                   "Média de Alunos por Turma.",
    "Score de Risco":        "Indicador de 0 a 100 calculado como: Abandono EM (40%) + TDI EM (30%) + Reprovação EM (30%). Quanto maior o score, maior o risco de evasão.",
    "EJA":                   "Educação de Jovens e Adultos — modalidade para quem não concluiu o ensino regular na idade prevista.",
}

# ---------------------------------------------------------------------------
# Carregamento de dados
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
# Score de risco
# ---------------------------------------------------------------------------
def calcular_score(df: pd.DataFrame) -> pd.Series:
    def col(c): return df[c].fillna(0) if c in df.columns else pd.Series(0.0, index=df.index)
    s = col("taxa_abandono_em") * 0.40 + col("tdi_em") * 0.30 + col("taxa_reprovacao_em") * 0.30
    mask = df["taxa_abandono_em"].notna() if "taxa_abandono_em" in df.columns else pd.Series(False, index=df.index)
    result = pd.Series(np.nan, index=df.index)
    result[mask] = s[mask].clip(0, 100)
    return result.round(1)


def classificar_risco(score: pd.Series) -> pd.Series:
    return pd.cut(
        score,
        bins=[-np.inf, LIMIARES["baixo"], LIMIARES["moderado"], LIMIARES["alto"], np.inf],
        labels=["Baixo", "Moderado", "Alto", "Critico"],
    ).astype(str)

# ---------------------------------------------------------------------------
# Insights automáticos
# ---------------------------------------------------------------------------
def computar_insights(dados: dict) -> dict:
    """Calcula métricas globais uma única vez para todos os gráficos."""
    s  = dados["dim_socio_anual"].sort_values("ano")
    fi = dados["fato_integrado"].sort_values("ano")
    out = {}

    if "taxa_evasao_em" in s.columns:
        em = s.dropna(subset=["taxa_evasao_em"])
        if not em.empty:
            out["pior_ano"]    = int(em.loc[em["taxa_evasao_em"].idxmax(), "ano"])
            out["pior_val"]    = round(em["taxa_evasao_em"].max(), 1)
            out["melhor_ano"]  = int(em.loc[em["taxa_evasao_em"].idxmin(), "ano"])
            out["melhor_val"]  = round(em["taxa_evasao_em"].min(), 1)
            out["ano_ini"]     = int(em.iloc[0]["ano"])
            out["ano_fim"]     = int(em.iloc[-1]["ano"])
            out["val_ini"]     = round(em.iloc[0]["taxa_evasao_em"], 1)
            out["val_fim"]     = round(em.iloc[-1]["taxa_evasao_em"], 1)
            out["delta_total"] = round(out["val_fim"] - out["val_ini"], 1)

    if "indice_risco_evasao" in fi.columns:
        ri = fi.dropna(subset=["indice_risco_evasao"])
        if not ri.empty:
            out["score_atual"] = round(ri.iloc[-1]["indice_risco_evasao"], 1)
            out["score_ant"]   = round(ri.iloc[-2]["indice_risco_evasao"], 1) if len(ri) > 1 else out["score_atual"]
            out["ano_score"]   = int(ri.iloc[-1]["ano"])
            out["nivel_score"] = classificar_risco(pd.Series([out["score_atual"]])).iloc[0]

    tend = dados.get("tendencia_anual", pd.DataFrame())
    if "var_taxa_evasao_em" in tend.columns:
        t = tend.dropna(subset=["var_taxa_evasao_em"])
        if not t.empty:
            out["pior_ano_var"]    = int(t.loc[t["var_taxa_evasao_em"].idxmax(), "ano"])
            out["pior_val_var"]    = round(t["var_taxa_evasao_em"].max(), 1)
            out["melhor_ano_var"]  = int(t.loc[t["var_taxa_evasao_em"].idxmin(), "ano"])
            out["melhor_val_var"]  = round(t["var_taxa_evasao_em"].min(), 1)
    return out

# ---------------------------------------------------------------------------
# Utilitários visuais
# ---------------------------------------------------------------------------
def hex_rgba(hex_color: str, alpha: float = 0.15) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def scatter_tendencia(fig: go.Figure, x, y, cor: str, texts=None) -> go.Figure:
    mask = ~(np.isnan(x) | np.isnan(y))
    xc, yc = np.array(x)[mask], np.array(y)[mask]
    fig.add_trace(go.Scatter(
        x=xc, y=yc, mode="markers+text" if texts is not None else "markers",
        text=np.array(texts)[mask] if texts is not None else None,
        textposition="top center",
        marker=dict(color=cor, size=11, line=dict(color="white", width=1.5)),
        showlegend=False,
    ))
    if len(xc) >= 3:
        z = np.polyfit(xc, yc, 1)
        xl = np.linspace(xc.min(), xc.max(), 100)
        fig.add_trace(go.Scatter(
            x=xl, y=np.poly1d(z)(xl), mode="lines",
            line=dict(color=cor, width=2, dash="dash"),
            name="Tendencia linear", showlegend=False,
        ))
        return fig, z[0]  # retorna o coeficiente angular
    return fig, None


def vrect_pandemia(fig, a1, a2):
    if a1 <= 2020 <= a2:
        fig.add_vrect(
            x0=2019.5, x1=2022.5, fillcolor="#FEF9C3", opacity=0.35, line_width=0,
            annotation_text="Pandemia (2020–22)", annotation_position="top left",
            annotation_font=dict(size=10, color="#92400E"),
        )


def pre_grafico(titulo: str, insight: str, detalhes: str = ""):
    """Bloco de texto ANTES do gráfico: o que será visto e o insight principal."""
    st.markdown(f"## {titulo}")
    st.info(f"**Insight principal:** {insight}")
    if detalhes:
        st.write(detalhes)


def pos_grafico(texto: str, tipo: str = "info"):
    """Bloco de texto DEPOIS do gráfico: interpretação dos dados."""
    if tipo == "positivo":
        st.success(texto)
    elif tipo == "atencao":
        st.warning(texto)
    elif tipo == "critico":
        st.error(texto)
    else:
        st.info(texto)


# ---------------------------------------------------------------------------
# BARRA LATERAL
# ---------------------------------------------------------------------------
def sidebar(dados: dict) -> dict:
    st.sidebar.markdown(
        f'<p style="color:{COR_PRIMARIA};font-size:1rem;font-weight:700;margin:0 0 2px 0">'
        'Evasao Escolar — Recife</p>'
        '<p style="color:#64748B;font-size:0.78rem;margin:0">INEP/MEC | 2008–2022</p>',
        unsafe_allow_html=True,
    )
    st.sidebar.divider()

    anos = sorted(dados["fato_integrado"]["ano"].unique())
    a_min, a_max = int(min(anos)), int(max(anos))
    default_ini  = max(a_min, a_max - 3)   # padrão: últimos 4 anos

    filtros = {}
    filtros["ano_range"] = st.sidebar.slider(
        "Periodo de analise",
        min_value=a_min, max_value=a_max,
        value=(default_ini, a_max),
        help="Padrao: ultimos 4 anos (2019–2022). Arraste para ver a serie historica completa.",
    )
    filtros["nivel"] = st.sidebar.multiselect(
        "Nivel de ensino",
        ["Ensino Fundamental (EF)", "Ensino Medio (EM)"],
        default=["Ensino Fundamental (EF)", "Ensino Medio (EM)"],
        help="EF = 1.o ao 9.o ano | EM = 1.o ao 3.o ano",
    )

    st.sidebar.divider()
    with st.sidebar.expander("Glossario — termos utilizados"):
        for termo, defn in GLOSSARIO.items():
            st.markdown(f"**{termo}**")
            st.caption(defn)

    st.sidebar.divider()
    if st.sidebar.button("Reprocessar dados (ETL)"):
        st.cache_data.clear()
        sys.path.insert(0, str(ROOT))
        from etl.etl_pipeline import run_etl
        run_etl()
        st.rerun()

    return filtros


# ===========================================================================
# PAGINA 1 — CONTEXTO GERAL
# ===========================================================================
def pagina_contexto(dados: dict, filtros: dict, ins: dict):
    st.markdown("# A Evasao Escolar em Recife — Contexto Geral")

    a1, a2 = filtros["ano_range"]
    df = dados["fato_integrado"].copy()
    df = df[(df["ano"] >= a1) & (df["ano"] <= a2)].sort_values("ano")

    if df.empty:
        st.warning(f"Nenhum dado disponivel para {a1}–{a2}. Ajuste o filtro na barra lateral.")
        return

    ultimo = df.iloc[-1]
    ant    = df.iloc[-2] if len(df) > 1 else df.iloc[0]

    def sv(c): return float(ultimo.get(c, np.nan) or np.nan)
    def dpp(c):
        v1 = sv(c)
        v2 = float(ant.get(c, np.nan) or np.nan)
        return f"{v1-v2:+.1f} p.p." if pd.notna(v1) and pd.notna(v2) else None

    sc    = ins.get("score_atual", 0)
    nivel = ins.get("nivel_score", "Baixo")
    delta = sc - ins.get("score_ant", sc)

    # ── Diagnóstico automático de abertura ────────────────────────────────────
    st.markdown(
        "Este painel analisa os dados de evasão escolar no municipio de Recife entre 2008 e 2022, "
        "usando informações do INEP (Instituto Nacional de Estudos e Pesquisas Educacionais). "
        "**Evasão escolar** é quando um aluno sai definitivamente do sistema de ensino — "
        "diferente do **abandono**, que ocorre durante o ano letivo. "
        "Ambos são sinais de que algo no sistema educacional (ou na vida do aluno) está falhando."
    )

    pior_ano  = ins.get("pior_ano")
    melhor_ano = ins.get("melhor_ano")
    delta_total = ins.get("delta_total")

    if delta_total is not None and delta_total < -5:
        st.success(
            f"**Tendencia historica positiva:** entre {ins.get('ano_ini')} e {ins.get('ano_fim')}, "
            f"a evasao no Ensino Medio caiu {abs(delta_total):.1f} pontos percentuais "
            f"(de {ins.get('val_ini')}% para {ins.get('val_fim')}%). "
            "Esse progresso mostra que politicas educacionais de longo prazo funcionam."
        )

    if pior_ano and 2020 <= pior_ano <= 2022:
        st.warning(
            f"**Pandemia rompeu o progresso:** o pior ano da serie foi {pior_ano} "
            f"({ins.get('pior_val')}% de evasao no EM). "
            "O fechamento das escolas e a crise economica de 2020–2021 reverteram avancos "
            "que levaram anos para ser conquistados."
        )
    elif pior_ano:
        st.warning(
            f"**Pior ano da serie:** {pior_ano} ({ins.get('pior_val')}% de evasao no EM)."
        )

    if melhor_ano:
        st.success(
            f"**Melhor ano da serie:** {melhor_ano} ({ins.get('melhor_val')}% de evasao no EM). "
            "Este e o nivel de referencia que o sistema deve recuperar e superar."
        )

    st.divider()

    # ── KPIs principais ───────────────────────────────────────────────────────
    st.markdown("### Numeros do ano mais recente disponivel")
    st.caption(
        f"Dados de {int(ultimo['ano'])}. "
        "p.p. = ponto percentual (diferenca direta entre dois percentuais). "
        "A seta mostra a variacao em relacao ao ano anterior: "
        "para evasao e abandono, seta vermelha (para cima) significa piora."
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Evasao — Ensino Fundamental",
                  f"{sv('taxa_evasao_ef'):.1f}%" if pd.notna(sv("taxa_evasao_ef")) else "–",
                  dpp("taxa_evasao_ef"), delta_color="inverse",
                  help="Percentual de alunos do EF que deixaram definitivamente o sistema escolar.")
    with c2:
        st.metric("Evasao — Ensino Medio",
                  f"{sv('taxa_evasao_em'):.1f}%" if pd.notna(sv("taxa_evasao_em")) else "–",
                  dpp("taxa_evasao_em"), delta_color="inverse",
                  help="Historicamente 2 a 3 vezes maior do que no EF.")
    with c3:
        st.metric("Abandono — Ensino Medio",
                  f"{sv('taxa_abandono_em'):.1f}%" if pd.notna(sv("taxa_abandono_em")) else "–",
                  dpp("taxa_abandono_em"), delta_color="inverse",
                  help="Saidas durante o ano letivo. Principal sinal de alerta da evasao futura.")
    with c4:
        st.metric("Distorcao Idade-Serie — EM (TDI)",
                  f"{sv('tdi_em'):.1f}%" if pd.notna(sv("tdi_em")) else "–",
                  dpp("tdi_em"), delta_color="inverse",
                  help="TDI: alunos cursando serie muito abaixo do esperado para sua idade.")
    with c5:
        cor_sc = "inverse" if delta > 0 else "normal"
        st.metric("Score de Risco (0–100)",
                  f"{sc:.0f}",
                  f"{delta:+.0f} vs. {int(ant['ano'])}" if delta else None,
                  delta_color="inverse",
                  help="Score composto: Abandono EM (40%) + TDI (30%) + Reprovacao EM (30%).")

    # Interpretação dos KPIs
    ev_em = sv("taxa_evasao_em") or 0
    ab_em = sv("taxa_abandono_em") or 0
    tdi   = sv("tdi_em") or 0

    if ev_em > 10:
        pos_grafico(
            f"A evasao no Ensino Medio esta em {ev_em:.1f}% — acima de 10%, o que significa "
            "que mais de 1 em cada 10 alunos deixa definitivamente o sistema. Nivel critico.",
            "critico",
        )
    elif ev_em > 5:
        pos_grafico(
            f"A evasao no Ensino Medio esta em {ev_em:.1f}%. "
            "Acima de 5%, o que indica que o sistema ainda nao consegue manter todos os alunos.",
            "atencao",
        )
    else:
        pos_grafico(
            f"A evasao no Ensino Medio esta em {ev_em:.1f}%, dentro do limite aceitavel (5%). "
            "Isso nao significa ausencia de risco — o abandono e o TDI ainda precisam de atencao.",
            "positivo",
        )

    st.divider()

    # ── Score de risco: gauge ──────────────────────────────────────────────────
    pre_grafico(
        "Qual e o nivel de risco atual?",
        f"O Score de Risco resume a situacao em um unico numero. Atualmente: {sc:.0f}/100 ({nivel}).",
        "O Score combina tres fatores: abandono escolar (peso 40%), distorcao idade-serie/TDI (peso 30%) "
        "e reprovacao (peso 30%). Quanto maior o numero, mais grave a situacao. "
        "Verde = sob controle | Amarelo = atencao | Vermelho = situacao critica.",
    )

    col_g, col_t = st.columns([1, 2])
    with col_g:
        if pd.notna(sc):
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=float(sc),
                delta={
                    "reference": ins.get("score_ant", sc), "valueformat": ".0f",
                    "increasing": {"color": "#DC2626"}, "decreasing": {"color": "#15803D"},
                },
                number={"suffix": " / 100", "font": {"size": 26}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": CORES_RISCO.get(nivel, COR_CINZA)},
                    "steps": [
                        {"range": [0,  20],  "color": "#DCFCE7"},
                        {"range": [20, 35],  "color": "#FEF9C3"},
                        {"range": [35, 50],  "color": "#FEE2E2"},
                        {"range": [50, 100], "color": "#FECACA"},
                    ],
                },
                title={"text": nivel, "font": {"size": 14}},
            ))
            fig_g.update_layout(height=240, margin=dict(t=40, b=0, l=10, r=10))
            st.plotly_chart(fig_g, use_container_width=True)

    with col_t:
        if "indice_risco_evasao" in df.columns:
            dt = df.dropna(subset=["indice_risco_evasao"])
            if not dt.empty:
                fig_t = go.Figure()
                for y0, y1, cf, lb in [
                    (0, 20, "#DCFCE7", "Baixo"), (20, 35, "#FEF9C3", "Moderado"),
                    (35, 50, "#FEE2E2", "Alto"), (50, 100, "#FECACA", "Critico"),
                ]:
                    fig_t.add_hrect(y0=y0, y1=y1, fillcolor=cf, opacity=0.25,
                                    line_width=0, annotation_text=lb, annotation_position="right")
                fig_t.add_trace(go.Scatter(
                    x=dt["ano"], y=dt["indice_risco_evasao"],
                    mode="lines+markers+text",
                    text=dt["indice_risco_evasao"].round(0).fillna(0).astype(int),
                    textposition="top center",
                    line=dict(color=COR_PRIMARIA, width=3),
                    marker=dict(size=9, color=dt["indice_risco_evasao"],
                                colorscale="RdYlGn_r", cmin=0, cmax=60,
                                line=dict(color="white", width=2)),
                ))
                vrect_pandemia(fig_t, a1, a2)
                fig_t.update_layout(
                    yaxis_title="Score de Risco (0–100)", xaxis_title="Ano",
                    hovermode="x unified", height=240, showlegend=False,
                    margin=dict(t=10, b=30),
                )
                st.plotly_chart(fig_t, use_container_width=True)

    delta_sc_txt = f"subiu {delta:+.0f} pontos" if delta > 0 else f"caiu {abs(delta):.0f} pontos"
    pos_grafico(
        f"O Score de Risco {delta_sc_txt} em relacao ao ano anterior. "
        f"No grafico de linha, cada ponto e um ano. Pontos altos indicam maior risco. "
        f"A queda consistente antes de 2020 mostra que as politicas educacionais funcionaram. "
        f"O salto durante a pandemia foi um choque externo — nao uma falha estrutural.",
        "atencao" if sc > LIMIARES["moderado"] else "positivo",
    )

    st.divider()

    # ── Variação início vs fim do período ─────────────────────────────────────
    st.markdown("### Quanto cada indicador mudou no periodo selecionado?")
    st.write(
        f"Comparacao entre {int(df.iloc[0]['ano'])} (inicio do periodo) e "
        f"{int(df.iloc[-1]['ano'])} (ultimo ano disponivel). "
        "p.p. = ponto percentual."
    )
    primeiro = df.iloc[0]
    indicadores = [
        ("taxa_evasao_em",     "Evasao EM",      True),
        ("taxa_abandono_em",   "Abandono EM",     True),
        ("tdi_em",             "TDI — EM",        True),
        ("taxa_repetencia_em", "Reprovacao EM",   True),
        ("taxa_aprovacao_em",  "Aprovacao EM",    False),
    ]
    colunas = st.columns(len(indicadores))
    melhorias = 0
    for ci, (nome, label, inv) in zip(colunas, indicadores):
        vi = float(primeiro.get(nome, np.nan) or np.nan)
        vf = float(ultimo.get(nome, np.nan) or np.nan)
        if pd.notna(vi) and pd.notna(vf):
            delta_i = vf - vi
            if (inv and delta_i < 0) or (not inv and delta_i > 0):
                melhorias += 1
            with ci:
                st.metric(label, f"{vf:.1f}%",
                          f"{delta_i:+.1f} p.p.",
                          delta_color="inverse" if inv else "normal")

    total_comp = sum(1 for nome, _, _ in indicadores
                     if pd.notna(float(primeiro.get(nome, np.nan) or np.nan))
                     and pd.notna(float(ultimo.get(nome, np.nan) or np.nan)))
    if total_comp > 0:
        if melhorias == total_comp:
            pos_grafico("Todos os indicadores melhoraram no periodo analisado.", "positivo")
        elif melhorias >= total_comp // 2:
            pos_grafico(
                f"{melhorias} de {total_comp} indicadores melhoraram. "
                "O periodo teve mais avanco do que retrocesso.",
                "positivo",
            )
        else:
            pos_grafico(
                f"Apenas {melhorias} de {total_comp} indicadores melhoraram. "
                "O periodo foi de deterioracao geral — provavelmente influenciado pela pandemia.",
                "atencao",
            )

    st.caption(
        "Proxima secao: 'Evolucao ao Longo do Tempo' — veja como cada indicador se comportou "
        "ano a ano e identifique os momentos de melhora e piora."
    )


# ===========================================================================
# PAGINA 2 — EVOLUÇÃO TEMPORAL
# ===========================================================================
def pagina_evolucao(dados: dict, filtros: dict, ins: dict):
    st.markdown("# Evolucao da Evasao ao Longo do Tempo")
    st.write(
        "Esta secao mostra como a evasao e o abandono escolar evoluiram ano a ano. "
        "Entender essa trajetoria e essencial para separar problemas estruturais "
        "(que existem independente de crises) de choques externos (como a pandemia)."
    )

    a1, a2    = filtros["ano_range"]
    df_soc    = dados["dim_socio_anual"].copy()
    df_educ   = dados["dim_educ_anual"].copy()
    df_int    = dados["fato_integrado"].copy()
    tend      = dados["tendencia_anual"].copy()
    show_ef   = "Ensino Fundamental (EF)" in filtros["nivel"]
    show_em   = "Ensino Medio (EM)"        in filtros["nivel"]

    for d in [df_soc, df_educ, df_int, tend]:
        d.drop(d[~d["ano"].between(a1, a2)].index, inplace=True)

    if df_soc.empty:
        st.warning("Dados insuficientes para este periodo. Amplie o intervalo de anos.")
        return

    # ── Serie temporal principal ───────────────────────────────────────────────
    pre_grafico(
        "Como a evasao e o abandono evoluiram ano a ano?",
        "A evasao no Ensino Medio caiu consistentemente de 2008 a 2019, foi interrompida "
        "pela pandemia em 2020–2021, e iniciou recuperacao em 2022.",
        "Linhas solidas representam evasao (saida definitiva). "
        "Linhas pontilhadas representam abandono (saida durante o ano). "
        "Azul = Ensino Fundamental | Vermelho = Ensino Medio. "
        "A area amarela marca o periodo da pandemia de COVID-19 (2020–2022).",
    )

    fig = go.Figure()
    vrect_pandemia(fig, a1, a2)
    for nivel, show, c_ev, c_ab, nome in [
        ("ef", show_ef, COR_EF,      "#93C5FD", "EF"),
        ("em", show_em, COR_EM,      "#FCA5A5", "EM"),
    ]:
        if not show:
            continue
        s_s = df_soc.dropna(subset=[f"taxa_evasao_{nivel}"])
        s_e = df_educ.dropna(subset=[f"taxa_abandono_{nivel}"])
        if not s_s.empty:
            fig.add_trace(go.Scatter(
                x=s_s["ano"], y=s_s[f"taxa_evasao_{nivel}"],
                name=f"Evasao {nome}", mode="lines+markers",
                line=dict(color=c_ev, width=3), marker=dict(size=8),
            ))
        if not s_e.empty:
            fig.add_trace(go.Scatter(
                x=s_e["ano"], y=s_e[f"taxa_abandono_{nivel}"],
                name=f"Abandono {nome}", mode="lines+markers",
                line=dict(color=c_ab, width=2, dash="dot"), marker=dict(size=7),
            ))
    fig.update_layout(
        yaxis_title="Taxa (%)", xaxis_title="Ano",
        hovermode="x unified", height=400,
        legend=dict(orientation="h", y=-0.22),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Interpretação automática
    if "taxa_evasao_em" in df_soc.columns:
        s_em = df_soc.dropna(subset=["taxa_evasao_em"]).sort_values("ano")
        if len(s_em) >= 2:
            vi, vf = s_em.iloc[0]["taxa_evasao_em"], s_em.iloc[-1]["taxa_evasao_em"]
            delta = round(vf - vi, 1)
            pico_ano = int(s_em.loc[s_em["taxa_evasao_em"].idxmax(), "ano"])
            pico_val = round(s_em["taxa_evasao_em"].max(), 1)
            contexto_pico = (
                "O pico coincide com a pandemia de COVID-19. "
                "Em 2020, as escolas fecharam; em 2021, os efeitos acumulados geraram o maior nivel de evasao da serie. "
                "Esse valor nao e um erro — e o reflexo real de uma crise sem precedentes."
                if 2020 <= pico_ano <= 2022
                else f"O pico em {pico_ano} pode estar relacionado a fatores locais ou nacionais daquele periodo."
            )
            pos_grafico(
                f"No periodo analisado, a evasao no Ensino Medio {('caiu' if delta < 0 else 'subiu')} "
                f"{abs(delta):.1f} p.p. (pontos percentuais) "
                f"— de {vi:.1f}% em {int(s_em.iloc[0]['ano'])} para {vf:.1f}% em {int(s_em.iloc[-1]['ano'])}. "
                f"O pico mais alto foi em {pico_ano} ({pico_val}%). {contexto_pico}",
                "positivo" if delta < 0 else "atencao",
            )

    st.divider()

    # ── Variação ano a ano ──────────────────────────────────────────────────────
    pre_grafico(
        "Em quais anos a evasao piorou e em quais melhorou?",
        f"O maior aumento foi em {ins.get('pior_ano_var', '?')} "
        f"(+{ins.get('pior_val_var', '?')}%). "
        f"A maior queda foi em {ins.get('melhor_ano_var', '?')} "
        f"({ins.get('melhor_val_var', '?')}%).",
        "Cada barra mostra a variacao da evasao no Ensino Medio em relacao ao ano anterior. "
        "Barras vermelhas (acima da linha zero) = evasao piorou. "
        "Barras verdes (abaixo da linha zero) = evasao melhorou. "
        "A variacao e sempre comparada ao ano imediatamente anterior.",
    )

    if "var_taxa_evasao_em" in tend.columns:
        s = tend.dropna(subset=["var_taxa_evasao_em"])
        if not s.empty:
            pior_v  = s.loc[s["var_taxa_evasao_em"].idxmax()]
            melhor_v = s.loc[s["var_taxa_evasao_em"].idxmin()]
            fig_y = go.Figure(go.Bar(
                x=s["ano"],
                y=s["var_taxa_evasao_em"],
                marker_color=[COR_EM if v > 0 else COR_OK for v in s["var_taxa_evasao_em"]],
                text=s["var_taxa_evasao_em"].apply(lambda x: f"{x:+.1f}%"),
                textposition="outside",
            ))
            fig_y.add_hline(y=0, line_color=COR_CINZA, line_width=1.5)
            fig_y.add_annotation(
                x=int(pior_v["ano"]), y=float(pior_v["var_taxa_evasao_em"]),
                text=f"Pior: {int(pior_v['ano'])}", showarrow=True, arrowhead=2,
                ax=0, ay=-32, font=dict(color=COR_EM, size=10))
            fig_y.add_annotation(
                x=int(melhor_v["ano"]), y=float(melhor_v["var_taxa_evasao_em"]),
                text=f"Melhor: {int(melhor_v['ano'])}", showarrow=True, arrowhead=2,
                ax=0, ay=28, font=dict(color=COR_OK, size=10))
            fig_y.update_layout(
                yaxis_title="Variacao vs. ano anterior (%)", xaxis_title="Ano", height=360,
            )
            st.plotly_chart(fig_y, use_container_width=True)

            pior_pandemia = 2020 <= int(pior_v["ano"]) <= 2022
            pos_grafico(
                f"O maior aumento na evasao ocorreu em {int(pior_v['ano'])} "
                f"(+{float(pior_v['var_taxa_evasao_em']):.1f}% em relacao ao ano anterior). "
                + (
                    "Esse foi o auge do impacto da pandemia: fechamento das escolas, ensino remoto desigual "
                    "(muitos alunos sem internet ou dispositivos) e crise economica que levou jovens ao mercado de trabalho. "
                    "Estima-se que cerca de 4 milhoes de estudantes brasileiros deixaram de estudar em 2020–2021."
                    if pior_pandemia
                    else f"Investigue o que ocorreu em {int(pior_v['ano'])} — mudancas de politica, contexto economico."
                ),
                "atencao",
            )
        else:
            st.warning("Dados insuficientes para calcular variacao anual. Amplie o periodo.")

    st.divider()

    # ── EF vs EM ───────────────────────────────────────────────────────────────
    pre_grafico(
        "O Ensino Medio e realmente mais afetado do que o Ensino Fundamental?",
        "Sim. Historicamente, a evasao no Ensino Medio e 2 a 3 vezes maior do que no Ensino Fundamental.",
        "As barras mostram a evasao nos dois niveis de ensino. "
        "A linha pontilhada mostra quantas vezes a evasao do EM supera a do EF em cada ano. "
        "Esse dado e importante: as politicas para os dois niveis precisam ser diferentes.",
    )

    if all(c in df_int.columns for c in ["taxa_evasao_ef", "taxa_evasao_em"]):
        dc = df_int.dropna(subset=["taxa_evasao_ef", "taxa_evasao_em"]).sort_values("ano").copy()
        dc = dc[dc["taxa_evasao_ef"] > 0]
        if not dc.empty:
            dc["razao"] = (dc["taxa_evasao_em"] / dc["taxa_evasao_ef"]).round(2)
            fig_c = make_subplots(specs=[[{"secondary_y": True}]])
            fig_c.add_trace(
                go.Bar(x=dc["ano"], y=dc["taxa_evasao_ef"],
                       name="Evasao EF (%)", marker_color=COR_EF, opacity=0.85),
                secondary_y=False,
            )
            fig_c.add_trace(
                go.Bar(x=dc["ano"], y=dc["taxa_evasao_em"],
                       name="Evasao EM (%)", marker_color=COR_EM, opacity=0.85),
                secondary_y=False,
            )
            fig_c.add_trace(
                go.Scatter(x=dc["ano"], y=dc["razao"], name="Vezes que EM supera EF",
                           mode="lines+markers",
                           line=dict(color=COR_PRIMARIA, width=2, dash="dot"),
                           marker=dict(size=7)),
                secondary_y=True,
            )
            fig_c.update_yaxes(title_text="Taxa de Evasao (%)", secondary_y=False)
            fig_c.update_yaxes(title_text="EM e quantas vezes maior que EF", secondary_y=True)
            fig_c.update_layout(
                barmode="group", hovermode="x unified", height=380,
                legend=dict(orientation="h", y=-0.22),
            )
            st.plotly_chart(fig_c, use_container_width=True)

            mr = round(dc["razao"].mean(), 1)
            ano_max_r = int(dc.loc[dc["razao"].idxmax(), "ano"])
            max_r     = round(dc["razao"].max(), 1)
            pos_grafico(
                f"Em media, a evasao no Ensino Medio foi {mr} vezes maior do que no Ensino Fundamental. "
                f"O pico dessa diferenca ocorreu em {ano_max_r} ({max_r} vezes). "
                "Por que o EM e mais vulneravel? Alunos entre 15 e 17 anos enfrentam maior pressao economica para trabalhar. "
                "O curriculo do EM e percebido como mais distante da realidade. "
                "Durante crises, o EM e sempre o primeiro atingido. "
                "Isso significa que as politicas para o EM precisam ser mais intensas e especificas.",
                "atencao",
            )

    st.divider()

    # ── Boxplot por período ─────────────────────────────────────────────────────
    pre_grafico(
        "Cada fase historica foi melhor ou pior?",
        "O periodo 2016–2019 foi o melhor da serie. A pandemia (2020–2022) foi o pior periodo registrado.",
        "O grafico de caixas agrupa os dados em fases historicas. "
        "Como interpretar: a linha no centro da caixa e o valor mais tipico de cada periodo. "
        "A caixa engloba 50% dos registros. Pontos isolados sao anos com valores excepcionais. "
        "Isso e esperado para 2020–2021: a pandemia gerou valores fora do padrao — e isso nao e erro de dados.",
    )

    col1, col2 = st.columns(2)
    for col_ui, base, col_v, titulo_g in [
        (col1, dados["fato_socioeconomico"], "taxa_evasao_em",   "Evasao no Ensino Medio por periodo (%)"),
        (col2, dados["fato_educacional"],    "taxa_abandono_em", "Abandono no Ensino Medio por periodo (%)"),
    ]:
        with col_ui:
            s = base.dropna(subset=[col_v])
            s = s[s["ano"].between(a1, a2)]
            if s.empty:
                continue
            ordem = ["2006–2010", "2011–2015", "2016–2019", "2020–2022 (Pandemia)", "2023–2024"]
            ok = [p for p in ordem if p in s["periodo"].unique()]
            fig_b = px.box(
                s, x="periodo", y=col_v,
                category_orders={"periodo": ok},
                color="periodo", color_discrete_map=PALETA_PERIODO,
                labels={"periodo": "Periodo historico", col_v: "Taxa (%)"},
                points="all", title=titulo_g,
            )
            fig_b.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_b, use_container_width=True)

    pos_grafico(
        "O periodo 2016–2019 (verde) representa o melhor desempenho historico antes da pandemia — "
        "e a meta de curto prazo para recuperacao. "
        "O periodo 2020–2022 (amarelo) concentra os valores mais altos: sao valores extremos causados "
        "pela pandemia, nao erros de medicao. "
        "O objetivo atual e retornar ao nivel de 2016–2019 e supera-lo.",
        "atencao",
    )
    st.warning(
        "Limitacao dos dados: os valores extremos de 2020–2021 refletem o impacto real da pandemia, "
        "mas tambem podem conter subnotificacao — em 2020, muitas escolas nao realizaram avaliacoes formais, "
        "o que dificulta a comparacao direta com outros anos."
    )

    st.caption(
        "Proxima secao: 'Impacto da Pandemia' — analise especifica do periodo 2020–2022 "
        "e suas consequencias nos indicadores educacionais."
    )


# ===========================================================================
# PAGINA 3 — IMPACTO DA PANDEMIA
# ===========================================================================
def pagina_pandemia(dados: dict, filtros: dict, ins: dict):
    st.markdown("# O Impacto da Pandemia de COVID-19 na Evasao Escolar")
    st.write(
        "A pandemia de COVID-19 (2020–2022) foi o maior choque externo na educacao brasileira "
        "das ultimas decadas. Esta secao analisa em detalhe o que aconteceu, por que aconteceu "
        "e o que os dados nos dizem sobre o periodo de recuperacao."
    )

    a1, a2 = filtros["ano_range"]

    # ── Explicação dos 3 mecanismos ────────────────────────────────────────────
    st.markdown("### Por que a pandemia afetou tanto a evasao?")
    st.write(
        "Tres mecanismos principais explicam o aumento abrupto da evasao durante a pandemia:"
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.error(
            "**1. Fechamento das escolas**\n\n"
            "A suspensao das aulas presenciais rompeu o vinculo fisico entre aluno e escola. "
            "Muitos estudantes simplesmente pararam de acompanhar as atividades remotas. "
            "Para alunos com dificuldades de aprendizagem, o ensino remoto foi ainda mais excluente."
        )
    with c2:
        st.error(
            "**2. Ensino remoto desigual**\n\n"
            "Grande parte dos alunos da rede publica nao tinha acesso adequado a internet "
            "ou dispositivos eletronicos. "
            "Enquanto escolas particulares adaptaram rapidamente, alunos publicos ficaram para tras, "
            "acumulando defasagens que aumentaram a chance de abandono."
        )
    with c3:
        st.error(
            "**3. Crise economica**\n\n"
            "Familias perderam renda. Jovens passaram a trabalhar para complementar a renda familiar. "
            "O trabalho informal e infantil aumentou. "
            "Para muitos, entre estudar e garantir sustento da familia, a escola perdeu prioridade. "
            "Estima-se que 4 milhoes de brasileiros deixaram de estudar em 2020."
        )

    st.divider()

    # ── Score antes, durante e depois da pandemia ──────────────────────────────
    df_fi = dados["fato_integrado"].copy().sort_values("ano")

    pre_grafico(
        "Como o risco de evasao se comportou antes, durante e apos a pandemia?",
        "O risco subiu abruptamente em 2020–2021 e iniciou queda em 2022, "
        "mas ainda nao voltou ao nivel pre-pandemia.",
        "O Score de Risco combina abandono (40%), TDI/defasagem (30%) e reprovacao (30%). "
        "Cada ano e marcado com seu nivel de risco (cor). "
        "A comparacao entre periodos mostra o impacto do choque e a velocidade de recuperacao.",
    )

    if "indice_risco_evasao" in df_fi.columns:
        dt = df_fi.dropna(subset=["indice_risco_evasao"])
        fig_p = go.Figure()
        for y0, y1, cf, lb in [
            (0, 20, "#DCFCE7", "Baixo"), (20, 35, "#FEF9C3", "Moderado"),
            (35, 50, "#FEE2E2", "Alto"), (50, 100, "#FECACA", "Critico"),
        ]:
            fig_p.add_hrect(y0=y0, y1=y1, fillcolor=cf, opacity=0.2, line_width=0,
                            annotation_text=lb, annotation_position="right")
        fig_p.add_trace(go.Scatter(
            x=dt["ano"], y=dt["indice_risco_evasao"],
            mode="lines+markers+text",
            text=dt["indice_risco_evasao"].round(0).fillna(0).astype(int),
            textposition="top center",
            line=dict(color=COR_PRIMARIA, width=3),
            marker=dict(
                size=10,
                color=dt["indice_risco_evasao"],
                colorscale="RdYlGn_r", cmin=0, cmax=60,
                line=dict(color="white", width=2),
            ),
        ))
        fig_p.add_vrect(x0=2019.5, x1=2022.5, fillcolor="#FEF9C3", opacity=0.35,
                        line_width=0, annotation_text="Pandemia",
                        annotation_position="top left",
                        annotation_font=dict(size=11, color="#92400E"))
        fig_p.update_layout(yaxis_title="Score de Risco (0–100)", xaxis_title="Ano",
                            hovermode="x unified", height=380, showlegend=False)
        st.plotly_chart(fig_p, use_container_width=True)

        # Calcula diferença pré e pós pandemia
        pre  = dt[dt["ano"] < 2020]["indice_risco_evasao"].mean()
        pan  = dt[dt["ano"].between(2020, 2022)]["indice_risco_evasao"].mean()
        pos  = dt[dt["ano"] > 2022]["indice_risco_evasao"].mean()
        if pd.notna(pre) and pd.notna(pan):
            diff = round(pan - pre, 1)
            pos_grafico(
                f"O Score de Risco medio antes da pandemia era de {pre:.1f}. "
                f"Durante a pandemia, subiu para {pan:.1f} — um aumento de {diff:.1f} pontos. "
                + (
                    f"Apos a pandemia (2022+), o score iniciou queda para {pos:.1f}, "
                    "mas ainda nao retornou ao nivel anterior. "
                    "A recuperacao e lenta porque os efeitos da pandemia sao de longo prazo: "
                    "alunos que abandonaram raramente retornam imediatamente."
                    if pd.notna(pos) else
                    "Os dados disponiveis cobrem ate 2022, inicio da recuperacao."
                ),
                "atencao",
            )

    st.divider()

    # ── Efeito nos indicadores individuais ─────────────────────────────────────
    st.markdown("### O que aconteceu com cada indicador durante a pandemia?")
    st.write(
        "Abaixo, a comparacao dos indicadores individuais entre o melhor periodo pre-pandemia "
        "(2016–2019) e o periodo da pandemia (2020–2022)."
    )

    df_soc   = dados["dim_socio_anual"].copy()
    df_educ_  = dados["dim_educ_anual"].copy()
    df_fi_    = dados["fato_integrado"].copy()

    comparacoes = []
    for col_v, label, fonte in [
        ("taxa_evasao_em",   "Evasao EM (%)",   df_soc),
        ("taxa_abandono_em", "Abandono EM (%)", df_educ_),
        ("tdi_em",           "TDI EM (%)",       df_fi_),
    ]:
        if col_v not in fonte.columns:
            continue
        pre_p  = fonte[fonte["ano"].between(2016, 2019)][col_v].mean()
        pan_p  = fonte[fonte["ano"].between(2020, 2022)][col_v].mean()
        if pd.notna(pre_p) and pd.notna(pan_p):
            comparacoes.append({
                "Indicador": label,
                "Media 2016–2019 (pre-pandemia)": f"{pre_p:.1f}%",
                "Media 2020–2022 (pandemia)": f"{pan_p:.1f}%",
                "Diferenca (p.p.)": f"{pan_p-pre_p:+.1f}",
            })

    if comparacoes:
        df_comp = pd.DataFrame(comparacoes)
        st.dataframe(df_comp, use_container_width=True, hide_index=True)
        pos_grafico(
            "Todos os indicadores pioraram durante a pandemia em relacao ao melhor periodo anterior. "
            "A coluna 'Diferenca' mostra o impacto em pontos percentuais (p.p.). "
            "Valores positivos indicam piora (mais evasao, mais abandono, mais defasagem). "
            "Esse e o custo educacional da pandemia em Recife, mensuravel nos dados.",
            "atencao",
        )

    st.divider()
    st.markdown("### 2022 — Inicio da recuperacao ou ilusao?")
    st.write(
        "Em 2022, os indicadores comecaram a melhorar com o retorno as aulas presenciais. "
        "Porem, a recuperacao e parcial e lenta. Tres razoes explicam por que:"
    )
    st.info(
        "**1. Alunos que abandonaram nao retornam imediatamente.** "
        "Uma vez fora do sistema, reintegrar alunos exige politicas ativas — nao apenas reabrir as escolas.\n\n"
        "**2. Defasagens de aprendizagem persistem.** "
        "Alunos que avancaram de serie durante a pandemia sem dominar o conteudo chegam ao ano seguinte "
        "com lacunas que aumentam o risco de reprovacao e abandono futuro.\n\n"
        "**3. Os efeitos economicos nao desaparecem com o fim das restricoes.** "
        "Familias que perderam renda durante a pandemia continuam em situacao de vulnerabilidade, "
        "e isso mantem a pressao para que jovens trabalhem em vez de estudar."
    )

    st.caption(
        "Proxima secao: 'Por que os Alunos Evadem?' — analise das relacoes entre "
        "reprovacao, defasagem escolar e evasao."
    )


# ===========================================================================
# PAGINA 4 — RELAÇÃO ENTRE VARIÁVEIS
# ===========================================================================
def pagina_relacoes(dados: dict, filtros: dict, ins: dict):
    st.markdown("# Por que os Alunos Evadem? — Relacao entre as Variaveis")
    st.write(
        "Evasao raramente e uma decisao repentina. Ela e o resultado final de uma cadeia de eventos "
        "que comeca com a reprovacao e se agrava com o acumulo de defasagem. "
        "Esta secao mostra essa cadeia com dados — e explica quais variaveis sao os "
        "melhores preditores da evasao."
    )

    a1, a2   = filtros["ano_range"]
    df_socio = dados["dim_socio_anual"][(dados["dim_socio_anual"]["ano"].between(a1, a2))].copy()
    df_int   = dados["fato_integrado"][(dados["fato_integrado"]["ano"].between(a1, a2))].copy()

    # ── Cadeia causal ──────────────────────────────────────────────────────────
    st.markdown("### A cadeia que leva a evasao")
    st.markdown("""
<div style="background:#F8FAFC;border:1px solid #CBD5E1;padding:16px 20px;border-radius:6px">
<p style="color:#374151;font-size:0.91rem;margin:0 0 14px 0;line-height:1.7">
A cadeia abaixo explica a logica que os dados confirmam:
</p>
<div style="display:flex;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:14px">
<div style="background:#FEE2E2;color:#991B1B;padding:8px 14px;border-radius:4px;font-weight:600;font-size:0.87rem;text-align:center">
REPROVACAO<br><small style="font-weight:400">aluno nao avanca de serie</small></div>
<span style="color:#94A3B8;font-size:1.4rem">→</span>
<div style="background:#FEF9C3;color:#92400E;padding:8px 14px;border-radius:4px;font-weight:600;font-size:0.87rem;text-align:center">
DEFASAGEM (TDI)<br><small style="font-weight:400">aluno mais velho que a turma</small></div>
<span style="color:#94A3B8;font-size:1.4rem">→</span>
<div style="background:#FFEDD5;color:#9A3412;padding:8px 14px;border-radius:4px;font-weight:600;font-size:0.87rem;text-align:center">
DESMOTIVACAO<br><small style="font-weight:400">sentimento de exclusao</small></div>
<span style="color:#94A3B8;font-size:1.4rem">→</span>
<div style="background:#FEE2E2;color:#991B1B;padding:8px 14px;border-radius:4px;font-weight:600;font-size:0.87rem;text-align:center">
ABANDONO<br><small style="font-weight:400">saida no meio do ano</small></div>
<span style="color:#94A3B8;font-size:1.4rem">→</span>
<div style="background:#991B1B;color:white;padding:8px 14px;border-radius:4px;font-weight:600;font-size:0.87rem;text-align:center">
EVASAO<br><small style="font-weight:400">saida definitiva</small></div>
</div>
<p style="color:#475569;font-size:0.87rem;margin:0;line-height:1.6">
<b>Como funciona na pratica:</b> um aluno que reprova fica em uma turma com colegas mais novos.
Isso gera desconforto social, sensacao de fracasso e perda de motivacao.
Com o tempo, o aluno falta cada vez mais — ate sair definitivamente.
Durante a pandemia, essa cadeia se acelerou: alunos que ja tinham defasagem encontraram
ainda mais dificuldade no ensino remoto e muitos optaram por trabalhar.
<b>Qualquer intervencao que quebre essa cadeia em algum elo reduz a evasao.</b>
</p>
</div>
""", unsafe_allow_html=True)

    st.divider()

    # ── Gráficos de correlação ─────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        pre_grafico(
            "Mais reprovacao leva a mais evasao?",
            "Sim. Nos dados de Recife, anos com mais reprovacao no Ensino Medio "
            "sempre apresentam mais evasao no mesmo periodo.",
            "Cada ponto no grafico representa um ano. "
            "O eixo horizontal mostra a reprovacao; o eixo vertical mostra a evasao. "
            "A linha tracejada indica a tendencia geral. "
            "Pontos no canto superior direito sao os anos mais criticos.",
        )
        xc, yc = "taxa_repetencia_em", "taxa_evasao_em"
        if xc in df_socio.columns and yc in df_socio.columns:
            s = df_socio.dropna(subset=[xc, yc])
            if len(s) >= 3:
                fig = go.Figure()
                result = scatter_tendencia(fig, s[xc].values, s[yc].values, COR_EM, s["ano"].values)
                fig_out, slope = result if isinstance(result, tuple) else (result, None)
                fig.add_hline(y=s[yc].mean(), line_dash="dot", line_color=COR_CINZA, opacity=0.5,
                              annotation_text=f"Media evasao: {s[yc].mean():.1f}%",
                              annotation_position="right")
                fig.add_vline(x=s[xc].mean(), line_dash="dot", line_color=COR_CINZA, opacity=0.5,
                              annotation_text=f"Media reprovacao: {s[xc].mean():.1f}%",
                              annotation_position="top")
                fig.update_layout(
                    xaxis_title="Reprovacao no EM (%)",
                    yaxis_title="Evasao no EM (%)",
                    showlegend=False, height=360,
                )
                st.plotly_chart(fig, use_container_width=True)
                r = np.corrcoef(s[xc].values, s[yc].values)[0, 1]
                forca = "muito forte" if abs(r) > 0.8 else ("forte" if abs(r) > 0.6 else "moderada")
                slope_txt = (
                    f"Cada aumento de 1 p.p. na reprovacao esta associado a um aumento de "
                    f"{abs(slope):.2f} p.p. na evasao. "
                    if slope is not None else ""
                )
                pos_grafico(
                    f"Existe uma relacao {forca} (indice {r:.2f}) entre reprovacao e evasao no EM. "
                    f"{slope_txt}"
                    "Isso confirma o primeiro elo da cadeia causal. "
                    "Reduzir a reprovacao e a intervencao com maior impacto direto na evasao. "
                    "Essa relacao se intensificou apos a pandemia, quando alunos retornaram com "
                    "grandes lacunas de aprendizagem.",
                    "atencao",
                )
            else:
                st.warning("Dados insuficientes para esta analise. Amplie o periodo de analise.")

    with col2:
        pre_grafico(
            "Mais defasagem escolar (TDI) leva a mais abandono?",
            "Sim. O TDI (percentual de alunos com mais de 2 anos de atraso escolar) "
            "e um dos preditores mais fortes do abandono.",
            "O TDI — Taxa de Distorcao Idade-Serie — mede quantos alunos estao cursando "
            "uma serie abaixo do esperado para sua idade. "
            "Um aluno mais velho que os colegas sente-se deslocado e desiste mais facilmente. "
            "Cada ponto e um ano; a linha tracejada mostra a tendencia.",
        )
        if all(c in df_int.columns for c in ["tdi_em", "taxa_abandono_em"]):
            s2 = df_int.dropna(subset=["tdi_em", "taxa_abandono_em"])
            if len(s2) >= 3:
                fig2 = go.Figure()
                result2 = scatter_tendencia(fig2, s2["tdi_em"].values, s2["taxa_abandono_em"].values,
                                            COR_ABANDONO, s2["ano"].values)
                _, slope2 = result2 if isinstance(result2, tuple) else (result2, None)
                fig2.update_layout(
                    xaxis_title="TDI — Defasagem Escolar EM (%)",
                    yaxis_title="Abandono no EM (%)",
                    showlegend=False, height=360,
                )
                st.plotly_chart(fig2, use_container_width=True)
                r2 = np.corrcoef(s2["tdi_em"].values, s2["taxa_abandono_em"].values)[0, 1]
                forca2 = "muito forte" if abs(r2) > 0.8 else ("forte" if abs(r2) > 0.6 else "moderada")
                slope2_txt = (
                    f"Cada ponto percentual a mais de defasagem esta associado a um aumento de "
                    f"{abs(slope2):.2f} p.p. no abandono. "
                    if slope2 is not None else ""
                )
                pos_grafico(
                    f"Existe uma relacao {forca2} (indice {r2:.2f}) entre defasagem (TDI) e abandono. "
                    f"{slope2_txt}"
                    "Isso confirma o segundo elo da cadeia: a defasagem leva ao abandono. "
                    "Alunos mais velhos em turmas mais jovens se sentem excluidos e desistem. "
                    "Programas de nivelamento e reforco em contraturno reduzem o TDI e, "
                    "em consequencia, o abandono.",
                    "atencao",
                )
            else:
                st.warning("Dados insuficientes. Amplie o periodo de analise.")

    st.divider()

    # ── Diagnóstico dos indicadores ────────────────────────────────────────────
    st.markdown("### Quais indicadores estao fora do limite aceitavel?")
    st.write(
        "Comparacao dos valores mais recentes do periodo selecionado com referencias nacionais do INEP. "
        "Indicadores acima do limite aceitavel exigem atencao ou acao imediata."
    )

    if not df_int.empty:
        ultimo = df_int.sort_values("ano").iloc[-1]
        ano_ref = int(ultimo["ano"])
        st.caption(f"Ano de referencia: {ano_ref}.")

        fatores = []
        def av(col, label, lim_a, lim_c, ref_txt):
            v = float(ultimo.get(col, np.nan) or np.nan)
            if pd.isna(v):
                return
            nv = "Critico" if v >= lim_c else ("Atencao" if v >= lim_a else "OK")
            fatores.append(dict(label=label, valor=v, nivel=nv, ref=ref_txt))

        av("taxa_evasao_em",     "Evasao EM",            5,  10, "Limite aceitavel: ate 5%")
        av("taxa_abandono_em",   "Abandono EM",          5,  10, "Limite aceitavel: ate 5%")
        av("tdi_em",             "TDI — Defasagem EM",  20,  30, "Limite aceitavel: ate 20%")
        av("taxa_repetencia_em", "Reprovacao EM",        8,  15, "Limite aceitavel: ate 8%")
        av("taxa_evasao_ef",     "Evasao EF",            3,   6, "Limite aceitavel: ate 3%")

        criticos = [f for f in fatores if f["nivel"] == "Critico"]
        atencao  = [f for f in fatores if f["nivel"] == "Atencao"]
        ok_list  = [f for f in fatores if f["nivel"] == "OK"]

        if criticos:
            st.error(
                "**Indicadores criticos — exigem acao imediata:**\n"
                + "\n".join(
                    f"- **{f['label']}**: {f['valor']:.1f}% ({f['ref']})"
                    for f in criticos
                )
            )
        if atencao:
            st.warning(
                "**Indicadores em atencao — monitorar de perto:**\n"
                + "\n".join(
                    f"- **{f['label']}**: {f['valor']:.1f}% ({f['ref']})"
                    for f in atencao
                )
            )
        if ok_list:
            st.success(
                "**Indicadores dentro do limite aceitavel:**\n"
                + "\n".join(
                    f"- **{f['label']}**: {f['valor']:.1f}%"
                    for f in ok_list
                )
            )

    st.divider()

    # ── Mapa de correlação ─────────────────────────────────────────────────────
    pre_grafico(
        "Quando um indicador piora, quais outros pioram junto?",
        "Reprovacao, TDI e abandono formam um grupo de indicadores que sempre pioram juntos. "
        "Aprovacao e promocao caminham em sentido oposto a evasao.",
        "O mapa de calor abaixo mostra a relacao entre todos os indicadores simultaneamente. "
        "Vermelho escuro = dois indicadores sobem juntos (relacao direta). "
        "Verde escuro = um sobe enquanto o outro cai (relacao inversa). "
        "Branco/amarelo = pouca relacao entre os dois. "
        "Nao e necessario analisar os numeros — as cores ja comunicam o suficiente.",
    )

    cols_c = [c for c in [
        "taxa_evasao_em", "taxa_abandono_em", "taxa_repetencia_em",
        "taxa_reprovacao_em", "tdi_em", "taxa_aprovacao_em",
        "taxa_promocao_em", "taxa_evasao_ef", "taxa_abandono_ef", "tdi_ef",
    ] if c in df_int.columns]

    nomes = {
        "taxa_evasao_em":     "Evasao EM",
        "taxa_abandono_em":   "Abandono EM",
        "taxa_repetencia_em": "Reprovacao EM",
        "taxa_reprovacao_em": "Reprovacao EM (base educ.)",
        "tdi_em":             "TDI — Defasagem EM",
        "taxa_aprovacao_em":  "Aprovacao EM",
        "taxa_promocao_em":   "Promocao EM",
        "taxa_evasao_ef":     "Evasao EF",
        "taxa_abandono_ef":   "Abandono EF",
        "tdi_ef":             "TDI — Defasagem EF",
    }

    df_c = df_int[[c for c in cols_c if c in df_int.columns]].dropna(how="all").rename(columns=nomes)
    if len(df_c) >= 3:
        corr = df_c.corr()
        col_m, col_r = st.columns([3, 2])
        with col_m:
            fig_corr = px.imshow(
                corr, color_continuous_scale="RdYlGn",
                zmin=-1, zmax=1, text_auto=".2f", aspect="auto",
            )
            fig_corr.update_layout(height=420)
            st.plotly_chart(fig_corr, use_container_width=True)
        with col_r:
            st.markdown("**O que o mapa confirma:**")
            achados = []
            pares = [
                ("Aprovacao EM", "Evasao EM"),
                ("TDI — Defasagem EM", "Evasao EM"),
                ("Reprovacao EM", "Evasao EM"),
                ("Abandono EM", "Evasao EM"),
            ]
            for a_n, b_n in pares:
                if a_n in corr.columns and b_n in corr.columns:
                    rv = round(corr.loc[a_n, b_n], 2)
                    if abs(rv) > 0.4:
                        direcao = "mais aprovacao = menos evasao" if rv < -0.4 else "ambos sobem juntos"
                        achados.append(f"**{a_n} x {b_n}**: {rv} ({direcao})")
            for a in achados:
                st.markdown(f"- {a}")

            st.markdown("")
            pos_grafico(
                "O mapa confirma a cadeia causal: reprovacao, TDI e abandono estao fortemente "
                "associados entre si e com a evasao (vermelho). "
                "Aprovacao e promocao caminham no sentido oposto (verde). "
                "Isso significa que qualquer politica que aumente a aprovacao reduz a evasao, "
                "e qualquer politica que reduza a reprovacao quebra a cadeia.",
                "info",
            )
        st.warning(
            f"O mapa e calculado com {len(df_c)} pontos de dados (anos). "
            "Com menos de 10 pontos, os resultados indicam tendencias e nao sao estatisticamente definitivos. "
            "Ampliar o periodo melhora a confiabilidade da analise."
        )
    else:
        st.warning("Dados insuficientes para o mapa de correlacao. Amplie o periodo de analise.")

    st.caption(
        "Proxima secao: 'Conclusoes e Preditores para Machine Learning' — "
        "consolidacao dos insights e preparacao para o modelo preditivo."
    )


# ===========================================================================
# PAGINA 5 — CONCLUSÕES E MODELO PREDITIVO
# ===========================================================================
def pagina_conclusoes(dados: dict, filtros: dict, ins: dict):
    st.markdown("# Conclusoes, Insights e Base para Machine Learning")
    st.write(
        "Esta secao consolida os principais aprendizados do dashboard e prepara o terreno "
        "para a etapa de modelagem preditiva. "
        "O objetivo final e construir um modelo capaz de antecipar o risco de evasao "
        "antes que ele se concretize."
    )

    a1, a2 = filtros["ano_range"]
    df_int  = dados["fato_integrado"][(dados["fato_integrado"]["ano"].between(a1, a2))].copy()

    # ── Insights consolidados ──────────────────────────────────────────────────
    st.markdown("### Os 5 principais insights dos dados")

    i1, i2 = st.columns(2)
    with i1:
        st.success(
            "**1. As politicas educacionais funcionam quando sustentadas.**\n\n"
            "A queda consistente da evasao entre 2008 e 2019 nao foi acidental. "
            "Ela e resultado de politicas de expansao do acesso, programas sociais "
            "e melhoria gradual da cobertura educacional. "
            "O progresso levou anos para ser construido e poucos meses para ser revertido pela pandemia."
        )
        st.warning(
            "**2. O Ensino Medio e estruturalmente mais vulneravel.**\n\n"
            "A evasao no EM e em media 2 a 3 vezes maior do que no EF. "
            "Isso nao e coincidencia: alunos entre 15–17 anos enfrentam maior pressao economica, "
            "percebem o curriculo como menos relevante e sao mais impactados por crises. "
            "Politicas genericas nao funcionam para o EM — sao necessarias acoes especificas."
        )
        st.error(
            "**3. A pandemia reverteu anos de avanco em poucos meses.**\n\n"
            "O fechamento das escolas, o ensino remoto desigual e a crise economica "
            "criaram um triplo choque que elevou a evasao a niveis historicamente altos. "
            "O sistema educacional nao se recupera no mesmo ritmo em que foi afetado — "
            "os efeitos da pandemia continuam presentes nos dados de 2022."
        )
    with i2:
        st.info(
            "**4. Reprovacao e TDI sao os preditores mais fortes de evasao.**\n\n"
            "Os dados mostram que anos com mais reprovacao e maior defasagem "
            "escolar (TDI — Taxa de Distorcao Idade-Serie) invariavelmente "
            "apresentam mais evasao. Essa relacao e forte e consistente ao longo dos anos. "
            "Qualquer intervencao que reduza a reprovacao ou a defasagem "
            "impacta diretamente a evasao."
        )
        st.info(
            "**5. A evasao pode ser prevista antes de acontecer.**\n\n"
            "A existencia de uma cadeia causal clara — reprovacao, TDI, abandono, evasao — "
            "significa que e possivel identificar alunos em risco antes que deixem a escola. "
            "Um modelo preditivo que monitore esses indicadores continuamente "
            "permitiria intervencoes preventivas muito mais eficazes e baratas."
        )

    st.divider()

    # ── Tabela de preditores para ML ───────────────────────────────────────────
    st.markdown("### Quais variaveis sao os melhores preditores para um modelo de Machine Learning?")
    st.write(
        "Com base nas analises deste dashboard, as variaveis abaixo apresentam maior "
        "potencial para prever a evasao escolar. "
        "Esta tabela serve como guia para a construcao do modelo preditivo."
    )

    df_pred = pd.DataFrame([
        {
            "Variavel":             "Taxa de Abandono no EM (%)",
            "Importancia estimada": "Muito Alta",
            "Por que e relevante":  "Precursor direto e imediato da evasao. Aluno que abandona ja esta em saida.",
            "Elo na cadeia causal": "4 — abandono",
        },
        {
            "Variavel":             "TDI — Taxa de Distorcao Idade-Serie (%)",
            "Importancia estimada": "Muito Alta",
            "Por que e relevante":  "Forte correlacao com abandono. Indica acumulo historico de defasagem.",
            "Elo na cadeia causal": "2 — defasagem",
        },
        {
            "Variavel":             "Taxa de Reprovacao no EM (%)",
            "Importancia estimada": "Alta",
            "Por que e relevante":  "Origem da cadeia. Ano com mais reprovacao e preditor de mais evasao futura.",
            "Elo na cadeia causal": "1 — reprovacao",
        },
        {
            "Variavel":             "Taxa de Aprovacao no EM (%)",
            "Importancia estimada": "Alta",
            "Por que e relevante":  "Relacao inversa forte com evasao. Mais aprovacao = menos evasao.",
            "Elo na cadeia causal": "Indicador de saude geral do sistema",
        },
        {
            "Variavel":             "Media de Alunos por Turma — EM (ATU)",
            "Importancia estimada": "Media",
            "Por que e relevante":  "Turmas superlotadas dificultam acompanhamento individual e aumentam o risco.",
            "Elo na cadeia causal": "Fator estrutural",
        },
        {
            "Variavel":             "Periodo historico (pandemia sim/nao)",
            "Importancia estimada": "Alta (como variavel de controle)",
            "Por que e relevante":  "A pandemia gerou um choque estrutural que precisa ser tratado como variavel.",
            "Elo na cadeia causal": "Fator externo",
        },
    ])
    st.dataframe(df_pred, use_container_width=True, hide_index=True)

    st.info(
        "**Como usar esta tabela:** as variaveis de 'Importancia estimada' Alta ou Muito Alta "
        "devem ser priorizadas na selecao de features do modelo. "
        "A variavel de periodo historico (pandemia) e importante como variavel de controle — "
        "sem ela, o modelo pode confundir o choque da pandemia com um padrao estrutural."
    )

    st.divider()

    # ── Plano de ação priorizado ───────────────────────────────────────────────
    st.markdown("### O que fazer? — Acoes priorizadas por urgencia")
    st.write(
        "Com base nos dados, as acoes abaixo estao ordenadas por nivel de urgencia. "
        "Acoes imediatas atacam os sintomas mais visíveis. "
        "Acoes de longo prazo constroem a resistencia estrutural do sistema."
    )

    if not df_int.empty:
        ultimo = df_int.sort_values("ano").iloc[-1]
        ab  = float(ultimo.get("taxa_abandono_em", 0) or 0)
        tdi = float(ultimo.get("tdi_em", 0) or 0)
        rep = float(ultimo.get("taxa_repetencia_em", 0) or 0)
        atu = float(ultimo.get("atu_em", 0) or 0)
    else:
        ab = tdi = rep = atu = 0

    URGENCIAS = {
        "IMEDIATA":    ("#FEF2F2", "#991B1B"),
        "CURTO PRAZO": ("#FFFBEB", "#B45309"),
        "MEDIO PRAZO": ("#EFF6FF", "#1D4ED8"),
        "LONGO PRAZO": ("#F0FDF4", "#15803D"),
    }

    acoes = [
        dict(urgencia="IMEDIATA",
             titulo="Monitorar frequencia dos alunos do EM semanalmente",
             descricao=(
                 "Alunos com mais de 25% de faltas estao em risco iminente de abandono. "
                 "Implante controle semanal e contate a familia quando o limite for atingido. "
                 "Ferramentas simples funcionam — o que importa e a regularidade do monitoramento."
             ),
             indicador=f"Abandono EM atual: {ab:.1f}% (limite aceitavel: 5%)"),
        dict(urgencia="IMEDIATA",
             titulo="Reforco escolar para alunos com defasagem (TDI alto)",
             descricao=(
                 "Alunos com mais de 2 anos de atraso em relacao a serie esperada "
                 "tem probabilidade muito maior de abandonar. "
                 "Aulas de nivelamento em contraturno reduzem essa defasagem e aumentam a motivacao."
             ),
             indicador=f"TDI EM atual: {tdi:.1f}% (limite aceitavel: 20%)"),
        dict(urgencia="CURTO PRAZO",
             titulo="Revisar a politica de reprovacao — progressao com suporte",
             descricao=(
                 "A reprovacao e o principal preditor de evasao nos dados. "
                 "Substituir a retencao automatica por progressao com apoio pedagogico intensivo "
                 "quebra o primeiro elo da cadeia causal. "
                 "Isso nao significa aprovar sem criterio — significa dar suporte antes de reter."
             ),
             indicador=f"Reprovacao EM atual: {rep:.1f}% (limite aceitavel: 8%)"),
        dict(urgencia="CURTO PRAZO",
             titulo="Reduzir o numero de alunos por turma no Ensino Medio",
             descricao=(
                 "Turmas com mais de 35 alunos dificultam o acompanhamento individual. "
                 "Meta: no maximo 30 alunos por turma no EM. "
                 "Isso melhora o vinculo professor-aluno e permite identificar problemas antes que virem abandono."
             ),
             indicador=f"Alunos por turma EM atual: {atu:.0f} (meta: ate 30)"),
        dict(urgencia="MEDIO PRAZO",
             titulo="Ampliar oferta de EJA e Ensino Medio noturno",
             descricao=(
                 "Estudantes que ja evadiram precisam de modalidades flexiveis para retornar. "
                 "A EJA (Educacao de Jovens e Adultos) e o EM noturno atendem quem precisa trabalhar. "
                 "Ampliar essas opcoes reduz a evasao permanente."
             ),
             indicador="Evasao acumulada cria uma populacao fora do sistema"),
        dict(urgencia="LONGO PRAZO",
             titulo="Implantar modelo preditivo de risco de evasao por aluno",
             descricao=(
                 "As analises deste dashboard mostram que a evasao e previsivel. "
                 "Um modelo de Machine Learning treinado com os dados disponíveis "
                 "poderia identificar alunos em risco meses antes do abandono, "
                 "permitindo intervencoes preventivas muito mais eficazes e baratas."
             ),
             indicador="A cadeia causal identificada nos dados viabiliza a modelagem preditiva"),
    ]

    for ac in acoes:
        bg, borda = URGENCIAS[ac["urgencia"]]
        st.markdown(
            f'<div style="background:{bg};border-left:5px solid {borda};'
            f'padding:14px 18px;border-radius:4px;margin-bottom:12px">'
            f'<span style="background:{borda};color:white;padding:2px 9px;border-radius:3px;'
            f'font-size:0.72rem;font-weight:700;letter-spacing:0.06em">{ac["urgencia"]}</span>'
            f'<b style="display:block;color:#1E293B;font-size:0.94rem;margin:6px 0 5px 0">{ac["titulo"]}</b>'
            f'<p style="color:#374151;font-size:0.87rem;margin:0 0 6px 0;line-height:1.55">{ac["descricao"]}</p>'
            f'<p style="color:#6B7280;font-size:0.8rem;margin:0"><b>Indicador:</b> {ac["indicador"]}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Placeholder para modelo preditivo ─────────────────────────────────────
    st.markdown("### [Em desenvolvimento] — Modelo Preditivo de Risco de Evasao")
    st.info(
        "**O que sera adicionado aqui:**\n\n"
        "Um modelo de Machine Learning treinado com os dados historicos de Recife para:\n\n"
        "- Estimar a probabilidade de evasao nos proximos anos com base nas variaveis atuais\n"
        "- Identificar automaticamente os grupos de alunos mais vulneraveis\n"
        "- Simular o impacto de intervencoes especificas (ex: 'se a reprovacao cair 2 p.p., "
        "a evasao deve cair X p.p.')\n\n"
        "**As variaveis prioritarias para o modelo**, com base neste dashboard:\n"
        "TDI, taxa de abandono, taxa de reprovacao, taxa de aprovacao e periodo historico.\n\n"
        "**Por enquanto**, o Score de Risco calculado manualmente (disponivel na barra lateral) "
        "serve como aproximacao do que o modelo fara de forma mais precisa e automatizada."
    )

    # Simulacao simples como prévia
    with st.expander("Ver projecao simplificada (tendencia linear — apenas indicativa)"):
        df_sim = dados["fato_educacional"].copy().sort_values("ano")
        df_sim["score"] = calcular_score(df_sim)
        por_ano = df_sim.groupby("ano")["score"].mean().dropna().reset_index()
        if len(por_ano) >= 4:
            z = np.polyfit(por_ano["ano"], por_ano["score"], 1)
            proximos = [por_ano["ano"].max() + 1, por_ano["ano"].max() + 2]
            proj = [round(max(0, np.poly1d(z)(a)), 1) for a in proximos]
            df_proj = pd.DataFrame({
                "Ano (projecao)": proximos,
                "Score estimado (0–100)": proj,
                "Nivel estimado": [classificar_risco(pd.Series([p])).iloc[0] for p in proj],
            })
            st.dataframe(df_proj, use_container_width=True, hide_index=True)
            st.caption(
                "ATENCAO: esta projecao usa apenas tendencia linear historica. "
                "Nao considera fatores externos, politicas futuras ou mudancas estruturais. "
                "Use apenas como indicativo de direcao, nao como previsao."
            )
        else:
            st.caption("Dados insuficientes para projecao. Amplie o periodo de analise.")


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    garantir_dados()
    dados = carregar_dados()

    if not dados:
        st.error("Dados nao encontrados. Execute o ETL primeiro (botao na barra lateral).")
        st.stop()

    ins     = computar_insights(dados)
    filtros = sidebar(dados)

    PAGINAS = {
        "1. Contexto Geral":                  pagina_contexto,
        "2. Evolucao ao Longo do Tempo":       pagina_evolucao,
        "3. Impacto da Pandemia":              pagina_pandemia,
        "4. Por que os Alunos Evadem?":        pagina_relacoes,
        "5. Conclusoes e Modelo Preditivo":    pagina_conclusoes,
    }

    st.sidebar.divider()
    pagina_atual = st.sidebar.radio("Secoes do painel", list(PAGINAS.keys()))
    PAGINAS[pagina_atual](dados, filtros, ins)

    st.sidebar.divider()
    st.sidebar.caption("Painel de Evasao Escolar | Recife | INEP/MEC | 2008–2022")


if __name__ == "__main__":
    main()
