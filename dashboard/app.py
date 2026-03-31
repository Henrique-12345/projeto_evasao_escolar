"""
Dashboard — Evasão Escolar em Recife
======================================
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
    page_icon="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/Brasão_do_Recife.svg/32px-Brasão_do_Recife.svg.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS global — aparência institucional
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    h1 { font-size: 1.7rem !important; font-weight: 700; color: #1E3A5F; }
    h2 { font-size: 1.25rem !important; font-weight: 600; color: #1E3A5F; }
    h3 { font-size: 1.05rem !important; font-weight: 600; color: #334155; }
    .stMetric label { font-size: 0.8rem !important; color: #64748B; }
    .stCaption { color: #64748B; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"

# ---------------------------------------------------------------------------
# Paleta de cores — sem emojis, uso consistente em todo o dashboard
# ---------------------------------------------------------------------------
COR_PRIMARIA  = "#1E3A5F"   # Azul institucional escuro
COR_EF        = "#2563EB"   # Azul — Ensino Fundamental
COR_EM        = "#DC2626"   # Vermelho — Ensino Médio
COR_ABANDONO  = "#EA580C"   # Laranja — Abandono
COR_POSITIVO  = "#15803D"   # Verde — indicadores positivos / melhora
COR_ALERTA    = "#B45309"   # Âmbar — atenção moderada
COR_CRITICO   = "#991B1B"   # Vermelho escuro — risco crítico
COR_CINZA     = "#64748B"   # Cinza — referências / linhas auxiliares

CORES_RISCO = {
    "Critico":  "#991B1B",
    "Alto":     "#DC2626",
    "Moderado": "#B45309",
    "Baixo":    "#15803D",
}

PALETA_PERIODO = {
    "2006–2010":             "#94A3B8",
    "2011–2015":             "#60A5FA",
    "2016–2019":             "#34D399",
    "2020–2022 (Pandemia)":  "#FBBF24",
    "2023–2024":             "#F87171",
}

LIMIARES = {"baixo": 20, "moderado": 35, "alto": 50}

# ---------------------------------------------------------------------------
# Glossário — termos técnicos com definições curtas
# ---------------------------------------------------------------------------
GLOSSARIO = {
    "Evasão escolar":      "Saída definitiva do aluno do sistema de ensino, sem previsão de retorno. Diferente do abandono, que é a saída no meio de um ano letivo específico.",
    "Abandono escolar":    "Saída do aluno durante o ano letivo em curso, sem concluir o período. Considerado o precursor imediato da evasão.",
    "TDI":                 "Taxa de Distorção Idade-Série. Percentual de alunos com mais de 2 anos de atraso em relação à idade esperada para a série cursada. Alto TDI indica acúmulo histórico de reprovações.",
    "Taxa de Repetência":  "Percentual de alunos que não foram promovidos ao final do ano letivo (mesmo critério da Base Socioeconômica).",
    "Taxa de Reprovação":  "Percentual de alunos com resultado final negativo (Base Educacional). Equivalente à repetência.",
    "Taxa de Promoção":    "Percentual de alunos aprovados e promovidos para a série seguinte.",
    "Taxa de Aprovação":   "Percentual de alunos com resultado final positivo no ano letivo.",
    "ATU":                 "Média de Alunos por Turma. Indica o tamanho médio das turmas. Valores acima de 30 no EM são considerados elevados.",
    "HAD":                 "Horas-Aula Diárias. Média de horas de aula por dia letivo.",
    "EF":                  "Ensino Fundamental — corresponde do 1º ao 9º ano.",
    "EM":                  "Ensino Médio — corresponde do 1º ao 3º ano.",
    "p.p.":                "Ponto percentual — diferença absoluta entre duas taxas percentuais. Exemplo: de 10% para 12% = variação de +2 p.p.",
    "Score de Risco":      "Indicador composto (0 a 100) calculado como: Abandono EM × 40% + TDI EM × 30% + Reprovação EM × 30%. Quanto maior o score, maior o risco de evasão.",
    "Correlação de Pearson":"Mede a força e direção da relação linear entre dois indicadores. Varia de -1 (relação inversa perfeita) a +1 (relação direta perfeita). Valores próximos de 0 indicam pouca ou nenhuma relação linear.",
    "Base Socioeconômica": "Conjunto de dados com indicadores de promoção, repetência e evasão por ano.",
    "Base Educacional":    "Conjunto de dados com indicadores de infraestrutura escolar (ATU, HAD), distorção idade-série (TDI) e fluxo (aprovação, reprovação, abandono).",
}

# ---------------------------------------------------------------------------
# Carregamento de dados
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Carregando dados...")
def carregar_dados() -> dict[str, pd.DataFrame]:
    tabelas = {}
    for csv in PROC.glob("*.csv"):
        tabelas[csv.stem] = pd.read_csv(csv)
    return tabelas


def garantir_dados():
    if not PROC.exists() or not list(PROC.glob("*.csv")):
        with st.spinner("Executando ETL pela primeira vez..."):
            sys.path.insert(0, str(ROOT))
            from etl.etl_pipeline import run_etl
            run_etl()
        st.cache_data.clear()


# ---------------------------------------------------------------------------
# Score de Risco 0–100
# ---------------------------------------------------------------------------

def calcular_score(df: pd.DataFrame) -> pd.Series:
    """Score de risco 0–100: abandono_em(40%) + tdi_em(30%) + reprovacao_em(30%)."""
    def col(nome):
        return df[nome].fillna(0) if nome in df.columns else pd.Series(0.0, index=df.index)

    score = pd.Series(np.nan, index=df.index)
    mask_em = df["taxa_abandono_em"].notna() if "taxa_abandono_em" in df.columns else pd.Series(False, index=df.index)
    if mask_em.any():
        s = col("taxa_abandono_em") * 0.40 + col("tdi_em") * 0.30 + col("taxa_reprovacao_em") * 0.30
        score[mask_em] = s[mask_em].clip(0, 100)
    mask_ef = score.isna() & (df["taxa_abandono_ef"].notna() if "taxa_abandono_ef" in df.columns else pd.Series(False, index=df.index))
    if mask_ef.any():
        s = col("taxa_abandono_ef") * 0.40 + col("tdi_ef") * 0.30 + col("taxa_reprovacao_ef") * 0.30
        score[mask_ef] = s[mask_ef].clip(0, 100)
    return score.round(1)


def classificar_risco(score: pd.Series) -> pd.Series:
    return pd.cut(
        score,
        bins=[-np.inf, LIMIARES["baixo"], LIMIARES["moderado"], LIMIARES["alto"], np.inf],
        labels=["Baixo", "Moderado", "Alto", "Critico"],
    ).astype(str)


# ---------------------------------------------------------------------------
# Componentes visuais reutilizáveis
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
            x=xl, y=np.poly1d(z)(xl),
            mode="lines", line=dict(color=cor, width=2, dash="dash"),
            name="Tendência linear", showlegend=False,
        ))
    return fig


def caixa_destaque(tipo: str, titulo: str, texto: str):
    """Bloco informativo sem emojis, com borda colorida por tipo."""
    estilos = {
        "alerta":   ("#FEF2F2", "#DC2626"),
        "atencao":  ("#FFFBEB", "#B45309"),
        "positivo": ("#F0FDF4", "#15803D"),
        "info":     ("#EFF6FF", "#1D4ED8"),
        "neutro":   ("#F8FAFC", "#475569"),
    }
    bg, borda = estilos.get(tipo, estilos["info"])
    st.markdown(
        f"""<div style="background:{bg};border-left:5px solid {borda};
        padding:14px 18px;border-radius:4px;margin:10px 0 16px 0;">
        <b style="color:{borda};font-size:0.85rem;text-transform:uppercase;letter-spacing:0.04em">{titulo}</b>
        <p style="color:#374151;font-size:0.9rem;margin:6px 0 0 0;line-height:1.6">{texto}</p>
        </div>""",
        unsafe_allow_html=True,
    )


def caixa_acao(texto: str):
    """Bloco de implicação para tomada de decisão."""
    st.markdown(
        f"""<div style="background:#F0F9FF;border-left:5px solid #0369A1;
        padding:12px 18px;border-radius:4px;margin:12px 0 20px 0;">
        <b style="color:#0369A1;font-size:0.82rem;text-transform:uppercase;letter-spacing:0.04em">O que esta informacao permite fazer</b>
        <p style="color:#0C4A6E;font-size:0.9rem;margin:6px 0 0 0;line-height:1.6">{texto}</p>
        </div>""",
        unsafe_allow_html=True,
    )


def caixa_qualidade(texto: str):
    """Nota sobre qualidade ou limitação dos dados."""
    st.markdown(
        f"""<div style="background:#FEFCE8;border-left:4px solid #CA8A04;
        padding:10px 16px;border-radius:4px;margin:8px 0 16px 0;">
        <b style="color:#92400E;font-size:0.8rem;text-transform:uppercase">Nota sobre os dados</b>
        <p style="color:#78350F;font-size:0.85rem;margin:4px 0 0 0;line-height:1.5">{texto}</p>
        </div>""",
        unsafe_allow_html=True,
    )


def titulo_pergunta(pergunta: str, contexto: str = ""):
    st.markdown(f"### {pergunta}")
    if contexto:
        st.caption(contexto)
    st.markdown('<hr style="border:none;border-top:1px solid #E2E8F0;margin:8px 0 16px 0">', unsafe_allow_html=True)


def badge_risco(nivel: str) -> str:
    cor = CORES_RISCO.get(nivel, "#64748B")
    return f'<span style="background:{cor};color:white;padding:2px 10px;border-radius:3px;font-size:0.78rem;font-weight:600">{nivel.upper()}</span>'


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------

def sidebar(dados: dict) -> dict:
    st.sidebar.markdown(
        f'<h2 style="color:{COR_PRIMARIA};font-size:1.1rem;margin-bottom:0">Evasão Escolar</h2>'
        '<p style="color:#64748B;font-size:0.82rem;margin-top:2px">Recife — Ensino Fundamental e Médio<br>Fonte: INEP / MEC</p>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown('<hr style="border:none;border-top:1px solid #E2E8F0;margin:10px 0">', unsafe_allow_html=True)

    anos = sorted(dados["fato_integrado"]["ano"].unique())
    a_min, a_max = int(min(anos)), int(max(anos))
    # Janela padrão: últimos 4 anos disponíveis
    default_ini = max(a_min, a_max - 3)

    filtros = {}
    filtros["ano_range"] = st.sidebar.slider(
        "Período de análise",
        min_value=a_min, max_value=a_max,
        value=(default_ini, a_max),
        help="Selecione o intervalo de anos a ser analisado. O padrão são os últimos 4 anos disponíveis.",
    )
    filtros["nivel"] = st.sidebar.multiselect(
        "Nível de ensino",
        ["Ensino Fundamental (EF)", "Ensino Médio (EM)"],
        default=["Ensino Fundamental (EF)", "Ensino Médio (EM)"],
        help="EF = Ensino Fundamental (1º ao 9º ano) | EM = Ensino Médio (1º ao 3º ano)",
    )
    filtros["risco_filtro"] = st.sidebar.multiselect(
        "Filtrar por nível de risco",
        ["Baixo", "Moderado", "Alto", "Critico"],
        default=["Alto", "Critico"],
        help="Filtra os registros no ranking da página de análise de risco.",
    )

    st.sidebar.markdown('<hr style="border:none;border-top:1px solid #E2E8F0;margin:10px 0">', unsafe_allow_html=True)

    with st.sidebar.expander("Escala de risco (Score 0 a 100)"):
        st.markdown("""
        | Nível | Faixa do Score |
        |---|---|
        | **Critico** | Acima de 50 |
        | **Alto** | 35 a 50 |
        | **Moderado** | 20 a 35 |
        | **Baixo** | Abaixo de 20 |

        *Score = Abandono EM (40%) + TDI (30%) + Reprovação EM (30%)*
        """)

    with st.sidebar.expander("Glossário de termos"):
        for termo, definicao in GLOSSARIO.items():
            st.markdown(f"**{termo}**")
            st.caption(definicao)

    st.sidebar.markdown('<hr style="border:none;border-top:1px solid #E2E8F0;margin:10px 0">', unsafe_allow_html=True)

    if st.sidebar.button("Reprocessar ETL"):
        st.cache_data.clear()
        sys.path.insert(0, str(ROOT))
        from etl.etl_pipeline import run_etl
        run_etl()
        st.rerun()

    return filtros


# ===========================================================================
# PÁGINA 1 — PAINEL DE INDICADORES
# ===========================================================================

def pagina_painel(dados: dict, filtros: dict):
    st.markdown(f'<h1>Painel de Indicadores — Evasão Escolar em Recife</h1>', unsafe_allow_html=True)
    st.caption(
        "Visão executiva dos principais indicadores de evasão e abandono escolar. "
        "Use esta página para identificar rapidamente a situação atual e os alertas mais críticos."
    )
    st.markdown("---")

    a1, a2 = filtros["ano_range"]
    df_int  = dados["fato_integrado"].copy()
    df_educ = dados["fato_educacional"].copy()
    df_int  = df_int[(df_int["ano"]  >= a1) & (df_int["ano"]  <= a2)]
    df_educ = df_educ[(df_educ["ano"] >= a1) & (df_educ["ano"] <= a2)]

    if df_int.empty:
        st.warning(f"Não há dados disponíveis para o período {a1}–{a2}. Ajuste o filtro na barra lateral.")
        return

    df_s      = df_int.sort_values("ano")
    ultimo    = df_s.iloc[-1]
    penultimo = df_s.iloc[-2] if len(df_s) > 1 else ultimo

    def sv(col):
        v = ultimo.get(col, np.nan)
        return float(v) if pd.notna(v) else np.nan

    def delta_pp(col):
        v1 = sv(col)
        v2 = float(penultimo.get(col, np.nan) or np.nan)
        if pd.isna(v1) or pd.isna(v2): return None
        d = round(v1 - v2, 2)
        return f"{d:+.1f} p.p. vs. {int(penultimo['ano'])}"

    # ── Indicadores-chave ────────────────────────────────────────────────────
    st.markdown("#### Indicadores-chave — ano de referência: " + str(int(ultimo["ano"])))
    st.caption(
        "p.p. = ponto percentual (diferença absoluta entre dois percentuais). "
        "A seta indica a variação em relação ao ano anterior disponível."
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Evasão EF (%)",
                  f"{sv('taxa_evasao_ef'):.1f}%" if pd.notna(sv('taxa_evasao_ef')) else "–",
                  delta_pp("taxa_evasao_ef"),
                  delta_color="inverse",
                  help="Evasão no Ensino Fundamental: saída definitiva do sistema. Seta vermelha = piora.")
    with c2:
        st.metric("Evasão EM (%)",
                  f"{sv('taxa_evasao_em'):.1f}%" if pd.notna(sv('taxa_evasao_em')) else "–",
                  delta_pp("taxa_evasao_em"),
                  delta_color="inverse",
                  help="Evasão no Ensino Médio. Historicamente 2 a 3 vezes maior do que no EF.")
    with c3:
        st.metric("Abandono EM (%)",
                  f"{sv('taxa_abandono_em'):.1f}%" if pd.notna(sv('taxa_abandono_em')) else "–",
                  delta_pp("taxa_abandono_em"),
                  delta_color="inverse",
                  help="Abandono = saída durante o ano letivo. É o sinal de alerta mais imediato da evasão.")
    with c4:
        st.metric("TDI EM (%)",
                  f"{sv('tdi_em'):.1f}%" if pd.notna(sv('tdi_em')) else "–",
                  delta_pp("tdi_em"),
                  delta_color="inverse",
                  help="TDI = Taxa de Distorção Idade-Série. Percentual de alunos com mais de 2 anos de atraso escolar.")
    with c5:
        sc = sv("indice_risco_evasao")
        nivel_sc = classificar_risco(pd.Series([sc if pd.notna(sc) else 0])).iloc[0]
        st.metric("Score de Risco (0–100)",
                  f"{sc:.1f}" if pd.notna(sc) else "–",
                  delta_pp("indice_risco_evasao"),
                  delta_color="inverse",
                  help="Score composto: Abandono EM (40%) + TDI EM (30%) + Reprovação EM (30%). Quanto maior, maior o risco.")

    st.markdown("---")

    # ── Alertas contextualizados ──────────────────────────────────────────────
    st.markdown("#### Alertas e interpretação do cenário atual")

    ev_em  = sv("taxa_evasao_em") or 0
    ab_em  = sv("taxa_abandono_em") or 0
    tdi_em = sv("tdi_em") or 0
    sc     = sv("indice_risco_evasao") or 0
    sc_ant = float(penultimo.get("indice_risco_evasao", sc) or sc)
    delta_sc = sc - sc_ant
    n_alertas = 0

    if sc > LIMIARES["alto"]:
        caixa_destaque("alerta", f"Score de Risco Critico — {sc:.1f} pontos",
            f"O score de risco está acima de {LIMIARES['alto']}, indicando situação crítica. "
            f"Evasão EM: {ev_em:.1f}% | Abandono EM: {ab_em:.1f}% | TDI EM: {tdi_em:.1f}%. "
            "Um score neste nível significa que os três principais indicadores estão simultaneamente elevados, "
            "o que exige ação coordenada da gestão escolar e da Secretaria de Educação.")
        n_alertas += 1
    elif sc > LIMIARES["moderado"]:
        caixa_destaque("atencao", f"Score de Risco Elevado — {sc:.1f} pontos",
            f"O score está na faixa de atenção ({LIMIARES['moderado']}–{LIMIARES['alto']}). "
            f"Evasão EM: {ev_em:.1f}% | Abandono EM: {ab_em:.1f}%. "
            "A situação não é crítica, mas requer monitoramento próximo e intervenções preventivas.")
        n_alertas += 1

    if delta_sc > 2:
        caixa_destaque("atencao", f"Piora no Score de Risco (+{delta_sc:.1f} pontos vs. {int(penultimo['ano'])})",
            f"O score cresceu {delta_sc:.1f} pontos em relação ao período anterior. "
            "Uma variação positiva significa que os indicadores de risco pioraram — o que merece atenção mesmo que o nível absoluto ainda seja aceitável.")
        n_alertas += 1
    elif delta_sc < -2:
        caixa_destaque("positivo", f"Melhora no Score de Risco ({delta_sc:+.1f} pontos vs. {int(penultimo['ano'])})",
            f"Os indicadores melhoraram {abs(delta_sc):.1f} pontos em relação ao período anterior. "
            "Isso indica que as políticas educacionais em curso estão surtindo efeito — é importante identificar quais ações contribuíram para essa melhora e mantê-las.")
        n_alertas += 1

    if tdi_em > 25:
        caixa_destaque("atencao", f"Alta Distorção Idade-Série no EM — TDI: {tdi_em:.1f}%",
            f"Mais de 1 em cada 4 alunos do Ensino Médio está cursando uma série que não corresponde à sua faixa etária. "
            "A TDI elevada é resultado do acúmulo de reprovações ao longo dos anos e está diretamente associada ao aumento do abandono. "
            "Alunos com dois ou mais anos de defasagem têm probabilidade significativamente maior de abandonar a escola.")
        n_alertas += 1

    if ev_em > 10:
        caixa_destaque("alerta", f"Evasão no EM acima do limite crítico — {ev_em:.1f}%",
            "Uma taxa de evasão acima de 10% no Ensino Médio significa que mais de 1 em cada 10 alunos "
            "abandona definitivamente o sistema escolar. Esse nível exige intervenção imediata.")
        n_alertas += 1

    df_educ_s = df_educ.copy()
    df_educ_s["score"] = calcular_score(df_educ_s)
    df_educ_s["nivel"] = classificar_risco(df_educ_s["score"])
    n_crit = df_educ_s["nivel"].isin(["Alto", "Critico"]).sum()
    if n_crit > 0:
        caixa_destaque("atencao" if n_crit < 10 else "alerta",
            f"{n_crit} de {len(df_educ_s)} registros em nível Alto ou Critico",
            f"No período {a1}–{a2}, {n_crit} registros escolares apresentam score de risco elevado. "
            "Consulte a página 'Onde está o risco?' para ver o ranking detalhado e identificar os casos prioritários.")
        n_alertas += 1

    if n_alertas == 0:
        caixa_destaque("positivo", "Nenhum alerta critico no periodo selecionado",
            "Os indicadores do período estão dentro dos limiares aceitáveis. "
            "Continue monitorando a evolução para identificar tendências emergentes antes que se tornem críticas.")

    caixa_acao(
        "Os alertas acima indicam onde concentrar energia e recursos. Situações críticas (score > 50) exigem ação imediata; "
        "situações de atenção (score entre 35 e 50) exigem plano de melhoria a ser implantado nos próximos meses. "
        "Consulte a página 'O que fazer?' para recomendações específicas."
    )

    st.markdown("---")

    # ── Score de Risco: gauge + evolução ──────────────────────────────────────
    col1, col2 = st.columns([1, 2])

    with col1:
        titulo_pergunta("Qual é o nível de risco atual?",
                        "O Score de Risco resume em um único número a gravidade da situação no Ensino Médio.")
        sc_v = sv("indice_risco_evasao")
        if pd.notna(sc_v):
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=float(sc_v),
                delta={"reference": sc_ant, "valueformat": ".1f",
                       "increasing": {"color": COR_CRITICO}, "decreasing": {"color": COR_POSITIVO}},
                number={"suffix": " / 100", "font": {"size": 26}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#475569"},
                    "bar": {"color": CORES_RISCO.get(nivel_sc, COR_CINZA)},
                    "steps": [
                        {"range": [0,  20],  "color": "#DCFCE7"},
                        {"range": [20, 35],  "color": "#FEF9C3"},
                        {"range": [35, 50],  "color": "#FEE2E2"},
                        {"range": [50, 100], "color": "#FECACA"},
                    ],
                },
                title={"text": f"Nivel: {nivel_sc}", "font": {"size": 14}},
            ))
            fig_g.update_layout(height=270, margin=dict(t=40, b=0, l=20, r=20))
            st.plotly_chart(fig_g, use_container_width=True)
            st.caption(
                f"Score atual: {sc_v:.1f} — Nivel: {nivel_sc}. "
                f"A variação em relação a {int(penultimo['ano'])} é de {delta_sc:+.1f} pontos. "
                "Verde = baixo risco | Amarelo = atenção | Laranja = alto | Vermelho = critico."
            )

    with col2:
        titulo_pergunta("Como o score de risco evoluiu ao longo do tempo?",
                        "A linha mostra a evolução anual do score. As faixas coloridas indicam o nível de risco correspondente.")
        if "indice_risco_evasao" in df_int.columns:
            dt = df_int.dropna(subset=["indice_risco_evasao"]).sort_values("ano")
            fig_t = go.Figure()
            fig_t.add_hrect(y0=0,  y1=20,  fillcolor="#DCFCE7", opacity=0.25, line_width=0, annotation_text="Baixo",    annotation_position="right")
            fig_t.add_hrect(y0=20, y1=35,  fillcolor="#FEF9C3", opacity=0.25, line_width=0, annotation_text="Moderado", annotation_position="right")
            fig_t.add_hrect(y0=35, y1=50,  fillcolor="#FEE2E2", opacity=0.25, line_width=0, annotation_text="Alto",     annotation_position="right")
            fig_t.add_hrect(y0=50, y1=100, fillcolor="#FECACA", opacity=0.25, line_width=0, annotation_text="Critico",  annotation_position="right")
            fig_t.add_trace(go.Scatter(
                x=dt["ano"], y=dt["indice_risco_evasao"],
                mode="lines+markers+text",
                text=dt["indice_risco_evasao"].round(1),
                textposition="top center",
                line=dict(color=COR_PRIMARIA, width=3),
                marker=dict(size=9, color=dt["indice_risco_evasao"],
                            colorscale="RdYlGn_r", cmin=0, cmax=60,
                            line=dict(color="white", width=2)),
                name="Score de Risco",
            ))
            if a1 <= 2020 <= a2:
                fig_t.add_vrect(x0=2019.5, x1=2022.5, fillcolor="#FEF9C3", opacity=0.3,
                                line_width=0, annotation_text="Pandemia (2020–2022)",
                                annotation_position="top left",
                                annotation_font=dict(size=11, color="#92400E"))
            fig_t.update_layout(
                yaxis_title="Score de Risco (0–100)", xaxis_title="Ano",
                hovermode="x unified", height=300, showlegend=False,
            )
            st.plotly_chart(fig_t, use_container_width=True)
            if len(dt) >= 3:
                val_max = dt["indice_risco_evasao"].max()
                ano_max = int(dt.loc[dt["indice_risco_evasao"].idxmax(), "ano"])
                val_min = dt["indice_risco_evasao"].min()
                ano_min = int(dt.loc[dt["indice_risco_evasao"].idxmin(), "ano"])
                st.caption(
                    f"Maior score no período: {val_max:.1f} em {ano_max} | "
                    f"Menor score no período: {val_min:.1f} em {ano_min}. "
                    "A tendência de queda observada reflete avanços nas políticas educacionais do município."
                )

    st.markdown("---")

    # ── Comparação entre início e fim do período ──────────────────────────────
    st.markdown("#### Variação dos indicadores no periodo selecionado")
    st.caption(
        f"Comparação entre {int(df_s.iloc[0]['ano'])} (início do período) e {int(ultimo['ano'])} (último ano). "
        "p.p. = ponto percentual. Variações negativas em evasão, abandono e TDI indicam melhora."
    )

    primeiro = df_s.iloc[0]
    indicadores = [
        ("taxa_evasao_em",     "Evasão EM",          True),
        ("taxa_abandono_em",   "Abandono EM",         True),
        ("tdi_em",             "TDI EM",              True),
        ("taxa_repetencia_em", "Repetência EM",       True),
        ("taxa_aprovacao_em",  "Aprovação EM",        False),
    ]
    cols_i = st.columns(len(indicadores))
    for col_ui, (nome, label, inv) in zip(cols_i, indicadores):
        v_ini = float(primeiro.get(nome, np.nan) or np.nan)
        v_fim = float(ultimo.get(nome, np.nan) or np.nan)
        if pd.notna(v_ini) and pd.notna(v_fim):
            with col_ui:
                st.metric(
                    label=label + " (%)",
                    value=f"{v_fim:.1f}%",
                    delta=f"{v_fim - v_ini:+.1f} p.p. vs. {int(primeiro['ano'])}",
                    delta_color="inverse" if inv else "normal",
                )


# ===========================================================================
# PÁGINA 2 — ONDE ESTÁ O RISCO
# ===========================================================================

def pagina_onde(dados: dict, filtros: dict):
    st.markdown('<h1>Onde está o risco? — Identificação e Ranking de Risco</h1>', unsafe_allow_html=True)
    st.caption(
        "Esta página identifica quais períodos e registros escolares apresentam maior risco de evasão, "
        "com base no Score de Risco calculado. Use o ranking para priorizar onde intervir primeiro."
    )
    st.markdown("---")

    a1, a2 = filtros["ano_range"]
    df_educ = dados["fato_educacional"].copy()
    df_educ = df_educ[(df_educ["ano"] >= a1) & (df_educ["ano"] <= a2)]
    df_educ["score_risco"] = calcular_score(df_educ)
    df_educ["nivel_risco"] = classificar_risco(df_educ["score_risco"])

    total = len(df_educ)
    n_alto = df_educ["nivel_risco"].isin(["Alto", "Critico"]).sum()

    caixa_destaque(
        "atencao" if n_alto < total * 0.3 else "alerta",
        f"Resumo do Periodo {a1}–{a2}",
        f"Foram analisados {total} registros escolares. "
        f"Destes, {n_alto} ({n_alto/total*100:.0f}%) estão classificados como nível Alto ou Critico. "
        "O nível de risco é calculado a partir do abandono escolar (40%), da distorção idade-série — TDI (30%) "
        "e da taxa de reprovação (30%)."
    )

    # ── Distribuição de risco + evolução do score ──────────────────────────────
    col1, col2 = st.columns([1, 2])

    with col1:
        titulo_pergunta("Como os registros se distribuem entre os niveis de risco?",
                        "Cada registro representa um conjunto de escolas em um determinado ano.")
        contagem = df_educ["nivel_risco"].value_counts().reindex(
            ["Critico", "Alto", "Moderado", "Baixo"], fill_value=0
        )
        fig_pie = go.Figure(go.Pie(
            labels=contagem.index.tolist(),
            values=contagem.values.tolist(),
            marker_colors=[CORES_RISCO["Critico"], CORES_RISCO["Alto"],
                           CORES_RISCO["Moderado"], CORES_RISCO["Baixo"]],
            hole=0.5, textinfo="label+percent+value",
            textfont=dict(size=11),
        ))
        fig_pie.update_layout(
            height=320, showlegend=False,
            annotations=[dict(text=f"{total}<br>registros", x=0.5, y=0.5,
                              font=dict(size=13, color="#1E3A5F"), showarrow=False)],
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        st.caption(
            "O gráfico de rosca mostra a proporção de registros em cada nível de risco. "
            "Um percentual elevado em 'Alto' ou 'Critico' indica que o período analisado concentra "
            "muitos casos que precisam de atenção."
        )

    with col2:
        titulo_pergunta("Como o score de risco variou a cada ano?",
                        "A linha mostra a média anual do score. O intervalo sombreado mostra a variação entre o valor mínimo e máximo observados naquele ano.")
        score_ano = df_educ.groupby("ano")["score_risco"].agg(["mean", "max", "min"]).reset_index()
        score_ano.columns = ["ano", "medio", "maximo", "minimo"]

        fig_sc = go.Figure()
        for limiar, cor_f, label in [
            (20, "#DCFCE7", "Baixo"), (35, "#FEF9C3", "Moderado"),
            (50, "#FEE2E2", "Alto"),  (100, "#FECACA", "Critico"),
        ]:
            fig_sc.add_hrect(y0=0 if label == "Baixo" else LIMIARES[label.lower() if label.lower() in LIMIARES else "alto"],
                             y1=limiar, fillcolor=cor_f, opacity=0.2, line_width=0)

        fig_sc.add_trace(go.Scatter(
            x=pd.concat([score_ano["ano"], score_ano["ano"][::-1]]),
            y=pd.concat([score_ano["maximo"], score_ano["minimo"][::-1]]),
            fill="toself", fillcolor=hex_rgba(COR_EM, 0.1),
            line=dict(color="rgba(0,0,0,0)"), name="Variação (min–máx)",
        ))
        fig_sc.add_trace(go.Scatter(
            x=score_ano["ano"], y=score_ano["medio"],
            mode="lines+markers", name="Score médio anual",
            line=dict(color=COR_PRIMARIA, width=3),
            marker=dict(size=9, color=score_ano["medio"],
                        colorscale="RdYlGn_r", cmin=0, cmax=60,
                        line=dict(color="white", width=2)),
            text=score_ano["medio"].round(1), textposition="top center",
        ))
        if a1 <= 2020 <= a2:
            fig_sc.add_vrect(x0=2019.5, x1=2022.5, fillcolor="#FEF9C3", opacity=0.25,
                             line_width=0, annotation_text="Pandemia",
                             annotation_position="top left",
                             annotation_font=dict(size=10, color="#92400E"))
        fig_sc.update_layout(yaxis_title="Score de Risco (0–100)", xaxis_title="Ano",
                             hovermode="x unified", height=320,
                             legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_sc, use_container_width=True)

        if not score_ano.empty:
            ano_pior = int(score_ano.loc[score_ano["medio"].idxmax(), "ano"])
            pior_val = score_ano["medio"].max()
            ano_melhor = int(score_ano.loc[score_ano["medio"].idxmin(), "ano"])
            melhor_val = score_ano["medio"].min()
            st.caption(
                f"Pior ano no período: {ano_pior} (score médio de {pior_val:.1f}). "
                f"Melhor ano no período: {ano_melhor} (score médio de {melhor_val:.1f}). "
                "A área sombreada ao redor da linha mostra a dispersão entre os valores mínimo e máximo de cada ano — "
                "quanto maior essa área, maior a desigualdade de risco entre as escolas naquele ano."
            )

    caixa_acao(
        "Identifique os anos com score mais elevado e priorize ações nesses períodos. "
        "Use o ranking abaixo para localizar os registros específicos que mais contribuem para o risco total."
    )

    st.markdown("---")

    # ── Ranking ───────────────────────────────────────────────────────────────
    titulo_pergunta("Quais registros apresentam maior risco de evasão?",
                    "Tabela ordenada do maior para o menor score de risco. Use o filtro 'Nível de risco' na barra lateral para focar nos casos mais urgentes.")

    niveis_sel = filtros.get("risco_filtro", ["Alto", "Critico"])
    df_rank = df_educ[df_educ["nivel_risco"].isin(niveis_sel)].copy() if niveis_sel else df_educ.copy()
    df_rank = df_rank.sort_values("score_risco", ascending=False).reset_index(drop=True)
    df_rank.index += 1

    rename = {
        "ano": "Ano", "nivel_risco": "Nivel de Risco", "score_risco": "Score (0–100)",
        "taxa_abandono_em": "Abandono EM (%)", "taxa_abandono_ef": "Abandono EF (%)",
        "tdi_em": "TDI EM (%)", "tdi_ef": "TDI EF (%)",
        "taxa_reprovacao_em": "Reprovacao EM (%)", "taxa_reprovacao_ef": "Reprovacao EF (%)",
        "atu_em": "Alunos por Turma EM", "atu_ef": "Alunos por Turma EF",
    }
    cols_show = [c for c in rename if c in df_rank.columns]
    df_display = df_rank[cols_show].rename(columns=rename)

    st.dataframe(
        df_display, use_container_width=True, height=400,
        column_config={
            "Score (0–100)": st.column_config.ProgressColumn(
                "Score (0–100)", min_value=0, max_value=100, format="%.1f",
            ),
        },
    )
    st.caption(
        f"Exibindo {len(df_display)} registros para os níveis selecionados. "
        "TDI = Taxa de Distorção Idade-Série. Abandono = percentual de alunos que saíram no meio do ano letivo. "
        "Alunos por Turma = tamanho médio das turmas. Um valor de Score mais alto indica maior risco de evasão."
    )

    if df_display.empty:
        caixa_qualidade("Nenhum registro encontrado para os níveis de risco selecionados no período. "
                        "Tente ampliar o filtro de nível de risco ou o intervalo de anos.")

    caixa_acao(
        "Priorize os registros com score mais alto para ações imediatas. "
        "Combine a análise do score com os valores individuais de abandono e TDI para entender "
        "qual componente está puxando o risco para cima em cada caso."
    )

    st.markdown("---")

    # ── Mapa de calor ─────────────────────────────────────────────────────────
    titulo_pergunta("Quando e em qual nivel de risco se concentram os problemas?",
                    "O mapa de calor mostra quantos registros escolares estão em cada combinação de ano e nível de risco. Quanto mais escuro, maior a concentração.")

    pivot = df_educ.pivot_table(
        index="nivel_risco", columns="ano", values="score_risco",
        aggfunc="count", fill_value=0,
    ).reindex(["Critico", "Alto", "Moderado", "Baixo"])

    if not pivot.empty:
        fig_h = px.imshow(
            pivot, color_continuous_scale=["#DCFCE7", "#FEF9C3", "#FEE2E2", "#991B1B"],
            text_auto=True, labels={"color": "Nº de registros"},
            color_continuous_midpoint=pivot.values.mean(),
        )
        fig_h.update_layout(
            height=300,
            xaxis_title="Ano",
            yaxis_title="Nível de Risco",
        )
        st.plotly_chart(fig_h, use_container_width=True)
        st.caption(
            "Cada célula indica quantos registros escolares estão naquele nível de risco para aquele ano. "
            "Colunas com muitas células vermelhas (crítico/alto) indicam anos problemáticos. "
            "Observe como o período da pandemia (2020–2022) pode apresentar mudanças na distribuição."
        )
        caixa_qualidade(
            "Os dados não possuem identificador individual de escola — cada registro representa um conjunto de escolas "
            "com características semelhantes em determinado ano. Isso limita a análise a nível agregado."
        )


# ===========================================================================
# PÁGINA 3 — POR QUE OCORRE
# ===========================================================================

def pagina_causas(dados: dict, filtros: dict):
    st.markdown('<h1>Por que ocorre? — Fatores Associados à Evasão Escolar</h1>', unsafe_allow_html=True)
    st.caption(
        "Esta página analisa os principais fatores que estão associados ao risco de evasão. "
        "Compreender as causas é o primeiro passo para escolher as intervenções mais eficazes."
    )
    st.markdown("---")

    a1, a2 = filtros["ano_range"]
    df_int   = dados["fato_integrado"].copy()
    df_int   = df_int[(df_int["ano"] >= a1) & (df_int["ano"] <= a2)]
    df_socio = dados["dim_socio_anual"].copy()
    df_socio = df_socio[(df_socio["ano"] >= a1) & (df_socio["ano"] <= a2)]

    # ── Cadeia causal ──────────────────────────────────────────────────────────
    st.markdown("#### A cadeia causal da evasão escolar em Recife")
    st.markdown("""
    <div style="background:#F8FAFC;border:1px solid #CBD5E1;padding:18px 22px;border-radius:6px;line-height:2">
    <p style="margin:0;color:#1E293B;font-size:0.92rem">
    Os dados históricos de Recife evidenciam uma cadeia de causas e efeitos que se repete sistematicamente:
    </p>
    <div style="display:flex;align-items:center;flex-wrap:wrap;gap:4px;margin-top:12px;font-size:0.88rem">
    <span style="background:#FEE2E2;color:#991B1B;padding:5px 12px;border-radius:4px;font-weight:600">Reprovacao</span>
    <span style="color:#64748B;font-size:1rem">→</span>
    <span style="background:#FEF9C3;color:#92400E;padding:5px 12px;border-radius:4px;font-weight:600">Distorcao Idade-Serie (TDI)</span>
    <span style="color:#64748B;font-size:1rem">→</span>
    <span style="background:#FFEDD5;color:#9A3412;padding:5px 12px;border-radius:4px;font-weight:600">Desmotivacao e Abandono</span>
    <span style="color:#64748B;font-size:1rem">→</span>
    <span style="background:#FEE2E2;color:#991B1B;padding:5px 12px;border-radius:4px;font-weight:600">Evasao Definitiva</span>
    </div>
    <p style="margin:14px 0 0 0;color:#475569;font-size:0.88rem">
    <b>Como interpretar:</b> Um aluno que reprova permanece na mesma série por mais um ano,
    acumulando defasagem em relação à sua faixa etária — o que mede o TDI.
    Com o aumento da defasagem, cresce o sentimento de inadequação e a desmotivação,
    o que eleva a chance de abandono durante o ano letivo.
    O abandono reiterado resulta, eventualmente, na saída definitiva do sistema escolar — a evasão.
    <b>Interromper essa cadeia em qualquer etapa reduz o risco de evasão.</b>
    </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("")

    # ── Reprovação × Evasão ────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        titulo_pergunta("A reprovacao influencia a evasao escolar?",
                        "Cada ponto representa um ano. A linha tracejada mostra a tendência linear entre os dois indicadores.")
        xc, yc = "taxa_repetencia_em", "taxa_evasao_em"
        if xc in df_socio.columns and yc in df_socio.columns:
            s = df_socio.dropna(subset=[xc, yc])
            if len(s) >= 3:
                fig = go.Figure()
                scatter_tendencia(fig, s[xc].values, s[yc].values, "Ensino Médio", COR_EM, s["ano"].values)
                mx, my = s[xc].mean(), s[yc].mean()
                fig.add_hline(y=my, line_dash="dot", line_color=COR_CINZA, opacity=0.5,
                              annotation_text=f"Média de evasão: {my:.1f}%", annotation_position="right")
                fig.add_vline(x=mx, line_dash="dot", line_color=COR_CINZA, opacity=0.5,
                              annotation_text=f"Média de repetência: {mx:.1f}%", annotation_position="top")
                fig.update_layout(xaxis_title="Taxa de Repetência — EM (%)",
                                  yaxis_title="Taxa de Evasão — EM (%)",
                                  showlegend=False, height=380)
                st.plotly_chart(fig, use_container_width=True)
                r = np.corrcoef(s[xc].values, s[yc].values)[0, 1]
                slope = np.polyfit(s[xc].values, s[yc].values, 1)[0]
                caixa_destaque(
                    "atencao" if abs(r) > 0.5 else "neutro",
                    f"Correlacao de Pearson = {r:.2f} — Relacao {'forte' if abs(r) > 0.6 else 'moderada'}",
                    f"A correlação de {r:.2f} indica uma relação {'positiva e forte' if r > 0.6 else 'positiva moderada'} "
                    f"entre reprovação e evasão no EM. Isso significa que, historicamente, anos com mais reprovação "
                    f"tendem a ter mais evasão — e vice-versa. "
                    f"Tecnicamente, cada aumento de 1 ponto percentual na repetência está associado a um aumento "
                    f"de {slope:.2f} p.p. na evasão (p.p. = ponto percentual). "
                    "Reduzir a reprovação é, portanto, uma das estratégias mais eficazes para reduzir a evasão."
                )
            else:
                caixa_qualidade("Dados insuficientes para calcular a correlação no período selecionado. Amplie o intervalo de anos.")

    with col2:
        titulo_pergunta("A distorcao idade-serie (TDI) aumenta o abandono escolar?",
                        "Cada ponto representa um ano. Quanto maior a distorção, maior a tendência de abandono.")
        if all(c in df_int.columns for c in ["tdi_em", "taxa_abandono_em"]):
            s2 = df_int.dropna(subset=["tdi_em", "taxa_abandono_em"])
            if len(s2) >= 3:
                fig2 = go.Figure()
                scatter_tendencia(fig2, s2["tdi_em"].values, s2["taxa_abandono_em"].values,
                                  "Ensino Médio", COR_ABANDONO, s2["ano"].values)
                fig2.update_layout(xaxis_title="TDI — Taxa de Distorcao Idade-Serie EM (%)",
                                   yaxis_title="Taxa de Abandono — EM (%)",
                                   showlegend=False, height=380)
                st.plotly_chart(fig2, use_container_width=True)
                r2 = np.corrcoef(s2["tdi_em"].values, s2["taxa_abandono_em"].values)[0, 1]
                slope2 = np.polyfit(s2["tdi_em"].values, s2["taxa_abandono_em"].values, 1)[0]
                caixa_destaque(
                    "atencao" if abs(r2) > 0.5 else "neutro",
                    f"Correlacao de Pearson = {r2:.2f}",
                    f"A correlação entre TDI e abandono é de {r2:.2f}, indicando que "
                    f"anos com maior distorção idade-série tendem a apresentar mais abandono escolar. "
                    f"Cada ponto percentual a mais no TDI está associado a um aumento de {slope2:.2f} p.p. no abandono. "
                    "Isso confirma que alunos que ficaram 'para trás' na trajetória escolar têm maior tendência "
                    "de desistir. Programas de nivelamento e reforço reduzem o TDI e, consequentemente, o abandono."
                )
            else:
                caixa_qualidade("Dados insuficientes para o período selecionado.")

    caixa_acao(
        "Os dois gráficos acima mostram que a reprovação e a distorção idade-série são os fatores mais "
        "fortemente associados à evasão em Recife. Políticas que reduzam a reprovação — como a progressão "
        "continuada com suporte pedagógico — e que corrijam a distorção — como reforço escolar e nivelamento — "
        "atacam diretamente as causas da evasão."
    )

    st.markdown("---")

    # ── Diagnóstico fator a fator ──────────────────────────────────────────────
    titulo_pergunta("Quais indicadores estao criticos no periodo selecionado?",
                    "Avaliação fator a fator do último ano disponível no período. Cada indicador é comparado a um limiar de referência.")

    if not df_int.empty:
        ultimo = df_int.sort_values("ano").iloc[-1]
        ano_ref = int(ultimo["ano"])
        st.caption(f"Diagnóstico para o ano {ano_ref}. Limiares baseados em referências nacionais do INEP.")

        fatores = []
        def avaliar(col, label, lim_atenc, lim_crit, referencia=""):
            v = float(ultimo.get(col, np.nan) or np.nan)
            if pd.isna(v): return
            if v >= lim_crit:   nv, tipo = "Critico", "alerta"
            elif v >= lim_atenc: nv, tipo = "Atencao", "atencao"
            else:                nv, tipo = "Adequado", "positivo"
            fatores.append({"label": label, "valor": v, "nivel": nv, "tipo": tipo,
                             "lim_atenc": lim_atenc, "lim_crit": lim_crit, "ref": referencia})

        avaliar("taxa_evasao_em",     "Evasão EM",           5,  10, "Referência: até 5% é aceitável; acima de 10% é crítico.")
        avaliar("taxa_abandono_em",   "Abandono EM",         5,  10, "Referência: até 5% é aceitável; acima de 10% é crítico.")
        avaliar("tdi_em",             "TDI EM",             20,  30, "Referência: acima de 20% indica atenção; acima de 30% é crítico.")
        avaliar("taxa_repetencia_em", "Repetência EM",       8,  15, "Referência: acima de 8% merece atenção; acima de 15% é crítico.")
        avaliar("taxa_evasao_ef",     "Evasão EF",           3,   6, "Referência: até 3% é aceitável no EF.")
        avaliar("tdi_ef",             "TDI EF",             15,  25, "Referência: acima de 15% indica atenção.")

        ca, cb, cc = st.columns(3)
        crit = [f for f in fatores if f["nivel"] == "Critico"]
        atenc = [f for f in fatores if f["nivel"] == "Atencao"]
        adeq = [f for f in fatores if f["nivel"] == "Adequado"]

        with ca:
            st.markdown(f"**Indicadores Criticos (acima do limiar critico)**")
            if crit:
                for f in crit:
                    st.markdown(f"- **{f['label']}**: {f['valor']:.1f}% *(limiar: {f['lim_crit']}%)*")
                    st.caption(f['ref'])
            else:
                st.success("Nenhum indicador no nível crítico.")
        with cb:
            st.markdown(f"**Indicadores em Atencao**")
            if atenc:
                for f in atenc:
                    st.markdown(f"- **{f['label']}**: {f['valor']:.1f}% *(limiar: {f['lim_atenc']}%)*")
                    st.caption(f['ref'])
            else:
                st.success("Nenhum indicador em atenção.")
        with cc:
            st.markdown(f"**Indicadores Adequados**")
            if adeq:
                for f in adeq:
                    st.markdown(f"- **{f['label']}**: {f['valor']:.1f}%")
            else:
                st.warning("Nenhum indicador está dentro dos limiares aceitáveis.")

    st.markdown("---")

    # ── Matriz de correlação ───────────────────────────────────────────────────
    titulo_pergunta("Como os indicadores se relacionam entre si?",
                    "A matriz de correlação mostra a força e direção da relação entre os indicadores. "
                    "Verde = relação inversa (um sobe, o outro cai). Vermelho = relação direta (os dois sobem juntos).")

    cols_c = [c for c in [
        "taxa_evasao_em", "taxa_abandono_em", "taxa_repetencia_em", "taxa_reprovacao_em",
        "tdi_em", "atu_em", "taxa_aprovacao_em", "taxa_promocao_em",
        "taxa_evasao_ef", "taxa_abandono_ef", "tdi_ef",
    ] if c in df_int.columns]

    NOMES_AMIGAVEIS = {
        "taxa_evasao_em":     "Evasão EM", "taxa_abandono_em":    "Abandono EM",
        "taxa_repetencia_em": "Repetência EM","taxa_reprovacao_em":  "Reprovação EM",
        "tdi_em":             "TDI EM",    "atu_em":              "Alunos/Turma EM",
        "taxa_aprovacao_em":  "Aprovação EM","taxa_promocao_em":    "Promoção EM",
        "taxa_evasao_ef":     "Evasão EF", "taxa_abandono_ef":    "Abandono EF",
        "tdi_ef":             "TDI EF",
    }

    df_c = df_int[[c for c in cols_c if c in df_int.columns]].dropna(how="all")
    df_c = df_c.rename(columns=NOMES_AMIGAVEIS)

    if len(df_c) >= 3:
        corr = df_c.corr()
        col_c1, col_c2 = st.columns([3, 2])
        with col_c1:
            fig_corr = px.imshow(
                corr, color_continuous_scale="RdYlGn", zmin=-1, zmax=1,
                text_auto=".2f", aspect="auto",
            )
            fig_corr.update_layout(height=460)
            st.plotly_chart(fig_corr, use_container_width=True)

        with col_c2:
            st.markdown("**Como ler esta matriz:**")
            st.markdown("""
            - Cada célula mostra a correlação entre dois indicadores (escala de -1 a +1)
            - **Verde escuro**: relação inversa forte — quando um sobe, o outro cai  
            - **Vermelho escuro**: relação direta forte — ambos sobem ou caem juntos  
            - **Amarelo / branco**: pouca relação entre os indicadores  
            - Valores próximos de **+1 ou -1** indicam relação forte  
            - Valores próximos de **0** indicam pouca ou nenhuma relação linear
            """)

            st.markdown("**Principais achados:**")
            if "Aprovação EM" in corr.columns and "Evasão EM" in corr.columns:
                r_ap = corr.loc["Aprovação EM", "Evasão EM"]
                st.markdown(
                    f"- Aprovação EM × Evasão EM: **{r_ap:.2f}** — "
                    f"{'relação inversa forte: mais aprovação está associada a menos evasão' if r_ap < -0.5 else 'relação moderada'}"
                )
            if "TDI EM" in corr.columns and "Evasão EM" in corr.columns:
                r_tdi = corr.loc["TDI EM", "Evasão EM"]
                st.markdown(
                    f"- TDI EM × Evasão EM: **{r_tdi:.2f}** — "
                    f"{'relação direta forte: mais distorção está associada a mais evasão' if r_tdi > 0.5 else 'relação moderada'}"
                )
            if "Repetência EM" in corr.columns and "Evasão EM" in corr.columns:
                r_rep = corr.loc["Repetência EM", "Evasão EM"]
                st.markdown(
                    f"- Repetência EM × Evasão EM: **{r_rep:.2f}** — "
                    f"{'relação direta forte: mais reprovação está associada a mais evasão' if r_rep > 0.5 else 'relação moderada'}"
                )

        caixa_destaque("info", "Interpretação-chave da matriz",
            "A matriz confirma que os indicadores de desempenho positivo (aprovação, promoção) têm correlação "
            "negativa com a evasão — ou seja, escolas com mais aprovação tendem a ter menos evasão. "
            "Por outro lado, TDI e repetência têm correlação positiva com evasão e abandono. "
            "Isso valida a cadeia causal apresentada no início desta página.")
        caixa_qualidade(
            f"A matriz é calculada com {len(df_c)} pontos de dados (anos no período selecionado). "
            "Com menos de 10 pontos, as correlações devem ser interpretadas com cautela — elas indicam "
            "tendências, mas não têm significância estatística robusta. Para análises mais rigorosas, "
            "amplie o período ou incorpore dados individuais por escola."
        )
    else:
        caixa_qualidade("Dados insuficientes para calcular a matriz de correlação. Selecione um período maior.")


# ===========================================================================
# PÁGINA 4 — O QUE FAZER
# ===========================================================================

def pagina_acoes(dados: dict, filtros: dict):
    st.markdown('<h1>O que fazer? — Recomendações de Intervencao</h1>', unsafe_allow_html=True)
    st.caption(
        "Lista de ações prioritárias para reduzir a evasão escolar em Recife, "
        "ordenadas por urgência e baseadas nos fatores de risco identificados nos dados."
    )
    st.markdown("---")

    a1, a2 = filtros["ano_range"]
    df_int = dados["fato_integrado"].copy()
    df_int = df_int[(df_int["ano"] >= a1) & (df_int["ano"] <= a2)]
    ultimo = df_int.sort_values("ano").iloc[-1] if not df_int.empty else None

    if ultimo is not None:
        sc  = float(ultimo.get("indice_risco_evasao", 0) or 0)
        ev  = float(ultimo.get("taxa_evasao_em",       0) or 0)
        ab  = float(ultimo.get("taxa_abandono_em",     0) or 0)
        tdi = float(ultimo.get("tdi_em",               0) or 0)
        rep = float(ultimo.get("taxa_repetencia_em",   0) or 0)
        atu = float(ultimo.get("atu_em",               0) or 0)
        niv = classificar_risco(pd.Series([sc])).iloc[0]

        caixa_destaque("info", f"Diagnostico para o ano {int(ultimo['ano'])} — Score de Risco: {sc:.1f} ({niv})",
            f"Evasão EM: {ev:.1f}% | Abandono EM: {ab:.1f}% | TDI EM: {tdi:.1f}% | "
            f"Repetência EM: {rep:.1f}% | Alunos por turma EM: {atu:.0f}. "
            "As ações abaixo foram priorizadas com base nesses valores.")
    else:
        sc = ev = ab = tdi = rep = atu = 0

    st.markdown("---")
    st.markdown("#### Plano de Intervencao Prioritizado")
    st.caption("As ações estão organizadas por urgência: Imediata (implementar agora), Curto prazo (nos próximos meses), Médio e Longo prazo.")

    URGENCIAS = {
        "IMEDIATA":     ("#FEF2F2", "#991B1B"),
        "CURTO PRAZO":  ("#FFFBEB", "#B45309"),
        "MEDIO PRAZO":  ("#EFF6FF", "#1D4ED8"),
        "LONGO PRAZO":  ("#F0FDF4", "#15803D"),
    }

    acoes = [
        {
            "urgencia": "IMEDIATA",
            "titulo": "Implantar monitoramento semanal de frequencia",
            "descricao":
                "Alunos com mais de 25% de faltas têm risco elevado de abandono. "
                "O monitoramento semanal permite identificar esses alunos antes que abandonem "
                "e acionar a família e os orientadores educacionais a tempo. "
                "Ferramentas simples — como planilhas ou aplicativos de controle de frequência — "
                "já são suficientes para iniciar essa prática.",
            "impacto": "Pode reduzir o abandono em até 30%, segundo estudos nacionais do INEP.",
            "gatilho": f"Abandono EM atual: {ab:.1f}%",
        },
        {
            "urgencia": "IMEDIATA",
            "titulo": "Nivelamento e reforcamento para alunos com distorcao idade-serie",
            "descricao":
                "Alunos com TDI elevado — isto é, cursando uma série muito abaixo da esperada para sua idade — "
                "têm probabilidade significativamente maior de abandonar a escola. "
                "Aulas de nivelamento em contraturno, com foco em competências básicas, "
                "reduzem a defasagem e aumentam a autoestima e o engajamento desses alunos.",
            "impacto": "Redução do TDI e, consequentemente, do abandono.",
            "gatilho": f"TDI EM atual: {tdi:.1f}% (referência aceitável: abaixo de 20%)",
        },
        {
            "urgencia": "CURTO PRAZO",
            "titulo": "Revisar a politica de reprovacao — progressao com suporte",
            "descricao":
                "A reprovação é o maior preditor de evasão neste conjunto de dados (veja a página 'Por que ocorre?'). "
                "Substituir a reprovação automática por progressão com apoio pedagógico intensivo "
                "interrompe a cadeia reprovação → TDI → abandono → evasão. "
                "Isso não significa aprovar sem critérios, mas garantir suporte adicional ao invés de reprovar.",
            "impacto": "Interrompe a principal cadeia causal da evasão.",
            "gatilho": f"Repetência EM atual: {rep:.1f}% (referência aceitável: abaixo de 8%)",
        },
        {
            "urgencia": "CURTO PRAZO",
            "titulo": "Reduzir o numero de alunos por turma no Ensino Medio",
            "descricao":
                "Turmas com mais de 35 alunos dificultam o acompanhamento individualizado, "
                "reduzem o vínculo professor-aluno e estão associadas a maiores taxas de abandono. "
                "A meta deve ser turmas de no máximo 30 alunos no EM. "
                "Isso pode ser alcançado com a abertura de novas turmas ou redistribuição de matrículas.",
            "impacto": "Melhora na qualidade do ensino e no engajamento dos alunos.",
            "gatilho": f"Alunos por turma EM atual: {atu:.0f} (referência: até 30)",
        },
        {
            "urgencia": "MEDIO PRAZO",
            "titulo": "Ampliar a oferta de EJA e Ensino Medio noturno",
            "descricao":
                "Parte dos estudantes que evadiram o sistema escolar precisa de modalidades flexíveis "
                "para retornar. A Educação de Jovens e Adultos (EJA) e o Ensino Médio noturno "
                "atendem trabalhadores e pessoas que não conseguem frequentar a escola no período diurno. "
                "Ampliar essas modalidades reduz a evasão permanente.",
            "impacto": "Reintegração de evadidos ao sistema educacional.",
            "gatilho": "Evasão acumulada ao longo dos anos cria uma população fora do sistema.",
        },
        {
            "urgencia": "MEDIO PRAZO",
            "titulo": "Programas de apoio socioemocional e reducao de barreiras externas",
            "descricao":
                "Parte da evasão tem causas externas: pobreza, necessidade de trabalhar, violência, "
                "distância das escolas. Bolsas-estudo condicionadas à frequência, auxílio-transporte "
                "e apoio psicológico reduzem o impacto desses fatores e aumentam a permanência "
                "especialmente em populações vulneráveis.",
            "impacto": "Redução da evasão por causas socioeconômicas externas.",
            "gatilho": "Evasão está correlacionada com vulnerabilidade social.",
        },
        {
            "urgencia": "LONGO PRAZO",
            "titulo": "Dashboard com atualizacao mensal por escola",
            "descricao":
                "O monitoramento contínuo é mais eficaz do que ações pontuais. "
                "Um sistema de indicadores atualizado mensalmente — com dados de frequência, "
                "desempenho e perfil dos alunos — permite identificar problemas antes que se agravem "
                "e direcionar intervenções de forma precisa para as escolas e turmas mais vulneráveis.",
            "impacto": "Prevenção ao invés de remediação — mais barato e mais eficaz.",
            "gatilho": "Identificação tardia de problemas aumenta o custo de intervenção.",
        },
        {
            "urgencia": "LONGO PRAZO",
            "titulo": "Analise geografica da evasao por bairro e regiao",
            "descricao":
                "Cruzar os dados educacionais com informações socioeconômicas do IBGE por bairro e "
                "Região Político-Administrativa (RPA) de Recife permitiria identificar quais regiões "
                "concentram mais evasão e direcionar recursos públicos com maior precisão e eficiência.",
            "impacto": "Alocação eficiente de investimentos públicos em educação.",
            "gatilho": "Análise geoespacial ainda não realizada — dados disponíveis no IBGE.",
        },
    ]

    for ac in acoes:
        bg, borda = URGENCIAS[ac["urgencia"]]
        st.markdown(
            f"""<div style="background:{bg};border-left:5px solid {borda};
            padding:16px 20px;border-radius:4px;margin-bottom:14px;">
            <span style="background:{borda};color:white;padding:2px 10px;border-radius:3px;
            font-size:0.75rem;font-weight:700;letter-spacing:0.05em">{ac['urgencia']}</span>
            <h3 style="margin:8px 0 6px 0;color:#1E293B;font-size:0.98rem">{ac['titulo']}</h3>
            <p style="margin:0 0 8px 0;color:#374151;font-size:0.89rem;line-height:1.6">{ac['descricao']}</p>
            <p style="margin:0;color:#6B7280;font-size:0.82rem">
            <b>Impacto esperado:</b> {ac['impacto']} &nbsp;|&nbsp;
            <b>Indicador de alerta:</b> {ac['gatilho']}
            </p></div>""",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("#### Resumo das Acoes")
    st.caption("Visão consolidada para uso em apresentações ou relatórios.")
    df_acoes = pd.DataFrame([
        {"Urgencia": a["urgencia"], "Acao": a["titulo"], "Impacto Esperado": a["impacto"]}
        for a in acoes
    ])
    st.dataframe(df_acoes, use_container_width=True, hide_index=True)


# ===========================================================================
# PÁGINA 5 — EVOLUÇÃO HISTÓRICA
# ===========================================================================

def pagina_temporal(dados: dict, filtros: dict):
    st.markdown('<h1>Como a evasão evoluiu? — Analise Historica</h1>', unsafe_allow_html=True)
    st.caption(
        "Análise da evolução dos indicadores de evasão ao longo do tempo. "
        "Identifique tendências de melhora, piora e o impacto de eventos como a pandemia de COVID-19."
    )
    st.markdown("---")

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
    show_em = "Ensino Médio (EM)"        in filtros["nivel"]

    if a1 <= 2020 <= a2:
        caixa_destaque("atencao", "Impacto da Pandemia de COVID-19 (2020–2022)",
            "O período 2020–2022 foi marcado pela pandemia de COVID-19, que causou o fechamento das escolas "
            "e a transição para o ensino remoto emergencial. Esse contexto afetou diretamente os indicadores: "
            "em 2020, muitas escolas não coletaram dados de aprovação/reprovação (o que explica valores ausentes). "
            "Em 2021 e 2022, houve aumento no abandono e na dificuldade de aprendizagem acumulada. "
            "Ao analisar esse período, leve em conta que parte dos resultados reflete o impacto da pandemia, "
            "não necessariamente uma falha estrutural das políticas educacionais.")

    # ── Evasão e abandono histórico ────────────────────────────────────────────
    titulo_pergunta("Como a evasao e o abandono evoluiram ao longo do tempo?",
                    "Linha contínua = Evasão (saída definitiva). Linha tracejada = Abandono (saída durante o ano letivo). "
                    "As linhas azuis referem-se ao Ensino Fundamental e as vermelhas ao Ensino Médio.")

    fig = go.Figure()
    if a1 <= 2020 <= a2:
        fig.add_vrect(x0=2019.5, x1=2022.5, fillcolor="#FEF9C3", opacity=0.3,
                      line_width=0, annotation_text="Pandemia (2020–2022)",
                      annotation_position="top left",
                      annotation_font=dict(size=11, color="#92400E"))

    for nivel, show, c_ev, c_ab, nome in [
        ("ef", show_ef, COR_EF,      "#93C5FD", "EF"),
        ("em", show_em, COR_EM,      "#FCA5A5", "EM"),
    ]:
        if not show: continue
        s_s = df_soc.dropna(subset=[f"taxa_evasao_{nivel}"])
        s_e = df_educ.dropna(subset=[f"taxa_abandono_{nivel}"])
        if not s_s.empty:
            fig.add_trace(go.Scatter(
                x=s_s["ano"], y=s_s[f"taxa_evasao_{nivel}"],
                name=f"Evasão {nome}", mode="lines+markers",
                line=dict(color=c_ev, width=3), marker=dict(size=8),
            ))
        if not s_e.empty:
            fig.add_trace(go.Scatter(
                x=s_e["ano"], y=s_e[f"taxa_abandono_{nivel}"],
                name=f"Abandono {nome}", mode="lines+markers",
                line=dict(color=c_ab, width=2, dash="dot"), marker=dict(size=7),
            ))

    fig.update_layout(yaxis_title="Taxa (%)", xaxis_title="Ano",
                      hovermode="x unified", height=420,
                      legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, use_container_width=True)

    # Análise automática da tendência
    if "taxa_evasao_em" in df_soc.columns:
        s_em = df_soc.dropna(subset=["taxa_evasao_em"]).sort_values("ano")
        if len(s_em) >= 3:
            v_ini = s_em.iloc[0]["taxa_evasao_em"]
            v_fim = s_em.iloc[-1]["taxa_evasao_em"]
            delta = v_fim - v_ini
            trend = "queda" if delta < -1 else ("alta" if delta > 1 else "estável")
            caixa_destaque(
                "positivo" if delta < 0 else "atencao",
                f"Tendencia no periodo: {trend} na evasao do Ensino Medio",
                f"A evasão no EM passou de {v_ini:.1f}% em {int(s_em.iloc[0]['ano'])} "
                f"para {v_fim:.1f}% em {int(s_em.iloc[-1]['ano'])} — "
                f"uma variação de {delta:+.1f} p.p. (p.p. = ponto percentual). "
                f"{'Essa redução indica progresso nas políticas educacionais do município.' if delta < 0 else 'Esse aumento merece atenção e investigação das causas.'} "
                "Lembre-se que a pandemia pode ter distorcido os dados de 2020–2022."
            )

    caixa_acao(
        "Use o gráfico para identificar em quais anos a situação piorou e correlacionar com eventos históricos "
        "(mudanças de gestão, políticas implementadas, pandemia). "
        "Anos com variação positiva (piora) indicam momentos em que as políticas falharam ou não foram suficientes."
    )

    st.markdown("---")

    # ── Variação YoY ──────────────────────────────────────────────────────────
    titulo_pergunta("Em quais anos a evasao no Ensino Medio piorou ou melhorou?",
                    "Cada barra mostra a variação percentual da evasão EM em relação ao ano anterior. "
                    "Barras em vermelho indicam piora (evasão aumentou). Barras em verde indicam melhora.")

    if "var_taxa_evasao_em" in tend.columns:
        s = tend.dropna(subset=["var_taxa_evasao_em"])
        if not s.empty:
            # Identifica o maior aumento e a maior queda
            i_max = s["var_taxa_evasao_em"].idxmax()
            i_min = s["var_taxa_evasao_em"].idxmin()
            ano_pior   = int(s.loc[i_max, "ano"])
            var_pior   = s.loc[i_max, "var_taxa_evasao_em"]
            ano_melhor = int(s.loc[i_min, "ano"])
            var_melhor = s.loc[i_min, "var_taxa_evasao_em"]

            fig_y = go.Figure(go.Bar(
                x=s["ano"], y=s["var_taxa_evasao_em"],
                marker_color=[COR_EM if v > 0 else COR_POSITIVO for v in s["var_taxa_evasao_em"]],
                text=s["var_taxa_evasao_em"].apply(lambda x: f"{x:+.1f}%"),
                textposition="outside",
            ))
            fig_y.add_hline(y=0, line_color=COR_CINZA, line_width=1.5)
            # Anota pior e melhor
            fig_y.add_annotation(x=ano_pior, y=var_pior, text=f"Pior: {ano_pior}",
                                  showarrow=True, arrowhead=2, ax=0, ay=-30, font=dict(color=COR_EM, size=11))
            fig_y.add_annotation(x=ano_melhor, y=var_melhor, text=f"Melhor: {ano_melhor}",
                                  showarrow=True, arrowhead=2, ax=0, ay=30, font=dict(color=COR_POSITIVO, size=11))
            fig_y.update_layout(yaxis_title="Variação em relação ao ano anterior (%)", xaxis_title="Ano", height=360)
            st.plotly_chart(fig_y, use_container_width=True)
            st.caption(
                f"Maior piora: {ano_pior} (+{var_pior:.1f}%). "
                f"Maior melhora: {ano_melhor} ({var_melhor:.1f}%). "
                "A variação é calculada em relação ao ano imediatamente anterior — não ao início do período. "
                "Valores positivos significam que a evasão cresceu em relação ao ano anterior."
            )

    st.markdown("---")

    # ── EF vs EM ───────────────────────────────────────────────────────────────
    titulo_pergunta("O Ensino Medio e realmente mais critico que o Ensino Fundamental?",
                    "As barras mostram as taxas de evasão de EF e EM a cada ano. A linha mostra quantas vezes a evasão no EM é maior do que no EF.")

    if all(c in df_int.columns for c in ["taxa_evasao_ef", "taxa_evasao_em"]):
        dc = df_int.dropna(subset=["taxa_evasao_ef", "taxa_evasao_em"]).sort_values("ano").copy()
        dc = dc[dc["taxa_evasao_ef"] > 0]
        if not dc.empty:
            dc["razao"] = (dc["taxa_evasao_em"] / dc["taxa_evasao_ef"]).round(2)
            fig_c = make_subplots(specs=[[{"secondary_y": True}]])
            fig_c.add_trace(go.Bar(x=dc["ano"], y=dc["taxa_evasao_ef"],
                                   name="Evasão EF (%)", marker_color=COR_EF, opacity=0.85), secondary_y=False)
            fig_c.add_trace(go.Bar(x=dc["ano"], y=dc["taxa_evasao_em"],
                                   name="Evasão EM (%)", marker_color=COR_EM, opacity=0.85), secondary_y=False)
            fig_c.add_trace(go.Scatter(x=dc["ano"], y=dc["razao"], name="Razão EM ÷ EF",
                                       mode="lines+markers",
                                       line=dict(color=COR_PRIMARIA, width=2, dash="dot"),
                                       marker=dict(size=7)), secondary_y=True)
            fig_c.update_yaxes(title_text="Taxa de Evasão (%)", secondary_y=False)
            fig_c.update_yaxes(title_text="Razão EM ÷ EF (vezes)", secondary_y=True)
            fig_c.update_layout(barmode="group", hovermode="x unified", height=420,
                                legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig_c, use_container_width=True)

            media_razao = dc["razao"].mean()
            max_razao   = dc["razao"].max()
            ano_max_r   = int(dc.loc[dc["razao"].idxmax(), "ano"])
            caixa_destaque("info",
                f"Confirmado: a evasao no EM e consistentemente maior que no EF",
                f"Em média, a evasão no Ensino Médio foi {media_razao:.1f} vezes maior que no Ensino Fundamental "
                f"no período analisado. O pico dessa diferença foi em {ano_max_r}, quando a evasão no EM foi "
                f"{max_razao:.1f} vezes a do EF. "
                "Isso confirma que o Ensino Médio deve ser o foco prioritário das políticas de combate à evasão."
            )
        caixa_acao(
            "Ao planejar intervenções, priorize o Ensino Médio — especialmente o 1º ano do EM, "
            "que concentra a maior parte do abandono. A transição do EF para o EM é um momento crítico "
            "onde muitos alunos se perdem."
        )

    st.markdown("---")

    # ── Boxplot por período ────────────────────────────────────────────────────
    titulo_pergunta("Como a evasao se distribuiu em diferentes periodos historicos?",
                    "O gráfico de caixas (boxplot) mostra a distribuição dos valores em cada período. "
                    "A linha central é a mediana — metade dos valores estão acima, metade abaixo. "
                    "Os pontos fora da caixa são valores atípicos (outliers).")

    col1, col2 = st.columns(2)
    for col, base, col_v, nome_g in [
        (col1, dados["fato_socioeconomico"], "taxa_evasao_em",   "Taxa de Evasão — Ensino Médio"),
        (col2, dados["fato_educacional"],    "taxa_abandono_em", "Taxa de Abandono — Ensino Médio"),
    ]:
        with col:
            s = base.dropna(subset=[col_v])
            s = s[(s["ano"] >= a1) & (s["ano"] <= a2)]
            if s.empty: continue
            ordem = ["2006–2010", "2011–2015", "2016–2019", "2020–2022 (Pandemia)", "2023–2024"]
            ordem_ok = [p for p in ordem if p in s["periodo"].unique()]
            fig_b = px.box(
                s, x="periodo", y=col_v,
                category_orders={"periodo": ordem_ok},
                color="periodo", color_discrete_map=PALETA_PERIODO,
                labels={"periodo": "Período", col_v: "%"},
                points="all",
            )
            fig_b.update_layout(showlegend=False, height=400, title=nome_g)
            st.plotly_chart(fig_b, use_container_width=True)

    st.caption(
        "Como ler o boxplot: A caixa representa onde estão os 50% centrais dos valores. "
        "A linha dentro da caixa é a mediana. Os pontos individuais mostram cada registro. "
        "Períodos com caixas mais altas indicam maior nível médio de evasão/abandono. "
        "Caixas mais largas indicam maior variação entre os registros naquele período."
    )
    caixa_qualidade(
        "Os dados de 2023–2024 disponíveis nesta base são parciais e apresentam inconsistências, "
        "o que pode resultar em valores atípicos. Interprete os resultados desse período com cautela."
    )


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    garantir_dados()
    dados = carregar_dados()

    if not dados:
        st.error("Não foi possível carregar os dados processados. Execute o ETL primeiro.")
        st.stop()

    filtros = sidebar(dados)

    paginas = {
        "Painel de Indicadores":      pagina_painel,
        "Onde está o risco?":         pagina_onde,
        "Por que ocorre?":            pagina_causas,
        "O que fazer?":               pagina_acoes,
        "Como a evasão evoluiu?":     pagina_temporal,
    }

    st.sidebar.markdown('<hr style="border:none;border-top:1px solid #E2E8F0;margin:10px 0">', unsafe_allow_html=True)
    pagina_atual = st.sidebar.radio("Navegação", list(paginas.keys()))
    paginas[pagina_atual](dados, filtros)

    st.sidebar.markdown('<hr style="border:none;border-top:1px solid #E2E8F0;margin:10px 0">', unsafe_allow_html=True)
    st.sidebar.caption("Projeto: Análise de Evasão Escolar | Recife | INEP/MEC")


if __name__ == "__main__":
    main()
