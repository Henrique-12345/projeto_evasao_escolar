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
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"

# ---------------------------------------------------------------------------
# Paleta e limiares
# ---------------------------------------------------------------------------
COR_EF      = "#3B82F6"
COR_EM      = "#EF4444"
COR_WARN    = "#F59E0B"
COR_OK      = "#10B981"
COR_CRITICO = "#DC2626"
COR_ALTO    = "#EF4444"
COR_MEDIO   = "#F59E0B"
COR_BAIXO   = "#10B981"

PALETA_PERIODO = {
    "2006–2010":             "#94A3B8",
    "2011–2015":             "#60A5FA",
    "2016–2019":             "#34D399",
    "2020–2022 (Pandemia)":  "#FBBF24",
    "2023–2024":             "#F87171",
}

# Score: ≤20 Baixo · 20-35 Moderado · 35-50 Alto · >50 Crítico
LIMIARES = {"baixo": 20, "moderado": 35, "alto": 50}

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
# Helpers: Score de Risco 0–100
# ---------------------------------------------------------------------------

def calcular_score(df: pd.DataFrame) -> pd.Series:
    """
    Score de risco 0–100 por registro.
      abandono_em (40%) + tdi_em (30%) + reprovacao_em (30%)
    Usa EF como fallback quando EM não está disponível.
    """
    def col_ou_zero(nome):
        return df[nome].fillna(0) if nome in df.columns else pd.Series(0.0, index=df.index)

    score = pd.Series(np.nan, index=df.index)

    mask_em = df["taxa_abandono_em"].notna() if "taxa_abandono_em" in df.columns else pd.Series(False, index=df.index)
    if mask_em.any():
        s = (col_ou_zero("taxa_abandono_em") * 0.40
             + col_ou_zero("tdi_em")           * 0.30
             + col_ou_zero("taxa_reprovacao_em") * 0.30)
        score[mask_em] = s[mask_em].clip(0, 100)

    mask_ef = score.isna() & (df["taxa_abandono_ef"].notna() if "taxa_abandono_ef" in df.columns else pd.Series(False, index=df.index))
    if mask_ef.any():
        s = (col_ou_zero("taxa_abandono_ef") * 0.40
             + col_ou_zero("tdi_ef")          * 0.30
             + col_ou_zero("taxa_reprovacao_ef") * 0.30)
        score[mask_ef] = s[mask_ef].clip(0, 100)

    return score.round(1)


def classificar_risco(score: pd.Series) -> pd.Series:
    return pd.cut(
        score,
        bins=[-np.inf, LIMIARES["baixo"], LIMIARES["moderado"], LIMIARES["alto"], np.inf],
        labels=["🟢 Baixo", "🟡 Moderado", "🔴 Alto", "🚨 Crítico"],
    ).astype(str)


def cor_risco(nivel: str) -> str:
    if "Crítico" in nivel: return COR_CRITICO
    if "Alto"    in nivel: return COR_ALTO
    if "Moderado" in nivel: return COR_MEDIO
    return COR_BAIXO


# ---------------------------------------------------------------------------
# Helpers visuais
# ---------------------------------------------------------------------------

def hex_rgba(hex_color: str, alpha: float = 0.15) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def scatter_tendencia(fig: go.Figure, x, y, name: str, cor: str, texts=None) -> go.Figure:
    """Scatter + linha de tendência via numpy."""
    mask = ~(np.isnan(x) | np.isnan(y))
    xc, yc = np.array(x)[mask], np.array(y)[mask]
    fig.add_trace(go.Scatter(
        x=xc, y=yc,
        mode="markers+text" if texts is not None else "markers",
        name=name,
        text=np.array(texts)[mask] if texts is not None else None,
        textposition="top center",
        marker=dict(color=cor, size=10, line=dict(color="white", width=1)),
    ))
    if len(xc) >= 2:
        z = np.polyfit(xc, yc, 1)
        xl = np.linspace(xc.min(), xc.max(), 100)
        fig.add_trace(go.Scatter(
            x=xl, y=np.poly1d(z)(xl),
            mode="lines", line=dict(color=cor, width=2, dash="dash"),
            showlegend=False,
        ))
    return fig


def alerta(tipo: str, titulo: str, texto: str):
    """Bloco de alerta com borda colorida."""
    estilos = {
        "critico": ("#FEE2E2", "#DC2626", "🚨"),
        "alto":    ("#FEF3C7", "#D97706", "⚠️"),
        "ok":      ("#D1FAE5", "#065F46", "✅"),
        "info":    ("#DBEAFE", "#1D4ED8", "ℹ️"),
    }
    bg, borda, emoji = estilos.get(tipo, estilos["info"])
    st.markdown(
        f"""<div style="background:{bg};border-left:5px solid {borda};
        padding:12px 16px;border-radius:6px;margin-bottom:10px;">
        <b style="color:{borda}">{emoji} {titulo}</b><br>
        <span style="color:#374151;font-size:0.9rem">{texto}</span></div>""",
        unsafe_allow_html=True,
    )


def insight(texto: str):
    """Bloco de insight analítico em azul claro."""
    st.markdown(
        f"""<div style="background:#F0F9FF;border-left:4px solid #0284C7;
        padding:10px 14px;border-radius:6px;margin:10px 0 14px 0;">
        <span style="color:#0C4A6E;font-size:0.88rem">💡 {texto}</span></div>""",
        unsafe_allow_html=True,
    )


def secao(titulo: str, subtexto: str = "", icone: str = "📊"):
    st.markdown(f"### {icone} {titulo}")
    if subtexto:
        st.caption(subtexto)
    st.markdown("---")


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------

def sidebar(dados: dict) -> dict:
    try:
        st.sidebar.image(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/Brasão_do_Recife.svg/200px-Brasão_do_Recife.svg.png",
            width=65,
        )
    except Exception:
        pass

    st.sidebar.title("🎓 Evasão Escolar")
    st.sidebar.caption("Recife — EF e EM · INEP/MEC")
    st.sidebar.markdown("---")

    anos = sorted(dados["fato_integrado"]["ano"].unique())
    a_min, a_max = int(min(anos)), int(max(anos))
    filtros = {}

    filtros["ano_range"] = st.sidebar.slider(
        "📅 Período de Análise",
        min_value=a_min, max_value=a_max, value=(a_min, a_max),
    )
    filtros["nivel"] = st.sidebar.multiselect(
        "🏫 Nível de Ensino",
        ["Ensino Fundamental (EF)", "Ensino Médio (EM)"],
        default=["Ensino Fundamental (EF)", "Ensino Médio (EM)"],
    )
    filtros["risco_filtro"] = st.sidebar.multiselect(
        "🎯 Filtrar por Nível de Risco",
        ["🟢 Baixo", "🟡 Moderado", "🔴 Alto", "🚨 Crítico"],
        default=["🔴 Alto", "🚨 Crítico"],
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Legenda do Score de Risco**")
    st.sidebar.markdown(
        "🚨 **Crítico** → Score > 50  \n"
        "🔴 **Alto** → Score 35–50  \n"
        "🟡 **Moderado** → Score 20–35  \n"
        "🟢 **Baixo** → Score ≤ 20"
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("Score = Abandono EM (40%) + TDI EM (30%) + Reprovação EM (30%)")

    if st.sidebar.button("🔄 Reprocessar ETL"):
        st.cache_data.clear()
        sys.path.insert(0, str(ROOT))
        from etl.etl_pipeline import run_etl
        run_etl()
        st.rerun()

    return filtros


# ===========================================================================
# PÁGINA 1 — PAINEL DE ALERTAS
# ===========================================================================

def pagina_alertas(dados: dict, filtros: dict):
    st.title("🏠 Painel de Alertas")
    st.caption(
        "Visão executiva com indicadores-chave, alertas automáticos e score de risco de evasão. "
        "Use esta página como ponto de partida para identificar se há situações críticas."
    )
    st.markdown("---")

    a1, a2 = filtros["ano_range"]
    df_int  = dados["fato_integrado"].copy()
    df_educ = dados["fato_educacional"].copy()
    df_int  = df_int[(df_int["ano"]  >= a1) & (df_int["ano"]  <= a2)]
    df_educ = df_educ[(df_educ["ano"] >= a1) & (df_educ["ano"] <= a2)]

    if df_int.empty:
        st.warning("Sem dados para o período selecionado.")
        return

    df_int_s   = df_int.sort_values("ano")
    ultimo     = df_int_s.iloc[-1]
    penultimo  = df_int_s.iloc[-2] if len(df_int_s) > 1 else ultimo

    def safe_val(col, default=np.nan):
        v = ultimo.get(col, default)
        return float(v) if pd.notna(v) else default

    def delta_pp(col):
        v1, v2 = safe_val(col), float(penultimo.get(col, np.nan) or np.nan)
        if pd.isna(v1) or pd.isna(v2): return None
        d = round(v1 - v2, 2)
        return f"{d:+.1f}pp vs. {int(penultimo['ano'])}"

    # ── KPIs ──────────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("📉 Evasão EF", f"{safe_val('taxa_evasao_ef'):.1f}%" if pd.notna(safe_val('taxa_evasao_ef')) else "–",
                  delta_pp("taxa_evasao_ef"), help="Taxa de evasão no EF (saída definitiva do sistema)")
    with c2:
        st.metric("📉 Evasão EM", f"{safe_val('taxa_evasao_em'):.1f}%" if pd.notna(safe_val('taxa_evasao_em')) else "–",
                  delta_pp("taxa_evasao_em"), help="O EM é 2–3× mais crítico que o EF em todas as métricas")
    with c3:
        st.metric("🚪 Abandono EM", f"{safe_val('taxa_abandono_em'):.1f}%" if pd.notna(safe_val('taxa_abandono_em')) else "–",
                  delta_pp("taxa_abandono_em"), help="Abandono = saída no meio do ano. É o precursor imediato da evasão")
    with c4:
        st.metric("📐 TDI EM", f"{safe_val('tdi_em'):.1f}%" if pd.notna(safe_val('tdi_em')) else "–",
                  delta_pp("tdi_em"), help="Distorção Idade-Série: % de alunos com mais de 2 anos de atraso")
    with c5:
        score_val = safe_val("indice_risco_evasao")
        nivel_score = classificar_risco(pd.Series([score_val if pd.notna(score_val) else 0])).iloc[0]
        st.metric("🎯 Score de Risco", f"{score_val:.1f}" if pd.notna(score_val) else "–",
                  delta_pp("indice_risco_evasao"), help="Score 0–100: Evasão EM (40%) + TDI EM (30%) + Repetência EM (30%)")

    st.markdown("---")

    # ── Alertas automáticos ────────────────────────────────────────────────────
    st.subheader("🚨 Alertas Automáticos")
    st.caption("Gerados automaticamente a partir dos thresholds definidos nos indicadores.")

    evasao_em   = safe_val("taxa_evasao_em", 0)
    abandono_em = safe_val("taxa_abandono_em", 0)
    tdi_em      = safe_val("tdi_em", 0)
    score       = score_val if pd.notna(score_val) else 0
    score_ant   = float(penultimo.get("indice_risco_evasao", score) or score)
    delta_score = score - score_ant
    n_alertas   = 0

    if score > LIMIARES["alto"]:
        alerta("critico", f"Score de Risco Crítico — {score:.1f} pontos",
               f"Score acima de {LIMIARES['alto']} indica situação crítica no EM. "
               f"Evasão EM: {evasao_em:.1f}% · Abandono EM: {abandono_em:.1f}% · TDI EM: {tdi_em:.1f}%.")
        n_alertas += 1
    elif score > LIMIARES["moderado"]:
        alerta("alto", f"Score de Risco Elevado — {score:.1f} pontos",
               f"Indicadores acima da média. Atenção especial ao EM. "
               f"Evasão EM: {evasao_em:.1f}% · Abandono EM: {abandono_em:.1f}%.")
        n_alertas += 1

    if delta_score > 2:
        alerta("alto", f"Score em Crescimento ({delta_score:+.1f} pts vs. {int(penultimo['ano'])})",
               "O risco piorou em relação ao período anterior — tendência de alta merece atenção imediata.")
        n_alertas += 1
    elif delta_score < -2:
        alerta("ok", f"Melhora no Score de Risco ({delta_score:+.1f} pts vs. {int(penultimo['ano'])})",
               "Tendência positiva — os indicadores de evasão melhoraram em relação ao período anterior.")
        n_alertas += 1

    if tdi_em > 25:
        alerta("alto", f"Distorção Idade-Série elevada — TDI EM: {tdi_em:.1f}%",
               "Mais de 1 em cada 4 alunos do EM está fora da faixa etária esperada. "
               "A TDI é um dos maiores preditores de abandono e deve ser combatida com nivelamento.")
        n_alertas += 1

    if evasao_em > 10:
        alerta("critico", f"Evasão no EM acima do limiar crítico — {evasao_em:.1f}%",
               "Taxa de evasão acima de 10% requer intervenção imediata da gestão escolar e da Secretaria de Educação.")
        n_alertas += 1

    # Conta registros em risco alto/crítico no período
    df_educ_s = df_educ.copy()
    df_educ_s["score"] = calcular_score(df_educ_s)
    df_educ_s["nivel"] = classificar_risco(df_educ_s["score"])
    n_criticos = df_educ_s["nivel"].str.contains("Alto|Crítico", na=False).sum()

    if n_criticos > 0:
        alerta("alto" if n_criticos < 15 else "critico",
               f"{n_criticos} registros em nível Alto ou Crítico no período",
               f"De {len(df_educ_s)} registros analisados, {n_criticos} apresentam score elevado. "
               "Acesse 'Onde Está o Problema' para o ranking detalhado.")
        n_alertas += 1

    if n_alertas == 0:
        alerta("ok", "Nenhum alerta crítico no período selecionado",
               "Os indicadores estão dentro dos limiares aceitáveis. Continue monitorando para identificar tendências emergentes.")

    st.markdown("---")

    # ── Gauge + Tendência do score ─────────────────────────────────────────────
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("🎯 Score de Risco Atual")
        st.caption(f"Nível: **{nivel_score}** · Referência: {int(ultimo['ano'])}")
        if pd.notna(score_val):
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=float(score_val),
                delta={"reference": score_ant, "valueformat": ".1f"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": cor_risco(nivel_score)},
                    "steps": [
                        {"range": [0,  20],  "color": "#D1FAE5"},
                        {"range": [20, 35],  "color": "#FEF3C7"},
                        {"range": [35, 50],  "color": "#FEE2E2"},
                        {"range": [50, 100], "color": "#FECACA"},
                    ],
                },
                title={"text": nivel_score},
            ))
            fig_g.update_layout(height=280, margin=dict(t=40, b=0, l=20, r=20))
            st.plotly_chart(fig_g, use_container_width=True)
            insight(f"Score <b>{score_val:.1f}</b> → nível <b>{nivel_score}</b>. "
                    "Scores acima de 35 requerem atenção da gestão. "
                    "Acima de 50 indicam situação crítica.")

    with col2:
        st.subheader("📈 Evolução do Score de Risco ao Longo do Tempo")
        st.caption("Faixas coloridas indicam os limiares de classificação de risco.")
        if "indice_risco_evasao" in df_int.columns:
            dt = df_int.dropna(subset=["indice_risco_evasao"]).sort_values("ano")
            fig_t = go.Figure()
            fig_t.add_hrect(y0=0,  y1=20,  fillcolor="#D1FAE5", opacity=0.2, line_width=0, annotation_text="Baixo",    annotation_position="right")
            fig_t.add_hrect(y0=20, y1=35,  fillcolor="#FEF3C7", opacity=0.2, line_width=0, annotation_text="Moderado", annotation_position="right")
            fig_t.add_hrect(y0=35, y1=50,  fillcolor="#FEE2E2", opacity=0.2, line_width=0, annotation_text="Alto",     annotation_position="right")
            fig_t.add_hrect(y0=50, y1=100, fillcolor="#FECACA", opacity=0.2, line_width=0, annotation_text="Crítico",  annotation_position="right")
            fig_t.add_trace(go.Scatter(
                x=dt["ano"], y=dt["indice_risco_evasao"],
                mode="lines+markers+text",
                text=dt["indice_risco_evasao"].round(1),
                textposition="top center",
                line=dict(color=COR_EM, width=3),
                marker=dict(size=10, color=dt["indice_risco_evasao"],
                            colorscale="RdYlGn_r", cmin=0, cmax=60,
                            line=dict(color="white", width=2)),
                name="Score de Risco",
            ))
            fig_t.update_layout(
                yaxis_title="Score (0–100)", xaxis_title="Ano",
                hovermode="x unified", height=300, showlegend=False,
            )
            st.plotly_chart(fig_t, use_container_width=True)

    st.markdown("---")

    # ── Painel de indicadores do último ano ───────────────────────────────────
    st.subheader("📌 Evolução dos Indicadores Críticos")
    st.caption(f"Comparação entre {int(df_int_s.iloc[0]['ano'])} e {int(ultimo['ano'])}.")

    primeiro = df_int_s.iloc[0]
    indicadores = [
        ("taxa_evasao_em",     "Evasão EM",     "%", True),
        ("taxa_abandono_em",   "Abandono EM",   "%", True),
        ("tdi_em",             "TDI EM",        "%", True),
        ("taxa_repetencia_em", "Repetência EM", "%", True),
        ("atu_em",             "Alunos/Turma EM","", False),
    ]
    cols_i = st.columns(len(indicadores))
    for col_ui, (nome, label, unid, inv) in zip(cols_i, indicadores):
        v_ini = float(primeiro.get(nome, np.nan) or np.nan)
        v_fim = float(ultimo.get(nome, np.nan) or np.nan)
        if pd.notna(v_ini) and pd.notna(v_fim):
            with col_ui:
                st.metric(
                    label=label,
                    value=f"{v_fim:.1f}{unid}",
                    delta=f"{v_fim - v_ini:+.1f}{unid} vs. {int(primeiro['ano'])}",
                    delta_color="inverse" if inv else "normal",
                )


# ===========================================================================
# PÁGINA 2 — ONDE ESTÁ O PROBLEMA
# ===========================================================================

def pagina_onde_problema(dados: dict, filtros: dict):
    st.title("📍 Onde Está o Problema — Ranking de Risco")
    st.caption(
        "Identificação dos períodos e registros com maior score de risco. "
        "O ranking permite priorizar onde concentrar as intervenções."
    )
    st.markdown("---")

    a1, a2 = filtros["ano_range"]
    df_educ = dados["fato_educacional"].copy()
    df_educ = df_educ[(df_educ["ano"] >= a1) & (df_educ["ano"] <= a2)]
    df_educ["score_risco"] = calcular_score(df_educ)
    df_educ["nivel_risco"] = classificar_risco(df_educ["score_risco"])

    # ── Distribuição de risco ──────────────────────────────────────────────────
    col1, col2 = st.columns([1, 2])

    with col1:
        secao("Distribuição por Nível de Risco",
              "Quantos registros escolares se enquadram em cada categoria?", "🎯")
        contagem = df_educ["nivel_risco"].value_counts().reindex(
            ["🚨 Crítico", "🔴 Alto", "🟡 Moderado", "🟢 Baixo"], fill_value=0
        )
        fig_pie = go.Figure(go.Pie(
            labels=contagem.index, values=contagem.values,
            marker_colors=[COR_CRITICO, COR_ALTO, COR_MEDIO, COR_BAIXO],
            hole=0.45, textinfo="label+percent+value",
        ))
        fig_pie.update_layout(
            height=320, showlegend=False,
            annotations=[dict(text=f"{len(df_educ)}<br>registros", x=0.5, y=0.5, font_size=14, showarrow=False)],
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        n_alto = df_educ["nivel_risco"].str.contains("Alto|Crítico", na=False).sum()
        pct = n_alto / len(df_educ) * 100 if len(df_educ) > 0 else 0
        insight(f"{n_alto} registros ({pct:.0f}%) em nível Alto ou Crítico — esses devem ser priorizados.")

    with col2:
        secao("Score de Risco por Ano",
              "Média anual do score com intervalo min–max. Quanto mais alto, mais crítico o período.", "📊")
        score_ano = df_educ.groupby("ano")["score_risco"].agg(["mean", "max", "min"]).reset_index()
        score_ano.columns = ["ano", "medio", "maximo", "minimo"]

        fig_sc = go.Figure()
        fig_sc.add_hrect(y0=0,  y1=20,  fillcolor="#D1FAE5", opacity=0.15, line_width=0)
        fig_sc.add_hrect(y0=20, y1=35,  fillcolor="#FEF3C7", opacity=0.15, line_width=0)
        fig_sc.add_hrect(y0=35, y1=50,  fillcolor="#FEE2E2", opacity=0.15, line_width=0)
        fig_sc.add_hrect(y0=50, y1=100, fillcolor="#FECACA", opacity=0.15, line_width=0)
        # Faixa min-max
        fig_sc.add_trace(go.Scatter(
            x=pd.concat([score_ano["ano"], score_ano["ano"][::-1]]),
            y=pd.concat([score_ano["maximo"], score_ano["minimo"][::-1]]),
            fill="toself", fillcolor=hex_rgba(COR_EM, 0.1),
            line=dict(color="rgba(0,0,0,0)"), name="Intervalo",
        ))
        fig_sc.add_trace(go.Scatter(
            x=score_ano["ano"], y=score_ano["medio"],
            mode="lines+markers", name="Score médio",
            line=dict(color=COR_EM, width=3),
            marker=dict(size=9, color=score_ano["medio"],
                        colorscale="RdYlGn_r", cmin=0, cmax=60,
                        line=dict(color="white", width=2)),
            text=score_ano["medio"].round(1), textposition="top center",
        ))
        fig_sc.update_layout(yaxis_title="Score de Risco", xaxis_title="Ano",
                             hovermode="x unified", height=320,
                             legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_sc, use_container_width=True)

    st.markdown("---")

    # ── Ranking detalhado ──────────────────────────────────────────────────────
    secao("Ranking — Registros com Maior Score de Risco",
          "Ordenados do mais crítico ao menos crítico. Use o filtro de risco na sidebar para focar nos casos prioritários.", "🏆")

    niveis_sel = filtros.get("risco_filtro", ["🔴 Alto", "🚨 Crítico"])
    df_rank = df_educ[df_educ["nivel_risco"].isin(niveis_sel)].copy() if niveis_sel else df_educ.copy()
    df_rank = df_rank.sort_values("score_risco", ascending=False).reset_index(drop=True)
    df_rank.index += 1

    rename = {
        "ano": "Ano", "nivel_risco": "Nível de Risco", "score_risco": "Score (0–100)",
        "taxa_abandono_em": "Abandono EM (%)", "taxa_abandono_ef": "Abandono EF (%)",
        "tdi_em": "TDI EM (%)", "tdi_ef": "TDI EF (%)",
        "taxa_reprovacao_em": "Reprovação EM (%)", "taxa_reprovacao_ef": "Reprovação EF (%)",
        "atu_em": "Alunos/Turma EM", "atu_ef": "Alunos/Turma EF",
    }
    cols_show = [c for c in rename if c in df_rank.columns]
    df_display = df_rank[cols_show].rename(columns=rename)

    st.dataframe(
        df_display, use_container_width=True, height=420,
        column_config={
            "Score (0–100)": st.column_config.ProgressColumn("Score (0–100)", min_value=0, max_value=100, format="%.1f"),
        },
    )
    st.caption(f"Exibindo {len(df_display)} registros para os níveis selecionados.")

    st.markdown("---")

    # ── Mapa de calor nível de risco × ano ────────────────────────────────────
    secao("Mapa de Calor — Concentração de Risco por Ano",
          "Cada célula mostra quantos registros estão naquele nível de risco e ano.", "🗺️")

    pivot = df_educ.pivot_table(
        index="nivel_risco", columns="ano", values="score_risco",
        aggfunc="count", fill_value=0,
    ).reindex(["🚨 Crítico", "🔴 Alto", "🟡 Moderado", "🟢 Baixo"])

    if not pivot.empty:
        fig_h = px.imshow(
            pivot, color_continuous_scale=["#D1FAE5", "#FEF3C7", "#FEE2E2", "#DC2626"],
            text_auto=True, labels={"color": "Nº registros"},
        )
        fig_h.update_layout(height=300)
        st.plotly_chart(fig_h, use_container_width=True)
        insight("Anos com maior concentração nas linhas 'Alto' e 'Crítico' exigem atenção prioritária. "
                "Observe o impacto da pandemia (2020–2022) nas colunas mais recentes.")


# ===========================================================================
# PÁGINA 3 — POR QUE OCORRE
# ===========================================================================

def pagina_por_que(dados: dict, filtros: dict):
    st.title("🔍 Por Que Ocorre — Fatores Associados à Evasão")
    st.caption(
        "Análise dos fatores que explicam o risco de evasão. "
        "Entender as causas é essencial para intervir de forma eficaz e direcionada."
    )
    st.markdown("---")

    a1, a2 = filtros["ano_range"]
    df_int   = dados["fato_integrado"].copy()
    df_int   = df_int[(df_int["ano"] >= a1) & (df_int["ano"] <= a2)]
    df_socio = dados["dim_socio_anual"].copy()
    df_socio = df_socio[(df_socio["ano"] >= a1) & (df_socio["ano"] <= a2)]

    # ── Cadeia causal ──────────────────────────────────────────────────────────
    st.subheader("🔗 Cadeia Causal da Evasão")
    st.markdown(
        """<div style="background:#F8FAFC;border:1px solid #CBD5E1;padding:16px 20px;border-radius:8px;line-height:1.8">
        <p style="margin:0;color:#334155;font-size:0.92rem">
        Os dados de Recife evidenciam uma cadeia causal clara e recorrente:<br><br>
        <b style="color:#DC2626">① Reprovação</b> → acúmulo de anos na mesma série →
        <b style="color:#D97706">② Distorção Idade-Série (TDI)</b> → desmotivação e sentimento de inadequação →
        <b style="color:#EF4444">③ Abandono</b> no meio do ano letivo →
        <b style="color:#DC2626">④ Evasão definitiva</b> do sistema escolar.<br><br>
        <i>Escolas que reduzem reprovação tendem a apresentar menor TDI e, consequentemente, menor abandono e evasão.
        A correlação entre baixa reprovação e baixa evasão é uma das mais fortes neste dataset.</i>
        </p></div>""",
        unsafe_allow_html=True,
    )
    st.markdown("")

    # ── Reprovação × Evasão ────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        secao("Reprovação × Evasão no EM",
              "Cada ponto é um ano. A linha de tendência mostra a relação direta entre os dois indicadores.", "📌")
        xc, yc = "taxa_repetencia_em", "taxa_evasao_em"
        if xc in df_socio.columns and yc in df_socio.columns:
            s = df_socio.dropna(subset=[xc, yc])
            if not s.empty:
                fig = go.Figure()
                scatter_tendencia(fig, s[xc].values, s[yc].values, "EM", COR_EM, s["ano"].values)
                mx, my = s[xc].mean(), s[yc].mean()
                fig.add_hline(y=my, line_dash="dot", line_color="gray", opacity=0.5)
                fig.add_vline(x=mx, line_dash="dot", line_color="gray", opacity=0.5)
                fig.update_layout(xaxis_title="Repetência EM (%)", yaxis_title="Evasão EM (%)",
                                  showlegend=False, height=380)
                st.plotly_chart(fig, use_container_width=True)
                r = np.corrcoef(s[xc].values, s[yc].values)[0, 1]
                slope = np.polyfit(s[xc].values, s[yc].values, 1)[0]
                insight(f"Correlação Pearson = <b>{r:.2f}</b>. "
                        f"Cada +1pp de repetência está associado a +{slope:.2f}pp de evasão. "
                        f"<b>Reduzir reprovação = reduzir evasão.</b>")

    with col2:
        secao("TDI × Abandono no EM",
              "A distorção idade-série amplifica o risco: quanto maior o atraso, maior a chance de abandono.", "📐")
        if all(c in df_int.columns for c in ["tdi_em", "taxa_abandono_em"]):
            s = df_int.dropna(subset=["tdi_em", "taxa_abandono_em"])
            if not s.empty:
                fig2 = go.Figure()
                scatter_tendencia(fig2, s["tdi_em"].values, s["taxa_abandono_em"].values,
                                  "EM", COR_WARN, s["ano"].values)
                fig2.update_layout(xaxis_title="TDI EM (%)", yaxis_title="Abandono EM (%)",
                                   showlegend=False, height=380)
                st.plotly_chart(fig2, use_container_width=True)
                r2 = np.corrcoef(s["tdi_em"].values, s["taxa_abandono_em"].values)[0, 1]
                insight(f"Correlação TDI × Abandono = <b>{r2:.2f}</b>. "
                        "Alunos com mais de 2 anos de atraso têm risco de abandono substancialmente maior. "
                        "<b>Programas de nivelamento reduzem a TDI e o abandono simultaneamente.</b>")

    st.markdown("---")

    # ── Diagnóstico: Por que este período está em risco? ──────────────────────
    secao("Por Que Este Período Está em Risco?",
          f"Análise fator a fator do último ano do período selecionado ({int(df_int.sort_values('ano').iloc[-1]['ano']) if not df_int.empty else '–'}).", "🧠")

    if not df_int.empty:
        ultimo = df_int.sort_values("ano").iloc[-1]
        fatores = []

        def avaliar(col, label, lim_atenc, lim_crit, inv=False):
            v = float(ultimo.get(col, np.nan) or np.nan)
            if pd.isna(v): return
            if not inv:
                if v >= lim_crit:   st_nivel, impacto = "🚨 Crítico", "alto"
                elif v >= lim_atenc: st_nivel, impacto = "⚠️ Atenção", "médio"
                else:                st_nivel, impacto = "✅ OK", "baixo"
            else:
                if v <= lim_crit:   st_nivel, impacto = "🚨 Crítico", "alto"
                elif v <= lim_atenc: st_nivel, impacto = "⚠️ Atenção", "médio"
                else:                st_nivel, impacto = "✅ OK", "baixo"
            fatores.append({"Indicador": label, "Valor": f"{v:.1f}%", "Status": st_nivel, "Impacto": impacto})

        avaliar("taxa_evasao_em",     "Evasão EM",           5,  10)
        avaliar("taxa_abandono_em",   "Abandono EM",         5,  10)
        avaliar("tdi_em",             "Distorção Idade-Série EM", 20, 30)
        avaliar("taxa_repetencia_em", "Repetência EM",       8,  15)
        avaliar("taxa_reprovacao_em", "Reprovação EM",       8,  15)
        avaliar("taxa_evasao_ef",     "Evasão EF",           3,   6)
        avaliar("tdi_ef",             "Distorção Idade-Série EF", 15, 25)

        if fatores:
            criticos = [f for f in fatores if f["Impacto"] == "alto"]
            atencao  = [f for f in fatores if f["Impacto"] == "médio"]
            ok       = [f for f in fatores if f["Impacto"] == "baixo"]
            ca, cb, cc = st.columns(3)
            with ca:
                st.markdown("**🚨 Fatores Críticos**")
                for f in criticos: st.markdown(f"- **{f['Indicador']}**: {f['Valor']}")
                if not criticos: st.success("Nenhum fator crítico.")
            with cb:
                st.markdown("**⚠️ Fatores em Atenção**")
                for f in atencao: st.markdown(f"- **{f['Indicador']}**: {f['Valor']}")
                if not atencao: st.success("Nenhum em atenção.")
            with cc:
                st.markdown("**✅ Dentro do Esperado**")
                for f in ok: st.markdown(f"- **{f['Indicador']}**: {f['Valor']}")
                if not ok: st.warning("Todos precisam de atenção.")

    st.markdown("---")

    # ── Matriz de correlação ───────────────────────────────────────────────────
    secao("Matriz de Correlação",
          "Verde = relação inversa (mais aprovação → menos evasão). Vermelho = relação direta (mais reprovação → mais evasão).",
          "🗺️")

    cols_c = [c for c in [
        "taxa_evasao_em", "taxa_abandono_em", "taxa_repetencia_em", "taxa_reprovacao_em",
        "tdi_em", "atu_em", "taxa_aprovacao_em", "taxa_promocao_em",
        "taxa_evasao_ef", "taxa_abandono_ef", "tdi_ef",
    ] if c in df_int.columns]

    df_c = df_int[cols_c].dropna(how="all")
    if len(df_c) >= 3:
        corr = df_c.corr()
        col_c1, col_c2 = st.columns([3, 2])
        with col_c1:
            fig_corr = px.imshow(corr, color_continuous_scale="RdYlGn",
                                 zmin=-1, zmax=1, text_auto=".2f", aspect="auto")
            fig_corr.update_layout(height=480)
            st.plotly_chart(fig_corr, use_container_width=True)
        with col_c2:
            if "taxa_evasao_em" in corr.columns:
                corr_em = corr["taxa_evasao_em"].drop("taxa_evasao_em").dropna().sort_values()
                fig_bar = go.Figure(go.Bar(
                    x=corr_em.values.round(2), y=corr_em.index, orientation="h",
                    marker_color=[COR_OK if v < 0 else COR_EM for v in corr_em],
                    text=corr_em.values.round(2), textposition="outside",
                ))
                fig_bar.add_vline(x=0, line_color="black", line_width=1)
                fig_bar.update_layout(title="Correlações com Evasão EM",
                                      xaxis=dict(range=[-1.2, 1.2]), height=480)
                st.plotly_chart(fig_bar, use_container_width=True)
        insight("Confirma-se: <b>taxa_aprovacao</b> e <b>taxa_promocao</b> têm correlação "
                "<b>negativa forte</b> com evasão — escolas com mais aprovação têm menos evasão. "
                "TDI e repetência têm correlação <b>positiva</b> com evasão.")


# ===========================================================================
# PÁGINA 4 — O QUE FAZER
# ===========================================================================

def pagina_o_que_fazer(dados: dict, filtros: dict):
    st.title("✅ O Que Fazer — Ações Priorizadas")
    st.caption(
        "Recomendações de intervenção ordenadas por impacto e urgência, "
        "baseadas nos fatores de risco identificados nos dados."
    )
    st.markdown("---")

    a1, a2 = filtros["ano_range"]
    df_int = dados["fato_integrado"].copy()
    df_int = df_int[(df_int["ano"] >= a1) & (df_int["ano"] <= a2)]
    ultimo = df_int.sort_values("ano").iloc[-1] if not df_int.empty else None

    # ── Diagnóstico rápido ────────────────────────────────────────────────────
    if ultimo is not None:
        st.subheader("🩺 Diagnóstico Rápido do Período")
        score  = float(ultimo.get("indice_risco_evasao", 0) or 0)
        ev_em  = float(ultimo.get("taxa_evasao_em",    0) or 0)
        ab_em  = float(ultimo.get("taxa_abandono_em",  0) or 0)
        tdi    = float(ultimo.get("tdi_em",             0) or 0)
        rep    = float(ultimo.get("taxa_repetencia_em", 0) or 0)
        atu    = float(ultimo.get("atu_em",             0) or 0)
        nivel  = classificar_risco(pd.Series([score])).iloc[0]

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"**Score de Risco:** {score:.1f} ({nivel})")
            st.markdown(f"**Evasão EM:** {ev_em:.1f}%")
        with c2:
            st.markdown(f"**TDI EM:** {tdi:.1f}%")
            st.markdown(f"**Repetência EM:** {rep:.1f}%")
        with c3:
            st.markdown(f"**Abandono EM:** {ab_em:.1f}%")
            st.markdown(f"**Alunos/turma EM:** {atu:.0f}")
    else:
        ev_em = ab_em = tdi = rep = atu = 0

    st.markdown("---")

    # ── Ações priorizadas ──────────────────────────────────────────────────────
    st.subheader("🎯 Plano de Ações por Prioridade")
    st.caption("Ações ordenadas por urgência e impacto esperado, com base nos indicadores de risco identificados.")

    acoes = [
        {
            "prioridade": "🚨 IMEDIATA",
            "cor": "#DC2626", "bg": "#FEE2E2",
            "acao": "Sistema de monitoramento de frequência (alerta precoce)",
            "descricao": "Alunos com mais de 25% de faltas têm alto risco de abandono. "
                         "Implementar alerta semanal automático e contato imediato com família ao atingir o limiar.",
            "impacto": "Redução de até 30% no abandono escolar.",
            "gatilho": f"Abandono EM: {ab_em:.1f}%",
        },
        {
            "prioridade": "🚨 IMEDIATA",
            "cor": "#DC2626", "bg": "#FEE2E2",
            "acao": "Nivelamento e reforço para alunos com distorção idade-série",
            "descricao": "Alunos com TDI elevado têm 3× mais chance de abandonar. "
                         "Aulas de nivelamento em contraturno reduzem a defasagem e aumentam a permanência.",
            "impacto": "Redução direta no TDI e na taxa de abandono.",
            "gatilho": f"TDI EM: {tdi:.1f}%",
        },
        {
            "prioridade": "⚠️ CURTO PRAZO",
            "cor": "#D97706", "bg": "#FEF3C7",
            "acao": "Revisar política de reprovação — progressão com apoio intensivo",
            "descricao": "A reprovação é o maior preditor isolado de evasão neste dataset. "
                         "Substituir reprovação automática por progressão continuada com reforço pedagógico "
                         "quebra a cadeia: reprovação → TDI → abandono → evasão.",
            "impacto": "Ruptura da principal cadeia causal de evasão.",
            "gatilho": f"Repetência EM: {rep:.1f}%",
        },
        {
            "prioridade": "⚠️ CURTO PRAZO",
            "cor": "#D97706", "bg": "#FEF3C7",
            "acao": "Reduzir superlotação nas turmas do Ensino Médio",
            "descricao": "Turmas com mais de 35 alunos dificultam o acompanhamento individual, "
                         "aumentam o abandono e reduzem o vínculo professor-aluno. Meta: máximo 30 alunos/turma.",
            "impacto": "Melhora na qualidade do ensino e no engajamento.",
            "gatilho": f"ATU EM: {atu:.0f} alunos/turma",
        },
        {
            "prioridade": "📅 MÉDIO PRAZO",
            "cor": "#0284C7", "bg": "#DBEAFE",
            "acao": "Ampliar EJA e Ensino Médio noturno para recuperar evadidos",
            "descricao": "Estudantes que já abandonaram precisam de modalidades flexíveis. "
                         "EJA, cursos técnicos integrados e educação a distância aumentam a reintegração de adultos.",
            "impacto": "Reintegração de evadidos ao sistema educacional.",
            "gatilho": "Evasão acumulada ao longo dos anos.",
        },
        {
            "prioridade": "📅 MÉDIO PRAZO",
            "cor": "#0284C7", "bg": "#DBEAFE",
            "acao": "Programas de apoio socioeconômico (bolsas, transporte, alimentação)",
            "descricao": "Fatores externos (pobreza, trabalho infantil, distância) respondem por parte da evasão. "
                         "Benefícios condicionados à frequência reduzem a evasão em populações vulneráveis.",
            "impacto": "Redução da evasão por fatores socioeconômicos.",
            "gatilho": "Indicadores socioeconômicos correlacionados com evasão.",
        },
        {
            "prioridade": "🔭 LONGO PRAZO",
            "cor": "#059669", "bg": "#D1FAE5",
            "acao": "Dashboard com atualização mensal de indicadores por escola",
            "descricao": "Monitoramento contínuo é mais eficaz que ações pontuais. "
                         "Sistema com dados de frequência, notas e perfil socioeconômico por escola, "
                         "alimentado mensalmente, permite intervenções preventivas.",
            "impacto": "Prevenção ao invés de remediação.",
            "gatilho": "Monitoramento contínuo reduz tempo de reação.",
        },
        {
            "prioridade": "🔭 LONGO PRAZO",
            "cor": "#059669", "bg": "#D1FAE5",
            "acao": "Análise geoespacial — cruzar com dados do IBGE por bairro/RPA",
            "descricao": "Mapear concentração de evasão por microrregião de Recife usando IDH, "
                         "renda e desemprego para direcionar recursos com máxima precisão.",
            "impacto": "Alocação eficiente de recursos públicos.",
            "gatilho": "Análise espacial ainda não realizada.",
        },
    ]

    for acao in acoes:
        st.markdown(
            f"""<div style="background:{acao['bg']};border-left:5px solid {acao['cor']};
            padding:14px 18px;border-radius:6px;margin-bottom:12px;">
            <b style="color:{acao['cor']};font-size:0.78rem;letter-spacing:0.05em">{acao['prioridade']}</b>
            <h4 style="margin:4px 0 6px 0;color:#111827">{acao['acao']}</h4>
            <p style="margin:0 0 6px 0;color:#374151;font-size:0.9rem">{acao['descricao']}</p>
            <span style="color:#6B7280;font-size:0.82rem">
            📊 <b>Impacto:</b> {acao['impacto']} &nbsp;|&nbsp;
            📌 <b>Gatilho nos dados:</b> {acao['gatilho']}
            </span></div>""",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    secao("Resumo das Ações", "", "📋")
    st.dataframe(
        pd.DataFrame([{"Prioridade": a["prioridade"], "Ação": a["acao"], "Impacto Esperado": a["impacto"]} for a in acoes]),
        use_container_width=True, hide_index=True,
    )


# ===========================================================================
# PÁGINA 5 — EVOLUÇÃO TEMPORAL
# ===========================================================================

def pagina_temporal(dados: dict, filtros: dict):
    st.title("📈 Evolução Temporal")
    st.caption(
        "Análise histórica dos indicadores de evasão de Recife (2006–2024). "
        "Identifique tendências de melhora, piora e o impacto da pandemia de COVID-19."
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

    # ── Evasão + Abandono histórico ────────────────────────────────────────────
    secao("Evasão e Abandono — Série Histórica",
          "Abandono (saída no meio do ano) é o precursor da evasão (saída definitiva). Os dois devem cair juntos.",
          "📉")
    fig = go.Figure()
    fig.add_vrect(x0=2020, x1=2022, fillcolor="#FEF3C7", opacity=0.3,
                  line_width=0, annotation_text="Pandemia COVID-19", annotation_position="top left")

    for nivel, show, c_ev, c_ab, nome in [
        ("ef", show_ef, COR_EF,    "#93C5FD", "EF"),
        ("em", show_em, COR_EM,    "#FCA5A5", "EM"),
    ]:
        if not show: continue
        s_s = df_soc.dropna(subset=[f"taxa_evasao_{nivel}"])
        s_e = df_educ.dropna(subset=[f"taxa_abandono_{nivel}"])
        if not s_s.empty:
            fig.add_trace(go.Scatter(x=s_s["ano"], y=s_s[f"taxa_evasao_{nivel}"],
                                     name=f"Evasão {nome}", mode="lines+markers",
                                     line=dict(color=c_ev, width=3), marker=dict(size=8)))
        if not s_e.empty:
            fig.add_trace(go.Scatter(x=s_e["ano"], y=s_e[f"taxa_abandono_{nivel}"],
                                     name=f"Abandono {nome}", mode="lines+markers",
                                     line=dict(color=c_ab, width=2, dash="dot"), marker=dict(size=7)))

    fig.update_layout(yaxis_title="Taxa (%)", xaxis_title="Ano",
                      hovermode="x unified", height=420,
                      legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, use_container_width=True)
    insight("O período 2006–2019 apresentou <b>queda consistente</b> nos indicadores. "
            "A pandemia (2020–2022) interrompeu essa trajetória, especialmente no EM. "
            "Os dados de 2023–2024 mostram recuperação parcial, mas ainda abaixo dos níveis pré-pandemia.")

    st.markdown("---")

    # ── Variação YoY ──────────────────────────────────────────────────────────
    secao("Variação Ano a Ano — Evasão no Ensino Médio",
          "Vermelho = evasão piorou neste ano. Verde = evasão melhorou. Use para identificar anos críticos.", "📊")
    if "var_taxa_evasao_em" in tend.columns:
        s = tend.dropna(subset=["var_taxa_evasao_em"])
        if not s.empty:
            fig_y = go.Figure(go.Bar(
                x=s["ano"], y=s["var_taxa_evasao_em"],
                marker_color=[COR_EM if v > 0 else COR_OK for v in s["var_taxa_evasao_em"]],
                text=s["var_taxa_evasao_em"].apply(lambda x: f"{x:+.1f}%"),
                textposition="outside",
            ))
            fig_y.add_hline(y=0, line_color="black", line_width=1.5)
            fig_y.update_layout(yaxis_title="Variação (%)", xaxis_title="Ano", height=360)
            st.plotly_chart(fig_y, use_container_width=True)

    st.markdown("---")

    # ── EF vs EM lado a lado ───────────────────────────────────────────────────
    secao("EF vs. EM — Comparação Direta",
          "O EM é consistentemente mais crítico. A linha mostra quantas vezes a evasão no EM é maior que no EF.", "⚖️")
    if all(c in df_int.columns for c in ["taxa_evasao_ef", "taxa_evasao_em"]):
        dc = df_int.dropna(subset=["taxa_evasao_ef", "taxa_evasao_em"]).sort_values("ano").copy()
        dc["ratio"] = (dc["taxa_evasao_em"] / dc["taxa_evasao_ef"]).round(2)
        fig_c = make_subplots(specs=[[{"secondary_y": True}]])
        fig_c.add_trace(go.Bar(x=dc["ano"], y=dc["taxa_evasao_ef"], name="Evasão EF",
                               marker_color=COR_EF, opacity=0.8), secondary_y=False)
        fig_c.add_trace(go.Bar(x=dc["ano"], y=dc["taxa_evasao_em"], name="Evasão EM",
                               marker_color=COR_EM, opacity=0.8), secondary_y=False)
        fig_c.add_trace(go.Scatter(x=dc["ano"], y=dc["ratio"], name="Razão EM/EF",
                                   mode="lines+markers",
                                   line=dict(color="#7C3AED", width=2, dash="dot"),
                                   marker=dict(size=7)), secondary_y=True)
        fig_c.update_yaxes(title_text="Evasão (%)", secondary_y=False)
        fig_c.update_yaxes(title_text="Razão EM ÷ EF", secondary_y=True)
        fig_c.update_layout(barmode="group", hovermode="x unified", height=400,
                            legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_c, use_container_width=True)
        insight("A razão EM/EF ficou historicamente entre <b>2× e 5×</b>, confirmando que o Ensino Médio "
                "é o nível mais crítico e deve ser o foco prioritário das políticas de combate à evasão.")

    st.markdown("---")

    # ── Boxplot por período histórico ─────────────────────────────────────────
    secao("Distribuição por Período Histórico",
          "A caixa mostra a variação dos valores no período. A linha central é a mediana.", "📦")
    col1, col2 = st.columns(2)
    for col, base, col_v, nome_g in [
        (col1, dados["fato_socioeconomico"], "taxa_evasao_em",  "Evasão EM"),
        (col2, dados["fato_educacional"],    "taxa_abandono_em","Abandono EM"),
    ]:
        with col:
            s = base.dropna(subset=[col_v])
            s = s[(s["ano"] >= a1) & (s["ano"] <= a2)]
            if s.empty: continue
            ordem = ["2006–2010", "2011–2015", "2016–2019", "2020–2022 (Pandemia)", "2023–2024"]
            fig_b = px.box(s, x="periodo", y=col_v,
                           category_orders={"periodo": [p for p in ordem if p in s["periodo"].unique()]},
                           color="periodo", color_discrete_map=PALETA_PERIODO,
                           title=nome_g, labels={"periodo": "Período", col_v: "%"},
                           points="all")
            fig_b.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_b, use_container_width=True)


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    garantir_dados()
    dados = carregar_dados()

    if not dados:
        st.error("Não foi possível carregar os dados. Execute o ETL primeiro.")
        st.stop()

    filtros = sidebar(dados)

    paginas = {
        "🏠 Painel de Alertas":      pagina_alertas,
        "📍 Onde Está o Problema":   pagina_onde_problema,
        "🔍 Por Que Ocorre":         pagina_por_que,
        "✅ O Que Fazer":            pagina_o_que_fazer,
        "📈 Evolução Temporal":      pagina_temporal,
    }

    st.sidebar.markdown("---")
    pagina_atual = st.sidebar.radio("🗂️ Navegação", list(paginas.keys()))
    paginas[pagina_atual](dados, filtros)

    st.sidebar.markdown("---")
    st.sidebar.caption("Projeto: Evasão Escolar · Recife · INEP/MEC")


if __name__ == "__main__":
    main()
