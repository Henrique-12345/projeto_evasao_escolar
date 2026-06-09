"""
Simulação de cenários e impacto de intervenções educacionais.

Reutiliza o bundle do modelo final e ``predict_taxa_abandono_em`` sem retreinamento.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ml.educational_ml import predict_taxa_abandono_em

# Indicadores editáveis pelo gestor (subset acionável das features do modelo).
SIMULATABLE_FEATURES: dict[str, dict[str, Any]] = {
    "tdi_em": {
        "label": "TDI — defasagem idade-série (EM, %)",
        "min": 0.0,
        "max": 100.0,
        "step": 0.5,
        "default": 25.0,
        "unit": "%",
    },
    "taxa_reprovacao_em": {
        "label": "Taxa de reprovação (EM, %)",
        "min": 0.0,
        "max": 100.0,
        "step": 0.5,
        "default": 10.0,
        "unit": "%",
    },
    "taxa_aprovacao_em": {
        "label": "Taxa de aprovação (EM, %)",
        "min": 0.0,
        "max": 100.0,
        "step": 0.5,
        "default": 85.0,
        "unit": "%",
    },
    "taxa_repetencia_em": {
        "label": "Taxa de repetência (EM, %)",
        "min": 0.0,
        "max": 100.0,
        "step": 0.5,
        "default": 8.0,
        "unit": "%",
    },
    "taxa_abandono_ef": {
        "label": "Taxa de abandono (EF, %)",
        "min": 0.0,
        "max": 100.0,
        "step": 0.5,
        "default": 3.0,
        "unit": "%",
    },
    "atu_em": {
        "label": "Alunos por turma — ATU (EM)",
        "min": 10.0,
        "max": 45.0,
        "step": 0.5,
        "default": 35.0,
        "unit": "alunos",
    },
    "had_em": {
        "label": "Horas diárias de aula — HAD (EM)",
        "min": 3.0,
        "max": 10.0,
        "step": 0.1,
        "default": 5.5,
        "unit": "h",
    },
    "tdi_ef": {
        "label": "TDI — defasagem idade-série (EF, %)",
        "min": 0.0,
        "max": 100.0,
        "step": 0.5,
        "default": 20.0,
        "unit": "%",
    },
    "taxa_reprovacao_ef": {
        "label": "Taxa de reprovação (EF, %)",
        "min": 0.0,
        "max": 100.0,
        "step": 0.5,
        "default": 8.0,
        "unit": "%",
    },
}

STORYTELLING_SIMULACAO: dict[str, str] = {
    "titulo": "Simulação de Cenários e Impacto de Intervenções",
    "secao_dashboard": "5. Conclusões e Modelo Preditivo — O que aconteceria se?",
    "proposito": (
        "Permite que gestores alterem indicadores educacionais de uma escola–ano e vejam "
        "como o modelo final reestimaria a taxa de abandono no EM, apoiando o planejamento "
        "de intervenções antes da implementação na prática."
    ),
    "camada_valor": (
        "Quarta camada da solução: além de diagnosticar, priorizar e explicar perfis, "
        "o sistema simula cenários futuros com o mesmo pipeline de inferência já validado."
    ),
}


def baseline_value(row: pd.Series, feature: str, meta: dict[str, Any]) -> float:
    """Valor inicial do controle: observado na linha ou default documentado."""
    if feature in row.index and pd.notna(row[feature]):
        return float(row[feature])
    return float(meta["default"])


def predict_abandono_row(row_df: pd.DataFrame) -> float:
    """Previsão pontual usando o bundle serializado (mesmo fluxo do produto)."""
    out = predict_taxa_abandono_em(row_df)
    return float(out["pred_taxa_abandono_em"].iloc[0])


def classify_pred_abandono(pp: float) -> str:
    """Faixas simples para leitura gestor (abandono previsto em %)."""
    if pp < 5:
        return "Baixo"
    if pp < 10:
        return "Moderado"
    if pp < 15:
        return "Alto"
    return "Crítico"


def build_intervention_narrative(
    changes: list[tuple[str, float, float, str, str]],
    pred_original: float,
    pred_simulated: float,
    *,
    high_risk_threshold: float = 12.0,
) -> str:
    """
    Texto interpretativo da simulação para exibição no dashboard.

    ``changes``: lista (feature, valor_original, valor_novo, rótulo, unidade).
    """

    def _fmt(v: float, unit: str) -> str:
        if unit == "%":
            return f"{v:.1f}%"
        if unit == "h":
            return f"{v:.1f} h"
        return f"{v:.1f}"

    delta = pred_simulated - pred_original
    if not changes:
        return (
            "Nenhum indicador foi alterado em relação ao cenário atual da escola selecionada. "
            f"A previsão de abandono permanece em **{pred_original:.1f}%**."
        )

    frases: list[str] = []
    for _feat, old, new, label, unit in changes:
        if new < old - 1e-6:
            frases.append(
                f"a redução de **{label}** de {_fmt(old, unit)} para {_fmt(new, unit)}"
            )
        elif new > old + 1e-6:
            frases.append(
                f"o aumento de **{label}** de {_fmt(old, unit)} para {_fmt(new, unit)}"
            )

    if not frases:
        return (
            "Os indicadores foram ajustados, mas as mudanças foram muito pequenas para alterar "
            f"materialmente a previsão (**{pred_original:.1f}%** → **{pred_simulated:.1f}%**)."
        )

    if len(frases) == 1:
        corpo = frases[0].capitalize()
    else:
        corpo = f"{frases[0].capitalize()}, combinada com {', '.join(frases[1:])}"

    if delta <= -0.05:
        impacto = (
            f"{corpo} resultou em uma **diminuição estimada de {abs(delta):.1f} p.p.** "
            f"na taxa de abandono prevista (de **{pred_original:.1f}%** para **{pred_simulated:.1f}%**)."
        )
    elif delta >= 0.05:
        impacto = (
            f"{corpo} resultou em um **aumento estimado de {delta:.1f} p.p.** "
            f"na taxa de abandono prevista (de **{pred_original:.1f}%** para **{pred_simulated:.1f}%**)."
        )
    else:
        impacto = (
            f"{corpo} teve **impacto marginal** na previsão "
            f"({pred_original:.1f}% → {pred_simulated:.1f}%)."
        )

    if pred_simulated >= high_risk_threshold and delta < -0.05:
        impacto += (
            " Mesmo após a melhoria simulada, o risco previsto permanece **elevado** — "
            "outros indicadores (como reprovação ou TDI) ainda pesam no modelo."
        )
    elif pred_simulated >= high_risk_threshold and delta >= 0:
        impacto += (
            " O risco permanece **elevado** devido aos altos níveis de reprovação, defasagem "
            "ou outros indicadores observados pelo modelo."
        )
    elif delta <= -0.05:
        impacto += " A simulação sugere **ganho de risco** em relação ao cenário atual."

    impacto += (
        " *Resultado baseado nos padrões aprendidos pelo modelo — simulação de apoio "
        "à decisão, não garantia de efeito causal na rede.*"
    )
    return impacto
