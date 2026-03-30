# Análise de Evasão Escolar — Recife

Projeto de análise de dados sobre **evasão e abandono escolar no Ensino Fundamental e Ensino Médio** na cidade do Recife (PE), utilizando dados do INEP/MEC.

---

## Objetivo

Identificar padrões, tendências e fatores de risco associados à evasão e ao abandono escolar em Recife, integrando duas bases de dados:

- **Base Socioeconômica** — taxas de promoção, repetência e evasão por ano
- **Base Educacional** — indicadores de infraestrutura e desempenho escolar (ATU, HAD, TDI, aprovação, reprovação, abandono)

---

## Estrutura do Projeto

```
projeto_evasao_escolar/
├── data/
│   ├── raw/                     # CSVs originais (não versionados)
│   └── processed/               # Dados gerados pelo ETL (não versionados)
├── etl/
│   └── etl_pipeline.py          # Pipeline ETL: Extract → Transform → Load
├── dashboard/
│   └── app.py                   # Dashboard interativo (Streamlit + Plotly)
├── analise_evasao_escolar.ipynb  # Análise exploratória completa
├── iniciar_dashboard.bat        # Atalho para iniciar o dashboard (Windows)
├── requirements.txt
└── README.md
```

---

## Bases de Dados

| Arquivo | Descrição | Período | Linhas |
|---|---|---|---|
| `dados_socioeconomicos_recife.csv` | Taxas de promoção, repetência e evasão | 2008–2022 | 65 |
| `dados_educacionais_recife.csv` | ATU, HAD, TDI, aprovação, reprovação, abandono | 2006–2024 | 247 |

> **Fonte:** INEP / MEC — Município de Recife (código IBGE: 2611606)

### Dicionário de Variáveis

| Variável | Descrição |
|---|---|
| `taxa_evasao_ef / em` | Taxa de evasão escolar — EF e EM |
| `taxa_abandono_ef / em` | Taxa de abandono no ano letivo — EF e EM |
| `taxa_promocao_ef / em` | Taxa de promoção de série — EF e EM |
| `taxa_repetencia_ef / em` | Taxa de repetência — EF e EM |
| `taxa_aprovacao_ef / em` | Taxa de aprovação — EF e EM |
| `taxa_reprovacao_ef / em` | Taxa de reprovação — EF e EM |
| `tdi_ef / em` | Taxa de Distorção Idade-Série — EF e EM |
| `atu_ef / em` | Média de Alunos por Turma — EF e EM |
| `had_ef / em` | Horas-Aula Diárias — EF e EM |

---

## Como usar

### 1. Clonar o repositório

```bash
git clone https://github.com/<seu-usuario>/projeto_evasao_escolar.git
cd projeto_evasao_escolar
```

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

### 3. Adicionar os dados brutos

Copie os dois arquivos CSV para a pasta `data/raw/`:

```
data/raw/dados_socioeconomicos_recife.csv
data/raw/dados_educacionais_recife.csv
```

### 4. Executar o ETL

```bash
python etl/etl_pipeline.py
```

O ETL irá gerar em `data/processed/`:
- 7 tabelas CSV limpas e transformadas
- Banco SQLite `evasao_escolar.db`

### 5. Iniciar o Dashboard

```bash
streamlit run dashboard/app.py
```

Ou no Windows, clique duas vezes em `iniciar_dashboard.bat`.

O dashboard abre em **http://localhost:8501**.

---

## ETL Pipeline

O pipeline em `etl/etl_pipeline.py` executa três etapas:

**Extract** — Lê os CSVs brutos da pasta `data/raw/`

**Transform**
- Remove duplicatas (14 na base socio, 120 na base educacional)
- Corrige tipos de dados e faz clip de valores impossíveis (0–100%)
- Cria classificação por período histórico
- Classifica escolas por nível de risco de abandono
- Agrega indicadores por ano (média entre escolas)
- Integra as duas bases via JOIN pelo ano
- Calcula o **Índice de Risco de Evasão** composto:
  > Evasão EM (40%) + TDI EM (30%) + Repetência EM (30%)

**Load** — Salva 7 tabelas em CSVs processados e no banco SQLite

---

## Dashboard

O dashboard interativo é organizado em **5 páginas com estrutura de storytelling**, focadas em tomada de decisão:

| Página | Propósito | Conteúdo Principal |
|---|---|---|
| 🏠 Painel de Alertas | **Visão geral executiva** | KPIs, alertas automáticos, gauge de score de risco, tendência histórica do score |
| 📍 Onde Está o Problema | **Identificação dos casos críticos** | Ranking por score, mapa de calor nível×ano, distribuição em pizza por nível de risco |
| 🔍 Por Que Ocorre | **Análise causal** | Cadeia causal, scatter repetência×evasão, TDI×abandono, diagnóstico por fatores, matriz de correlação |
| ✅ O Que Fazer | **Recomendações priorizadas** | 8 ações ordenadas por urgência (Imediata / Curto / Médio / Longo prazo) com gatilho nos dados |
| 📈 Evolução Temporal | **Análise histórica** | Série temporal completa, variação ano a ano, EF vs EM, boxplot por período histórico |

**Funcionalidades do dashboard:**
- **Score de Risco 0–100** por registro e por ano (cor: 🟢 Baixo · 🟡 Moderado · 🔴 Alto · 🚨 Crítico)
- **Alertas automáticos** gerados a partir de thresholds nos indicadores
- **Gauge visual** do score de risco com faixas coloridas
- **Ranking filtrável** por nível de risco com barra de progresso
- **Seção "Por que está em risco?"** com diagnóstico fator a fator
- **Textos explicativos** em cada gráfico e blocos de insight analítico

**Filtros disponíveis na sidebar:**
- Período de análise (slider de anos)
- Nível de ensino (EF / EM)
- Nível de risco para filtragem do ranking

---

## Principais Insights

1. **Ensino Médio é 2–3× mais crítico** que o EF em todas as métricas de evasão
2. **Repetência → TDI → Abandono → Evasão** formam uma cadeia causal identificável nos dados
3. **Queda consistente** de 2008 a 2019; **retrocesso pós-pandemia** visível em 2021–2024
4. Escolas com **turmas superlotadas** (ATU > 35) concentram maior taxa de abandono
5. A **distorção idade-série** é o indicador mais fortemente correlacionado com evasão no EM

---

## Análise Exploratória

O notebook `analise_evasao_escolar.ipynb` contém 16 seções com visualizações estáticas (matplotlib/seaborn):

- Evolução temporal das taxas de evasão e abandono
- Comparação EF vs. EM
- Matriz de correlação completa
- Análise de dispersão (TDI × evasão, repetência × evasão)
- Distribuição dos indicadores por escola
- Boxplots por período histórico
- Insights estratégicos e recomendações de política pública

---

## Tecnologias

| Biblioteca | Uso |
|---|---|
| `pandas` / `numpy` | Manipulação e transformação de dados |
| `matplotlib` / `seaborn` | Visualizações estáticas (notebook) |
| `plotly` | Gráficos interativos (dashboard) |
| `streamlit` | Interface do dashboard web |
| `sqlite3` | Banco de dados local |
| `jupyter` | Análise exploratória |
