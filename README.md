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

O dashboard interativo é organizado em **5 páginas orientadas à tomada de decisão**, sem jargão excessivo, projetadas para usuários técnicos e não técnicos.

| Página | Propósito | Conteúdo Principal |
|---|---|---|
| Painel de Indicadores | Visão geral executiva | KPIs com variação em p.p., alertas automáticos, gauge do score de risco, evolução histórica |
| Onde está o risco? | Identificação dos casos críticos | Distribuição por nível de risco, ranking com barra de progresso, mapa de calor ano × risco |
| Por que ocorre? | Análise causal | Cadeia causal explicada, correlações reprovação × evasão e TDI × abandono, diagnóstico por indicador, matriz de correlação interativa |
| O que fazer? | Recomendações priorizadas | 8 ações ordenadas por urgência (Imediata / Curto / Médio / Longo prazo) com justificativa nos dados |
| Como a evasão evoluiu? | Análise histórica | Série temporal completa, variação ano a ano, comparação EF × EM, boxplots por período histórico |

**Funcionalidades:**
- **Score de Risco 0–100** por registro e por ano — calculado como: Abandono EM (40%) + TDI (30%) + Reprovação EM (30%)
- **Níveis de risco**: Baixo (0–20) | Moderado (20–35) | Alto (35–50) | Critico (acima de 50)
- **Alertas automáticos** com contexto e interpretação, gerados a partir de limiares nos indicadores
- **Textos interpretativos** em cada gráfico, respondendo o que o dado mostra e por que é relevante
- **Glossário completo** na barra lateral com definições de TDI, p.p., abandono, evasão, ATU, HAD etc.
- **Notas de qualidade dos dados** indicando limitações e cuidados na interpretação
- **Anotações de contexto histórico** (pandemia de COVID-19 destacada nos gráficos)
- **Janela temporal padrão de 4 anos** (últimos 4 anos disponíveis), ajustável via filtro

**Filtros disponíveis na barra lateral:**
- Período de análise (slider de anos — padrão: últimos 4 anos)
- Nível de ensino (EF / EM)
- Nível de risco para filtragem do ranking

**Glossário de termos técnicos** (disponível na barra lateral do dashboard):

| Termo | Definição |
|---|---|
| TDI | Taxa de Distorção Idade-Série: % de alunos com mais de 2 anos de atraso escolar |
| p.p. | Ponto percentual: diferença absoluta entre duas taxas (ex.: de 10% para 12% = +2 p.p.) |
| Abandono | Saída do aluno durante o ano letivo em curso |
| Evasão | Saída definitiva do sistema de ensino |
| ATU | Média de Alunos por Turma |
| HAD | Horas-Aula Diárias |
| EF | Ensino Fundamental (1º ao 9º ano) |
| EM | Ensino Médio (1º ao 3º ano) |

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
