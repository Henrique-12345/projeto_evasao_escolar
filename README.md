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
├── docs/
│   ├── definicao_problema_e_escopo.md   # Problema de ML: alvo abandono EM × granularidade escola–ano
│   ├── plano_tecnico_dados.md
│   └── politica_dados_ausentes.md
├── etl/
│   └── etl_pipeline.py          # Pipeline ETL: Extract → Transform → Load
├── ml/
│   └── baseline_municipio.py    # Pipeline sklearn + baseline Ridge + análise de impacto de ausentes
├── notebooks/
│   ├── modelagem_evasao_municipio.ipynb
│   └── analyse_missing_values.ipynb   # Qualidade de dados e relatório de ausentes
├── dashboard/
│   └── app.py                   # Dashboard interativo (Streamlit + Plotly)
├── analise_evasao_escolar.ipynb
├── iniciar_dashboard.bat
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
- Remove duplicatas (quantidades registradas nos logs na execução)
- Corrige tipos de dados e faz clip de valores impossíveis (0–100%)
- Cria classificação por período histórico
- Classifica linhas educacionais por nível de risco de abandono
- **Agrega apenas a base socioeconômica por ano** (uma série municipal `taxa_*`)
- **Agrega por ano uma cópia da educacional** para `dim_educ_anual` e para **`dim_integrado_anual`** (séries município–ano usadas em gráficos temporais)
- **`fato_integrado`**: mantém **granularidade escola–ano** na base educacional e faz `merge` das taxas socioeconômicas municipais por **`ano`** (indicadores municipais **replicados** em cada escola)
- Calcula o **Índice de Risco de Evasão** por linha (combinação de evasão municipal EM + TDI escola + repetência municipal EM)
- Gera identificador **`id_linha_educacional`** (o extrato atual **não** contém código INEP/nome de escola)

**Load** — Salva **8** tabelas em CSVs processados e no banco SQLite (`dim_integrado_anual` incluída)

Ao final do **Load**, o ETL gera **`outputs/relatorio_missing_values.csv`** (e figuras em `outputs/figures/` quando possível), com contagens e percentuais de ausentes por coluna, ranking e **formas das tabelas** (bruto vs processado).

---

## Política de valores ausentes

- **No ETL**, a maior parte dos `NaN` **permanece** nas tabelas exportadas — **sem imputação genérica** — para preservar rastreabilidade e evitar distorções antes da análise.
- **Índice de risco de evasão** e **score do dashboard**: componentes ausentes **não** são tratados como zero; os **pesos são renormalizados** sobre os indicadores disponíveis em cada linha.
- **Na modelagem (sklearn)**, `SimpleImputer` (**mediana** nas numéricas, **mais frequente** nas categóricas) fica **dentro do `Pipeline`** e é ajustado **apenas no treino**, evitando vazamento para o conjunto de teste.
- Detalhamento metodológico: **`docs/politica_dados_ausentes.md`**. Notebook exploratório: **`notebooks/analyse_missing_values.ipynb`**.

---

## Integração das Bases de Dados

A base socioeconômica agrega informações **municipais** por ano (várias linhas brutas por ano são consolidadas pela média no ETL). A base educacional mantém **uma linha por escola e ano** no arquivo original — sem código/nome de escola neste extrato; o ETL adiciona apenas **`id_linha_educacional`** como identificador estável da linha.

**`fato_integrado` (fact table principal):** para cada registro educacional, executa-se um `merge` **interno** por **`ano`** com as taxas socioeconômicas municipais daquele ano. Assim, indicadores educacionais (TDI, abandono, reprovação por escola etc.) permanecem **específicos da escola**, enquanto promoção, repetência e evasão municipal são **o mesmo valor para todas as escolas do ano** — representando o **contexto municipal compartilhado**.

**`dim_integrado_anual`:** tabela auxiliar **município–ano** (socio anual × educ anual), usada no dashboard para séries temporais e KPIs que exigem **uma linha por ano** (evita duplicar o mesmo ano centenas de vezes quando se usa `fato_integrado`).

**`fato_educacional`:** base educacional limpa, sem join com a socioeconômica.

O **Índice de Risco de Evasão** na linha escola–ano é calculado com **evasão EM municipal** (mesmo valor para todas as escolas do ano), **TDI EM da escola** e **repetência EM municipal**.

---

## Dashboard

O dashboard é organizado como uma **narrativa em 5 seções**, projetada para comunicar insights de forma clara para qualquer pessoa — técnica ou não. Cada visualização é acompanhada de texto explicativo antes e depois do gráfico.

| Seção | Pergunta respondida | Conteúdo principal |
|---|---|---|
| 1. Contexto Geral | Qual é o problema? | KPIs do ano mais recente, Score de Risco com gauge, diagnóstico automático, variação do período |
| 2. Evolução ao Longo do Tempo | Como o problema mudou ano a ano? | Série temporal de evasão e abandono, variação anual, comparação EF × EM, boxplots por período histórico |
| 3. Impacto da Pandemia | Por que 2020–2021 foram tão ruins? | Explicação dos 3 mecanismos (fechamento, ensino remoto, crise econômica), score antes/durante/após, comparação de indicadores por período |
| 4. Por que os Alunos Evadem? | Quais são as causas? | Cadeia causal (reprovação → TDI → abandono → evasão), correlações com scatter e tendência, diagnóstico por indicador, mapa de correlação |
| 5. Conclusões e Modelo Preditivo | O que fazer? Quais variáveis usar no modelo? | 5 insights consolidados, tabela de preditores para ML (com alvo formal: abandono EM em escola–ano), plano de ação por urgência, projeção simplificada |

**Princípios do dashboard:**
- **Texto antes de cada gráfico** — explica o que será analisado e qual é o insight principal
- **Texto depois de cada gráfico** — interpreta os dados, explica as causas e destaca pontos críticos
- **Insights automáticos** — pior ano, melhor ano, maior alta e maior queda calculados automaticamente dos dados
- **Score de Risco 0–100** — Abandono EM (40%) + TDI (30%) + Reprovação EM (30%); pesos renormalizados se algum componente estiver ausente
- **Período padrão de 4 anos** (últimos 4 disponíveis), ajustável pelo usuário
- **Glossário na barra lateral** com definições de todos os termos técnicos
- **Seção de Machine Learning** — tabela de variáveis preditoras e placeholder para modelo futuro

**Filtros disponíveis:**
- Período de análise (slider — padrão: últimos 4 anos)
- Nível de ensino (EF / EM)

**Glossário de termos** (também disponível na barra lateral do dashboard):

| Termo | Definição |
|---|---|
| Evasão escolar | Saída definitiva do aluno do sistema de ensino |
| Abandono escolar | Saída do aluno durante o ano letivo em curso (precursor da evasão) |
| TDI | Taxa de Distorção Idade-Série: % de alunos com mais de 2 anos de atraso em relação à série esperada |
| p.p. | Ponto percentual: diferença direta entre dois percentuais (ex.: de 10% para 12% = +2 p.p.) |
| ATU | Média de Alunos por Turma |
| EF | Ensino Fundamental (1º ao 9º ano) |
| EM | Ensino Médio (1º ao 3º ano) |
| Score de Risco | Indicador 0–100: mesmos pesos do painel; ausentes não viram zero — pesos renormalizados |

---

## Principais Insights

1. **Ensino Médio é 2–3× mais crítico** que o EF — causas estruturais: pressão econômica, currículo percebido como distante e maior impacto de crises externas
2. **Cadeia causal confirmada pelos dados:** Reprovação → TDI (defasagem escolar) → Abandono → Evasão definitiva. Qualquer intervenção que quebre essa cadeia reduz a evasão
3. **Queda consistente de 2008 a 2019** — políticas educacionais e programas sociais produziram avanços reais; **pandemia de 2020–2021 reverteu parte desse progresso** em meses
4. **TDI e taxa de abandono são preditores centrais** na cadeia rumo à evasão — também alinhados ao modelo escola–ano (alvo: `taxa_abandono_em`).
5. **Recuperação pós-pandemia é lenta:** em 2022, os indicadores melhoraram, mas não retornaram ao nível pré-pandemia; os efeitos educacionais de uma crise dessa magnitude são de longo prazo

### Variáveis prioritárias para modelagem preditiva

| Variável | Importância estimada |
|---|---|
| Taxa de Abandono no EM | Muito Alta — precursor direto da evasão |
| TDI — Distorção Idade-Série | Muito Alta — acumula histórico de defasagem |
| Taxa de Reprovação no EM | Alta — origem da cadeia causal |
| Taxa de Aprovação no EM | Alta — relação inversa forte com evasão |
| Período histórico (pandemia) | Alta — variável de controle essencial |

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

## Machine Learning (baseline)

**Problema central do projeto:** evasão escolar e fatores de risco (análises descritivas e dashboard continuam ancorados na evasão municipal e na cadeia reprovação → TDI → abandono → evasão).

**Modelagem preditiva (regressão supervisionada):** a taxa de abandono escolar do Ensino Médio (`taxa_abandono_em`) foi utilizada como **variável-alvo** devido à sua **disponibilidade em nível escola–ano**, à **variação entre escolas** e à **forte relação com o risco de evasão** na literatura educacional — como indicador operacional associado à evasão, não como substituto conceitual dela. **Não** se usa `taxa_evasao_em` como alvo em `fato_integrado`, porque no join municipal essa variável é **a mesma para todas as escolas do mesmo ano** e não discrimina observações escola–ano.

O notebook e o módulo usam a tabela **`fato_integrado` em nível escola–ano**. **`taxa_evasao_em`** permanece como **covariável de contexto municipal**. Não se usa `indice_risco_evasao` como preditor (risco de vazamento conceitual).

Definição formal: `docs/definicao_problema_e_escopo.md`.

| Artefato | Descrição |
|---|---|
| `ml/baseline_municipio.py` | Carrega `fato_integrado`, `ColumnTransformer` + `Pipeline`, **Ridge** + **DummyRegressor** + comparação (**ElasticNet**, **HistGradientBoosting**), `plot_model_comparison_figures`, validação temporal por ano |
| `notebooks/modelagem_evasao_municipio.ipynb` | EDA, baseline (§5), comparação multi-modelo e figuras (§7), gráfico observado × previsto |

A função **`run_missing_impact_analysis`** (mesmo módulo) compara métricas do Ridge **com todas as features** versus **sem colunas muito incompletas** (taxa de ausência no treino acima de um limiar, por padrão 50%). Isso ajuda a avaliar sensibilidade do modelo à imputação e à presença de covariáveis esparsas — sem substituir o desenho principal baseado no Pipeline completo.

**Validação:** treino nos anos `≤ 2017`, teste nos anos `≥ 2018` (split temporal; amostra pequena — métricas devem ser interpretadas com cautela). Regressão supervisionada: apenas linhas **com alvo observável** entram no fit; ausências nas covariáveis são imputadas dentro do Pipeline no treino.

**Métricas reportadas:** MAE (principal), RMSE e R² no conjunto de teste.

Execute o Jupyter a partir da **raiz do repositório**:

```bash
jupyter notebook notebooks/modelagem_evasao_municipio.ipynb
```

Ou use o VS Code / Cursor para abrir o `.ipynb` com o kernel Python onde `requirements.txt` foi instalado.

---

## Tecnologias

| Biblioteca | Uso |
|---|---|
| `pandas` / `numpy` | Manipulação e transformação de dados |
| `scikit-learn` | Pipeline de pré-processamento e modelo baseline (regressão) |
| `matplotlib` / `seaborn` | Visualizações estáticas (notebook) |
| `plotly` | Gráficos interativos (dashboard) |
| `streamlit` | Interface do dashboard web |
| `sqlite3` | Banco de dados local |
| `jupyter` | Análise exploratória |
