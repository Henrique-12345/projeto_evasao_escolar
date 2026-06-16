# Análise de Evasão Escolar — Recife

**Nome da solução:** `projeto_evasao_escolar`  
**Instituição:** [CESAR School](https://www.cesar.school/)  
**Disciplinas:** Aprendizado de Máquina · Projeto 6  
**Site do projeto (Google Sites):** [G15 — Evasão Escolar](https://sites.google.com/cesar.school/g15-evasoescolar/in%C3%ADcio?authuser=0)

### Equipe

| Nome | Email |
|------|--------|
| Henrique Magalhães | hlm2@cesar.school |
| Lucca Gomes | lvg2@cesar.school |
| João Victor Nunes | jvln@cesar.school |
| Pedro Antônio de Freitas | pafm@cesar.school |
| Felipe Bandeira | fbq@cesar.school |
| João Marcelo Pordeus | jmpq@cesar.school |
| Antonio Albuquerque Neto | aaon@cesar.school |
| Lucas Ferraz Santana | lfsf@cesar.school |

---

Projeto de análise de dados sobre **evasão e abandono escolar no Ensino Fundamental e Ensino Médio** na cidade do Recife (PE), utilizando dados do INEP/MEC. A solução integra **EDA**, **modelagem com validação temporal**, **MLflow (MLOps)**, **dashboard Streamlit** e **execução reprodutível via Docker**.

---

## Como rodar o projeto

> **Guia único de execução** (local ou Docker). Detalhes do ETL estão em [ETL Pipeline](#etl-pipeline); definição das bases em [Bases de Dados](#bases-de-dados).

### Pré-requisitos

| Item | Execução local | Execução Docker |
|------|----------------|-----------------|
| **Python 3.10+** | Obrigatório (recomendado 3.11) | Não — só Docker |
| **Docker + Compose** | Opcional | Obrigatório |
| **2 CSVs** em `data/raw/` | Obrigatório | Obrigatório |
| **Tempo (1ª vez)** | ~5–15 min (ETL + treino ML) | ~10–25 min (build da imagem + ETL + treino) |

> **Versões fixadas:** `requirements.txt` usa **`scikit-learn==1.7.2`**. O bundle `outputs/ml/final_model_bundle.pkl` só carrega com essa versão. Se alterar o scikit-learn, rode de novo `run_educational_ml_suite()` ou `docker compose run --rm train`.

### 1. Clonar o repositório

```bash
git clone https://github.com/Henrique-12345/projeto_evasao_escolar.git
cd projeto_evasao_escolar
```

### 2. Obter os dados brutos

Os CSVs **não vêm no clone do Git** (e `data/processed/` é gerado pelo ETL). Crie a pasta e copie os dois arquivos:

```
data/raw/dados_socioeconomicos_recife.csv
data/raw/dados_educacionais_recife.csv
```

**Onde obter:** (a) extrair/construir a partir dos microdados abertos do **INEP/MEC** para Recife (IBGE 2611606), conforme [Bases de Dados](#bases-de-dados); ou (b) usar os arquivos disponibilizados pela equipe G15 no [site do projeto](https://sites.google.com/cesar.school/g15-evasoescolar/in%C3%ADcio?authuser=0) ou pelo orientador da disciplina.

Sem esses arquivos, o ETL e o dashboard (páginas 1–5) não funcionam.

### 3. Escolher forma de execução

#### Opção A — Docker (recomendado)

**Pré-requisitos:** [Docker](https://docs.docker.com/get-docker/) e [Docker Compose](https://docs.docker.com/compose/).

**Permissão (Linux):** após instalar o Docker, adicione seu usuário ao grupo `docker` e faça login de novo (ou reinicie o terminal/Cursor):

```bash
sudo usermod -aG docker $USER
# depois: logout/login ou `newgrp docker` no terminal atual
groups   # deve listar "docker"
```

Na raiz do repositório (com os CSVs já em `data/raw/`):

```bash
# Build (primeira vez ou após mudar requirements.txt / código)
docker compose build

# Dashboard + MLflow juntos (recomendado)
docker compose up dashboard mlflow
```

| Serviço | URL | Porta no host |
|---------|-----|----------------|
| **Dashboard** (Streamlit) | http://localhost:8501 | 8501 |
| **MLflow UI** | http://localhost:5000 | 5000 |

**Portas:** não rode MLflow local (`scripts/mlflow_ui.sh`) e MLflow no Docker **ao mesmo tempo** — ambos usam a porta **5000**. Pare processos locais antes (`Ctrl+C` ou `kill` nos PIDs em `lsof -i :5000`).

Volumes montados: `./data`, `./mlruns`, `./outputs` — artefatos gerados localmente (bundle, figuras, `mlflow.db`) aparecem nos containers.

**Comandos úteis:**

```bash
# Só dashboard (ETL + treino se faltar bundle + Streamlit)
docker compose up dashboard

# Só MLflow
docker compose up mlflow

# Retreinar modelo + registrar no MLflow (sem subir UI)
docker compose run --rm train

# Segundo plano
docker compose up -d dashboard mlflow

# Parar
docker compose down
```

O entrypoint do dashboard executa ETL (se houver CSVs em `data/raw/`) e `run_educational_ml_suite()` se o bundle estiver ausente ou incompatível com o scikit-learn da imagem.

#### Opção B — Ambiente local (Python + `.venv`)

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # inclui scikit-learn==1.7.2

python etl/etl_pipeline.py
python -c "from ml.educational_ml import run_educational_ml_suite; run_educational_ml_suite()"
```

**MLflow (SQLite):** backend em `mlruns/mlflow.db` (não use a pasta `mlruns/` como URI no MLflow 3+):

```bash
bash scripts/mlflow_ui.sh          # http://localhost:5000
# equivalente:
mlflow ui --backend-store-uri "sqlite:///$(pwd)/mlruns/mlflow.db"
```

**Dashboard:**

```bash
streamlit run dashboard/app.py     # http://localhost:8501
```

No **Windows**, após ETL + suite ML, você também pode usar `iniciar_dashboard.bat` (atalho para o Streamlit).

**Notebooks:** use o kernel do `.venv` (contém `mlflow`, `sklearn`, etc.). Campanha de hiperparâmetros: `mlruns/experimentos_mlflow_parametros.ipynb`.

O comando `run_educational_ml_suite()` gera `outputs/ml/` (CSV, JSON, bundle `.pkl`, figuras) e registra runs em **`mlruns/mlflow.db`**. A **página 5** do dashboard (ML + simulação *O que aconteceria se?*) depende desses artefatos.

Desabilitar MLflow (opcional): `MLFLOW_DISABLED=1 python -c "..."`.

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
│   ├── raw/                     # CSVs brutos (obter antes de rodar — ver README)
│   └── processed/               # Dados gerados pelo ETL (não versionados)
├── docker/
│   └── entrypoint.sh            # Entrypoint Docker (dashboard / train / mlflow-ui)
├── docs/
│   ├── definicao_problema_e_escopo.md   # Problema de ML: alvo abandono EM × granularidade escola–ano
│   ├── plano_tecnico_dados.md
│   ├── politica_dados_ausentes.md
│   ├── historias_epicas_p6.md
│   └── relatorio_p6.md
├── etl/                         # ≡ src/ (dados) no enunciado AM
│   └── etl_pipeline.py          # Pipeline ETL: Extract → Transform → Load
├── ml/                          # ≡ src/ (treinamento) no enunciado AM
│   ├── __init__.py
│   ├── baseline_municipio.py    # Pré-processamento + regressores (HGB, árvore, KNN) + ausentes (Ridge)
│   ├── educational_ml.py        # Suite: comparação, tuning, KMeans, MLflow, export dashboard
│   ├── mlflow_tracking.py       # Rastreamento MLflow (parâmetros, métricas, modelos)
│   └── scenario_simulation.py   # Simulação what-if e narrativa de intervenções
├── mlruns/                      # Experimentos MLflow (mlflow.db + notebook de campanha)
│   └── experimentos_mlflow_parametros.ipynb
├── notebooks/
│   ├── modelagem_evasao_municipio.ipynb
│   └── analyse_missing_values.ipynb   # Qualidade de dados e relatório de ausentes
├── dashboard/                   # ≡ app/ no enunciado AM
│   └── app.py                   # Dashboard interativo (Streamlit + Plotly)
├── analise_evasao_escolar.ipynb
├── Dockerfile
├── docker-compose.yml
├── iniciar_dashboard.bat
├── requirements.txt
└── README.md
```

---

## Bases de Dados

Os arquivos abaixo devem estar em **`data/raw/`** antes de rodar o projeto (ver [Como rodar o projeto](#como-rodar-o-projeto)). Não são commitados no Git; `data/processed/` é gerado pelo ETL.

| Arquivo | Descrição | Período | Linhas |
|---|---|---|---|
| `dados_socioeconomicos_recife.csv` | Taxas de promoção, repetência e evasão | 2008–2022 | 65 |
| `dados_educacionais_recife.csv` | ATU, HAD, TDI, aprovação, reprovação, abandono | 2006–2024 | 247 |

> **Fonte:** INEP / MEC — Município de Recife (código IBGE: 2611606). Se não tiver os CSVs, consulte o [site G15](https://sites.google.com/cesar.school/g15-evasoescolar/in%C3%ADcio?authuser=0) ou a documentação da disciplina.

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
| 5. Conclusões e Modelo Preditivo | O que fazer? Quais variáveis usar no modelo? | 5 insights consolidados, plano de ação por urgência, **Apoio inteligente** (ML), **Simulação «O que aconteceria se?»**, projeção linear indicativa |

**Princípios do dashboard:**
- **Texto antes de cada gráfico** — explica o que será analisado e qual é o insight principal
- **Texto depois de cada gráfico** — interpreta os dados, explica as causas e destaca pontos críticos
- **Insights automáticos** — pior ano, melhor ano, maior alta e maior queda calculados automaticamente dos dados
- **Score de Risco 0–100** — Abandono EM (40%) + TDI (30%) + Reprovação EM (30%); pesos renormalizados se algum componente estiver ausente
- **Período padrão de 4 anos** (últimos 4 disponíveis), ajustável pelo usuário
- **Glossário na barra lateral** com definições de todos os termos técnicos
- **Seção de Machine Learning** — comparação de regressores, modelo final, validação e ranking de risco
- **Simulação de cenários** — sliders para TDI, reprovação, ATU etc.; recalcula abandono previsto com o modelo final (`ml/scenario_simulation.py`)

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

## Machine Learning

**Problema central do projeto:** evasão escolar e fatores de risco (análises descritivas e dashboard continuam ancorados na evasão municipal e na cadeia reprovação → TDI → abandono → evasão).

**Modelagem preditiva (regressão supervisionada):** a taxa de abandono escolar do Ensino Médio (`taxa_abandono_em`) foi utilizada como **variável-alvo** devido à sua **disponibilidade em nível escola–ano**, à **variação entre escolas** e à **forte relação com o risco de evasão** na literatura educacional — como indicador operacional associado à evasão, não como substituto conceitual dela. **Não** se usa `taxa_evasao_em` como alvo em `fato_integrado`, porque no join municipal essa variável é **a mesma para todas as escolas do mesmo ano** e não discrimina observações escola–ano.

O notebook e o módulo usam a tabela **`fato_integrado` em nível escola–ano**. **`taxa_evasao_em`** permanece como **covariável de contexto municipal**. Não se usa `indice_risco_evasao` como preditor (risco de vazamento conceitual).

Definição formal: `docs/definicao_problema_e_escopo.md`.

| Artefato | Descrição |
|---|---|
| `ml/baseline_municipio.py` | `fato_integrado`, `ColumnTransformer` + `Pipeline`, **HistGradientBoosting**, **DecisionTree**, **KNN**, split temporal; análise de ausentes ainda com **Ridge** |
| `ml/educational_ml.py` | Suite completa: comparação dos três regressores, **RandomizedSearchCV** do HGB, **TimeSeriesSplit** por ano, **MLflow**, função de inferência (`predict_taxa_abandono_em`), bundle final `.pkl`, CSV/JSON em `outputs/ml/`, figuras em `outputs/figures/` |
| `ml/mlflow_tracking.py` | Integração **MLflow**: parâmetros, métricas, múltiplos runs aninhados, registro do modelo final em `mlruns/` |
| `ml/mlflow_experiments.py` | Plano **EXPERIMENT_PLAN** (30 rodadas: 10× HGB, 10× árvore, 10× KNN); helpers de pipeline e registro MLflow; ver `mlruns/experimentos_mlflow_parametros.ipynb` |
| `mlruns/experimentos_mlflow_parametros.ipynb` | **Treino explícito** no notebook: loop por algoritmo/hiperparâmetros + registro no experimento `evasao_treino_parametros` |
| `ml/scenario_simulation.py` | Simulação what-if no dashboard: altera indicadores escola–ano e reutiliza `predict_taxa_abandono_em()`; narrativa automática de impacto |
| `notebooks/modelagem_evasao_municipio.ipynb` | EDA, baseline HGB (§5), comparação (§7), tuning + validação cruzada temporal + inferência do modelo final |

A função **`run_missing_impact_analysis`** (mesmo módulo) compara métricas do Ridge **com todas as features** versus **sem colunas muito incompletas** (taxa de ausência no treino acima de um limiar, por padrão 50%). Isso ajuda a avaliar sensibilidade do modelo à imputação e à presença de covariáveis esparsas — sem substituir o desenho principal baseado no Pipeline completo.

**Validação:** treino nos anos `≤ 2017`, teste nos anos `≥ 2018` (split temporal). O modelo final usa **RandomizedSearchCV** no treino com **TimeSeriesSplit por ano**, seguido de avaliação no teste holdout, exportação para o dashboard e **registro MLflow** via `python -c "from ml.educational_ml import run_educational_ml_suite; run_educational_ml_suite()"`.

**MLflow:** após a suite, consulte `bash scripts/mlflow_ui.sh` (local) ou `docker compose up mlflow` (Docker). Backend: **`sqlite:///mlruns/mlflow.db`**. Experimentos: `evasao_escolar_escola_ano` (suite principal) e `evasao_treino_parametros` (campanha do notebook com 30 runs); modelo registrado: `evasao_abandono_em_final`.

> Se você registrou experimentos antes da migração para SQLite, rode novamente `run_educational_ml_suite()` para popular o banco.

**Métricas reportadas:** MAE (principal), RMSE e R² no conjunto de teste, além de média / desvio por fold na validação cruzada temporal.

**Inferência mínima:** `ml.educational_ml.predict_taxa_abandono_em(...)` carrega o bundle salvo em `outputs/ml/final_model_bundle.pkl` e prevê `taxa_abandono_em` para novos dados com as mesmas features do pipeline.

**Simulação de cenários (dashboard):** na página 5, seção *O que aconteceria se?* — selecione escola/ano, ajuste indicadores com sliders e compare previsão original vs. simulada (mesmo fluxo de inferência, sem retreinamento). Ver `docs/historias_epicas_p6.md` (funcionalidade 4).

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
| `scikit-learn` | Pipeline de pré-processamento, tuning (`RandomizedSearchCV`), validação temporal e modelos de regressão / clusterização (**versão fixada: 1.7.2** — ver [Como rodar o projeto](#como-rodar-o-projeto)) |
| `matplotlib` / `seaborn` | Visualizações estáticas (notebook) |
| `plotly` | Gráficos interativos (dashboard) |
| `streamlit` | Interface do dashboard web |
| `sqlite3` | Banco de dados local |
| `jupyter` | Análise exploratória |
