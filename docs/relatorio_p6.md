# Relatório P6 — Atendimento aos requisitos de Machine Learning

**Projeto:** Evasão escolar — Recife (PE)  
**Documento:** Consolidação de como os dez requisitos acadêmicos de ML foram cumpridos  
**Data de referência:** maio/2026  
**Repositório:** `projeto_evasao_escolar`

---

## Sumário executivo

O projeto implementa um **pipeline reprodutível** de dados abertos do INEP até um **produto de apoio à decisão** (dashboard Streamlit), com **regressão supervisionada** em granularidade **escola–ano** para estimar `taxa_abandono_em`, comparar algoritmos, ajustar hiperparâmetros, validar no tempo e integrar inferência ao painel.

| # | Requisito | Status | Evidência principal |
|---|-----------|--------|---------------------|
| 1 | Definição formal do problema de ML | **Atendido** | `docs/definicao_problema_e_escopo.md` |
| 2 | Plano técnico de dados e viabilidade | **Atendido** | `docs/plano_tecnico_dados.md` |
| 3 | Notebook: carga, EDA e transformações | **Atendido** | `analise_evasao_escolar.ipynb`, `notebooks/modelagem_evasao_municipio.ipynb`, `notebooks/analyse_missing_values.ipynb` |
| 4 | Modelo inicial simples + métricas | **Atendido** | `ml/baseline_municipio.py` — KNN, árvore, HGB; split temporal |
| 5 | ≥2 modelos distintos + comparação visual | **Atendido** | Três regressores + figuras de comparação |
| 6 | Ajuste de hiperparâmetros | **Atendido** | `RandomizedSearchCV` em `ml/educational_ml.py` |
| 7 | Validação cruzada + curva de aprendizado | **Atendido** | CV temporal por ano; diagnóstico de overfitting |
| 8 | Inferência / integração ao produto | **Atendido** | `predict_taxa_abandono_em()` + dashboard (sem endpoint HTTP) |
| 9 | Pipeline completo com modelo final | **Atendido** | `run_educational_ml_suite()` + artefatos em `outputs/ml/` |
| 10 | Comunicação clara e coerência metodológica | **Atendido** | README, docs, JSON narrativo, dashboard |

**Observação metodológica:** o problema foi definido como **regressão** (alvo contínuo em percentual). Por isso, métricas e gráficos seguem MAE, RMSE, R², observado × previsto e distribuição de erros — e **não** matriz de confusão nem curvas ROC/AUC, que se aplicam a **classificação**.

---

## Requisito 1 — Definição formal do problema de aprendizado de máquina

### O que o requisito pede

Tipo de problema, variável-alvo, estrutura dos dados de entrada, métricas de avaliação e justificativa de valor para a aplicação.

### Como foi cumprido

O documento **`docs/definicao_problema_e_escopo.md`** (versão 1.2) formaliza:

| Elemento | Definição adotada |
|----------|-------------------|
| **Tipo de problema** | Aprendizado **supervisionado** — **regressão** (percentual contínuo). Complementar: **clusterização** (KMeans) em covariáveis para perfis, sem usar o alvo nos centróides. |
| **Variável-alvo** | **`taxa_abandono_em`** (abandono no Ensino Médio, %), por escola e ano. |
| **Justificativa do alvo** | `taxa_evasao_em` municipal é **replicada** para todas as escolas do mesmo ano em `fato_integrado` — não varia entre escolas e **não** é alvo válido em nível escola–ano. O abandono varia entre escolas e é **precursor operacional** associado à evasão. |
| **Dados de entrada** | Linhas de **`fato_integrado`**: indicadores educacionais por escola + taxas municipais alinhadas por `ano`. Exclusões: `indice_risco_evasao` (vazamento), `id_linha_educacional` (ID). |
| **Métricas** | **MAE** (erro médio em pontos percentuais), **RMSE** (penaliza erros grandes), **R²** (variância explicada). |
| **Validação** | **Split temporal por ano** (treino em anos iniciais, teste em anos posteriores). |
| **Valor para a aplicação** | Permite **priorizar** escolas com maior abandono previsto e apoiar **prevenção** de evasão, integrado ao dashboard e ao ranking de risco. |

### Evidências no código

- Alvo e exclusões: `ml/baseline_municipio.py` (`TARGET`, `EXCLUDE_FROM_FEATURES`).
- Enunciado na suite: `ml/educational_ml.py` (docstring do módulo).

---

## Requisito 2 — Plano técnico de dados e viabilidade de ML

### O que o requisito pede

Fontes de dados, formatos, volume estimado, desafios de coleta/preparação e análise preliminar de viabilidade de ML.

### Como foi cumprido

O documento **`docs/plano_tecnico_dados.md`** (versão 1.1) contém:

**Fontes**

- Dados **abertos** INEP/MEC: `data/raw/dados_educacionais_recife.csv` e `data/raw/dados_socioeconomicos_recife.csv` (município Recife, IBGE 2611606).

**Formatos e volume (acervo atual)**

| Arquivo | Linhas aprox. | Granularidade |
|---------|---------------|---------------|
| Educacional | ~246 registros | Escola–ano (múltiplas linhas por ano) |
| Socioeconômico | ~64 registros | Série municipal/temporal |
| Interseção temporal | 15 anos (2008–2022) | Inner join por `ano` |

**Processamento**

- ETL: `etl/etl_pipeline.py` → CSV em `data/processed/` e SQLite `evasao_escolar.db`.
- Tabela principal para ML: **`fato_integrado.csv`**.

**Desafios documentados**

- Desalinhamento de anos, duplicatas, ausentes, agregação municipal, **data leakage**, série curta no nível município–ano.

**Viabilidade de ML**

- **Viável com ressalvas** para regressão de `taxa_abandono_em` em escola–ano (centenas de observações).
- Evasão municipal como **feature**, não como alvo escola–ano.
- Política de ausentes: `docs/politica_dados_ausentes.md`.

---

## Requisito 3 — Notebook: carga, EDA, tratamento e pipeline reprodutível

### O que o requisito pede

Carregamento, análise exploratória, limpeza, normalização, encoding, tratamento de outliers, reprodutibilidade e modularização.

### Como foi cumprido

**Notebooks**

| Notebook | Papel |
|----------|--------|
| **`analise_evasao_escolar.ipynb`** (raiz) | EDA integrada das bases brutas e processadas; evolução temporal, pandemia, correlações; figuras em `imagens/`. |
| **`notebooks/modelagem_evasao_municipio.ipynb`** | Roteiro ML: carga de `fato_integrado`, EDA do alvo e features, split temporal, pipelines, comparação, tuning, CV, inferência. |
| **`notebooks/analyse_missing_values.ipynb`** | Completude, ausentes e política de imputação alinhada ao ETL/ML. |

**Modularização (boa prática)**

- Lógica reutilizável em **`etl/etl_pipeline.py`**, **`ml/baseline_municipio.py`** e **`ml/educational_ml.py`** — notebooks importam módulos e documentam execução a partir da **raiz do repositório**.
- Dependências fixadas em **`requirements.txt`**.

**Tratamentos implementados**

| Etapa | Onde | Detalhe |
|-------|------|---------|
| Limpeza / deduplicação | `etl/etl_pipeline.py` | `drop_duplicates()`; validação de `id_linha_educacional` |
| Limites de taxas (0–100) | ETL | `clip(lower=0, upper=100)` em colunas de percentual |
| Exclusão de linhas sem alvo | `prepare_temporal_supervised_split()` | `dropna(subset=[TARGET])` |
| Imputação | `build_preprocess_transformer()` | `SimpleImputer`: mediana (numéricas), moda (categóricas) — **só no treino** via `Pipeline` |
| Normalização | Pipeline numérico | `StandardScaler` |
| Encoding | Pipeline categórico | `OneHotEncoder(handle_unknown="ignore")` |
| Outliers | ETL (winsorização suave) | *Clip* em [0, 100] para taxas; não remoção arbitrária de linhas por IQR (documentado como escolha conservadora para volumes pequenos) |

**Reprodutibilidade**

- `random_state=42` nos modelos estocásticos.
- Comando documentado para regenerar artefatos:

```bash
python3 -c "from ml.educational_ml import run_educational_ml_suite; run_educational_ml_suite()"
```

---

## Requisito 4 — Modelo inicial simples e primeiras métricas

### O que o requisito pede

Treinar um modelo inicial adequado ao tipo de problema (ex.: KNN, regressão linear, árvore) e reportar métricas em conjunto de validação.

### Como foi cumprido

Em **`ml/baseline_municipio.py`**:

- **`run_baseline_experiment()`** — referência linear com **Ridge** (modelo parsimonioso inicial).
- Pipelines para **HistGradientBoostingRegressor**, **DecisionTreeRegressor** e **KNeighborsRegressor**.

Split de validação:

- **Treino:** anos `≤ 2017`
- **Teste (holdout):** anos `≥ 2018`
- Função: `prepare_temporal_supervised_split(year_cutoff=2017)`

Métricas: `evaluate_regression()` → MAE, RMSE, R².

**Exemplo de resultados no holdout** (exportados em `outputs/ml/ml_storytelling.json` após execução da suite):

| Modelo | MAE (p.p.) | RMSE | R² |
|--------|------------|------|-----|
| HistGradientBoosting | 0,55 | 0,70 | 0,91 |
| DecisionTreeRegressor | 0,53 | 0,67 | 0,92 |
| KNeighborsRegressor | 0,85 | 1,47 | 0,60 |

O notebook **`modelagem_evasao_municipio.ipynb`** (seções iniciais de modelagem) reproduz e exibe essas etapas.

---

## Requisito 5 — Pelo menos dois algoritmos distintos e comparação visual

### O que o requisito pede

Comparar desempenho com as métricas definidas e incluir visualizações (matriz de confusão, ROC, gráficos de erro, etc.).

### Como foram cumpridos

**Três regressores distintos** na mesma partição temporal:

1. **HistGradientBoostingRegressor** — modelo principal preditivo.
2. **DecisionTreeRegressor** — interpretabilidade por regras.
3. **KNeighborsRegressor** — similaridade entre escolas.

**Quarto algoritmo (complementar):** **KMeans** — clusterização de perfis (não supervisionada nas covariáveis transformadas).

**Comparação tabular**

- `metrics_comparison_dataframe()` e `ml_storytelling.json` → chave `"metrics"`.
- CSV enriquecido: `outputs/ml/escola_ano_ml_enriquecido.csv` (previsões por modelo e ranking).

**Visualizações** (geradas por `plot_model_comparison_figures()` e funções em `educational_ml.py`):

| Figura | Conteúdo |
|--------|----------|
| `model_comparison_mae_rmse.png` | Barras MAE/RMSE por modelo |
| `model_comparison_r2.png` | R² comparativo |
| `model_comparison_obs_vs_pred.png` | Observado × previsto (erro visual) |
| `model_comparison_error_boxplot.png` | Distribuição de resíduos |
| `ml_hgb_obs_vs_pred.png`, `ml_hgb_feature_importance.png` | Modelo principal |
| `ml_decision_tree.png` | Árvore limitada em profundidade |
| `ml_kmeans_*.png` | Cotovelo, silhueta, PCA |

*Nota:* por ser **regressão**, não há matriz de confusão nem ROC/AUC; os gráficos equivalentes são **observado × previsto**, **boxplot de erro** e **importância de variáveis** (permutation importance no HGB).

**Critério de escolha para priorização:** menor MAE no teste entre regressores; o **modelo final** é o HGB **ajustado** (requisito 6), por melhor equilíbrio entre desempenho preditivo após tuning e uso operacional (ranking), mesmo quando a árvore simples tem MAE ligeiramente menor no holdout pontual.

---

## Requisito 6 — Ajuste de hiperparâmetros

### O que o requisito pede

Grid Search, Random Search ou Optuna; registrar parâmetros avaliados e resultados.

### Como foi cumprido

Função **`_tune_hist_gradient_boosting()`** em **`ml/educational_ml.py`**:

- Técnica: **`RandomizedSearchCV`**
- Estimador base: pipeline HGB + pré-processamento
- Critério: **MAE médio de validação** (`scoring="neg_mean_absolute_error"`)
- **`n_iter=18`** combinações amostradas

**Espaço de hiperparâmetros avaliado**

| Parâmetro | Valores explorados |
|-----------|-------------------|
| `learning_rate` | 0,03 – 0,15 |
| `max_depth` | 2 – 5 |
| `max_iter` | 80 – 220 |
| `min_samples_leaf` | 5 – 24 |
| `l2_regularization` | 0,0 – 1,0 |
| `max_leaf_nodes` | 7, 15, 31 |

**CV durante a busca:** folds **temporais por ano** (`_build_year_based_time_series_splits`), não aleatórios — coerente com dados de painel.

**Artefatos**

- `outputs/ml/final_model_tuning_results.csv` — ranking de candidatos, MAE médio e desvio por fold de busca.
- `ml_storytelling.json` → `final_model_best_params`, `final_model_best_cv_mae`.

**Melhores parâmetros encontrados (execução de referência)**

```json
{
  "model__min_samples_leaf": 12,
  "model__max_leaf_nodes": 15,
  "model__max_iter": 220,
  "model__max_depth": 4,
  "model__learning_rate": 0.1,
  "model__l2_regularization": 0.3
}
```

**Optuna:** citada como evolução opcional em `plano_tecnico_dados.md`; não implementada — **RandomizedSearchCV** atende ao requisito.

---

## Requisito 7 — Validação cruzada, variância, overfitting e curva de aprendizado

### O que o requisito pede

k-fold ou stratified k-fold; análise de variância; identificação de over/underfitting; curvas de aprendizado ou análise de erro.

### Como foi cumprido

**Estratégia de CV:** validação cruzada **temporal por ano** (adaptação de `TimeSeriesSplit` aos índices de linhas), com **4 folds** no treino (anos `≤ 2017`). Cada fold valida em anos **posteriores** aos do treino dentro do conjunto de treino — evita misturar o mesmo ano em treino e validação.

**Funções**

- `_summarize_cross_validation()` — `cross_validate` com MAE, RMSE, R²; scores de treino e teste por fold.
- `_learning_curve_dataframe()` — curva de aprendizado com tamanhos crescentes do treino.
- Diagnóstico textual: `final_model_cv_diagnosis` no JSON.

**Artefatos**

| Arquivo | Conteúdo |
|---------|----------|
| `outputs/ml/final_model_cv_folds.csv` | Métricas por fold |
| `outputs/ml/final_model_learning_curve.csv` | MAE treino/validação vs tamanho do treino |
| `outputs/figures/ml_final_learning_curve_mae.png` | Visualização (após `run_educational_ml_suite`) |

**Resultados de estabilidade (execução de referência)**

- MAE médio de validação na CV do tuning: **~3,15 p.p.** (desvio **~1,92**).
- Diagnóstico registrado: *"Há instabilidade temporal relevante: os folds variam bastante… possível overfitting."*
- Holdout teste (anos ≥ 2018) do modelo final ajustado: MAE **~0,74**, R² **~0,82** — melhor que a média dos folds de CV no treino temporal, mas a **variância entre folds** é explicitada para não superestimar generalização.

**Honestidade metodológica:** o relatório não oculta que R² médio na CV pode ser negativo em alguns folds — típico de séries curtas e mudança de regime (pandemia, políticas). Isso atende ao requisito de **identificar problemas** de generalização.

---

## Requisito 8 — Função de inferência ou integração mínima ao produto

### O que o requisito pede

Função de inferência ou estrutura mínima de integração (pode ser endpoint HTTP).

### Como foi cumprido

**Inferência em Python**

```python
from ml.educational_ml import predict_taxa_abandono_em

pred = predict_taxa_abandono_em(df_novas_linhas)
# coluna gerada: pred_taxa_abandono_em
```

- Carrega **`outputs/ml/final_model_bundle.pkl`** (`load_final_model_bundle()`).
- Valida presença das **mesmas colunas de features** usadas no treino.
- Bundle contém: modelo refitado em todos os dados observados, lista de features, hiperparâmetros, metadados de anos.

**Integração ao produto (dashboard)**

- **`dashboard/app.py`** → `render_ml_inteligencia_section()` (página 5 — *Apoio inteligente*).
- **`dashboard/app.py`** → `render_simulacao_cenarios_section()` (página 5 — *O que aconteceria se?*): simulação what-if com sliders, reutilizando `predict_taxa_abandono_em()` e o bundle serializado.
- Lê `outputs/ml/ml_storytelling.json`, CSVs e figuras geradas pela suite.
- Exibe métricas, tuning, CV, árvore, KNN, KMeans, ranking e **simulação de intervenções** — **sem exigir que o gestor execute Python**.

**O que não foi implementado**

- **Endpoint HTTP** (FastAPI/Flask): não obrigatório pelo enunciado (“*pode* consistir em endpoint”). A integração mínima está na **função callable + dashboard**.

**Como regenerar integração**

```bash
python3 -c "from ml.educational_ml import run_educational_ml_suite; run_educational_ml_suite()"
streamlit run dashboard/app.py
```

---

## Requisito 9 — Pipeline completo com modelo final e evidências de estabilidade

### O que o requisito pede

Executar pipeline com dados atualizados, modelo final ajustado, métricas e evidências de estabilidade.

### Como foi cumprido

**Orquestrador:** `run_educational_ml_suite()` em **`ml/educational_ml.py`**

Fluxo:

1. ETL implícito (`ensure_processed_data()` se CSV ausente).
2. Split temporal e preparação de features.
3. Treino e avaliação dos três regressores no holdout.
4. KMeans + perfis + figuras.
5. **RandomizedSearchCV** → modelo final HGB.
6. **CV temporal** + **curva de aprendizado** no modelo final.
7. Exportação CSV/JSON/PNG e **`final_model_bundle.pkl`**.
8. Refit do modelo final em **todos** os anos disponíveis para inferência operacional.

**Métricas do modelo final (holdout teste, referência)**

| Métrica | Valor |
|---------|------:|
| MAE | 0,74 p.p. |
| RMSE | 0,99 |
| R² | 0,82 |

**Evidências de estabilidade / limitações**

- Tabela por fold: `final_model_cv_folds.csv`
- Resumo: `final_model_cv_summary` no JSON (média e desvio-padrão de MAE/RMSE/R² entre folds)
- Curva de aprendizado: `final_model_learning_curve.csv`
- Ranking exportado: `rank_risco_abandono_previsto` em `escola_ano_ml_enriquecido.csv`

**Reprodutibilidade:** notebook `modelagem_evasao_municipio.ipynb` §7–§9 executa a mesma suite e demonstra inferência com `predict_taxa_abandono_em`.

---

## Requisito 10 — Comunicação clara e coerência metodológica

### O que o requisito pede

Problema, pipeline, algoritmos, critério de escolha do modelo final, resultados, integração ao produto; generalização e aderência ao contexto.

### Como foi cumprido

| Aspecto | Onde está documentado |
|---------|----------------------|
| Definição do problema | `docs/definicao_problema_e_escopo.md`, README |
| Pipeline de dados | `docs/plano_tecnico_dados.md`, `etl/etl_pipeline.py`, README |
| Algoritmos e papéis | `ml/educational_ml.py` (docstring), notebook de modelagem |
| Critério do modelo final | HGB após RandomizedSearchCV; MAE na CV temporal; uso em ranking |
| Resultados numéricos | `outputs/ml/ml_storytelling.json`, CSVs, figuras |
| Integração ao produto | `dashboard/app.py`, `docs/historias_epicas_p6.md` (quatro funcionalidades, incl. simulação) |
| Narrativa em linguagem acessível | `narrativa_comparacao_regressao`, `kmeans_interpretacao`, `arvore_regras_resumo` no JSON |

**Coerência metodológica**

- Mesmo alvo, mesmas exclusões de features e mesmo pré-processamento em todos os modelos comparados.
- Imputação e escala **apenas no treino** (sem vazamento para o teste).
- Validação **temporal**, não aleatória — adequada a indicadores anuais.
- Limitações explícitas: proxy de abandono vs evasão municipal; instabilidade entre folds; volume finito de escolas.

**Capacidade de generalização**

- Avaliada no **holdout temporal** e na **CV por ano**; diagnóstico de overfitting registrado.
- Não se reivindica inferência causal — apenas **predição associativa** para priorização.

**Aderência ao contexto educacional**

- Dashboard traduz scores, evolução e ML para gestores (`docs/historias_epicas_p6.md`).
- Árvore e KMeans apoiam **explicação** e **segmentação**; HGB apoia **priorização**.

---

## Mapa de artefatos por requisito

```
data/raw/                          → Requisito 2 (fontes)
etl/etl_pipeline.py                → Requisitos 2, 3
docs/definicao_problema_e_escopo.md → Requisito 1
docs/plano_tecnico_dados.md         → Requisito 2
analise_evasao_escolar.ipynb        → Requisito 3
notebooks/modelagem_evasao_municipio.ipynb → Requisitos 3–9
ml/baseline_municipio.py            → Requisitos 3, 4, 5
ml/educational_ml.py                → Requisitos 5–9
outputs/ml/*.csv, *.json, *.pkl     → Requisitos 5–9
dashboard/app.py                    → Requisitos 8, 10
docs/historias_epicas_p6.md         → Requisito 10 (produto + simulação)
ml/scenario_simulation.py           → Requisito 8 (inferência interativa)
```

---

## Conclusão

Os **dez requisitos** do escopo P6 de Machine Learning foram atendidos no repositório, com documentação formal (`docs/`), código modular (`etl/`, `ml/`), notebooks reprodutíveis, artefatos exportados e integração ao dashboard. As adaptações ao tipo de problema (**regressão** em vez de classificação) estão justificadas e refletidas nas métricas e visualizações escolhidas.

**Pontos de atenção para avaliação / evolução**

1. Regenerar figuras em `outputs/figures/` localmente se não estiverem versionadas no Git.
2. Executar **Run All** nos notebooks antes de entregar para garantir outputs atualizados.
3. Endpoint REST permanece opcional; a inferência via função Python + Streamlit cumpre integração mínima.
4. Interpretar métricas de CV com cautela diante da **instabilidade temporal** documentada.

---

## Referências rápidas

| Recurso | Caminho |
|---------|---------|
| Definição do problema | `docs/definicao_problema_e_escopo.md` |
| Plano técnico | `docs/plano_tecnico_dados.md` |
| Funcionalidades do gestor | `docs/historias_epicas_p6.md` |
| Suite ML | `ml/educational_ml.py` |
| Baseline e pipelines | `ml/baseline_municipio.py` |
| Dashboard | `dashboard/app.py` |
| README do projeto | `README.md` |
