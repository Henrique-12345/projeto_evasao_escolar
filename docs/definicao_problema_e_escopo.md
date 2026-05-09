# Definição formal do problema e escopo (Machine Learning)

**Projeto:** Evasão escolar — Recife (PE)  
**Documento:** Requisito 1 — problema de aprendizado supervisionado vinculado à solução proposta  
**Versão:** 1.2 — alinhamento explícito do target de ML à granularidade **escola–ano** (`taxa_abandono_em`).  

---

## 1. Contexto e objetivo

O **objeto de estudo** do projeto permanece a **evasão escolar** (saída definitiva do sistema de ensino), compreendida em conjunto com abandono, reprovação e defasagem na **cadeia de risco** descrita no produto de análise (dashboard e relatórios).

Para **aprendizado supervisionado** em `fato_integrado`, adota-se formalmente o seguinte enunciado:

> **Estimar a taxa de abandono escolar no Ensino Médio como indicador associado ao risco de evasão escolar.**

**Por que não usar `taxa_evasao_em` como alvo em nível escola–ano:** na base integrada, a evasão municipal vem da série socioeconômica e é **replicada para todas as escolas do mesmo ano**. Assim, entre linhas escola–ano **não há variação** dessa variável — violando a consistência estatística de um problema de regressão cujo alvo deveria diferir entre observações. O abandono no EM, por outro lado, está disponível **por escola e ano** e varia entre escolas.

**Relação evasão × abandono:** na literatura e nos fluxos do INEP, o abandono no ano letivo é tratado como **evento precursor** e fortemente associado à evasão definitiva; não é substituto conceitual da evasão, mas **proxy operacional adequado** ao nível de agregação disponível para modelagem.

A solução no repositório combina **ETL**, **análise exploratória**, **dashboard** e **regressão supervisionada** com escopo e limitações abaixo.

---

## 2. Formulação do problema de aprendizado de máquina

| Elemento | Definição adotada |
|----------|-------------------|
| **Tipo de problema** | Aprendizado **supervisionado**, tarefa de **regressão** (variável alvo contínua em escala de percentual). |
| **Objetivo da função aprendida (baseline)** | Regressão dos valores observados de **`taxa_abandono_em`** em nível **escola–ano**, dados covariáveis (indicadores educacionais por escola e **taxas municipais** repetidas por ano). **`taxa_evasao_em`** entra apenas como **feature de contexto municipal**, não como alvo. |
| **Interpretação da saída** | Percentuais em [0, 100] coerentes com o INEP — estimativa da **taxa de abandono agregada à escola naquele ano**, não probabilidade individual nem “probabilidade de evasão por aluno”. |

Esta formulação separa claramente: (i) **modelagem municipal da evasão** — possível com `dim_integrado_anual` / série **município–ano**; (ii) **modelagem escola–ano** — exige alvo que **varie entre escolas**, cumprido por **`taxa_abandono_em`**.

---

## 3. Unidade de observação e escopo de modelagem

| Nível | Tabela / uso | Papel |
|-------|----------------|-------|
| **Escola–ano** | **`fato_integrado`** — join da educacional (linha por escola-ano) com taxas socio **por `ano`** | **Principal para ML:** uma observação por escola e ano; indicadores educacionais são específicos; socioeconômicos municipais **replicados**. Identificador técnico: `id_linha_educacional` (o extrato bruto **não** traz código/nome de escola). |
| **Município–ano** | **`dim_integrado_anual`** (e `dim_socio_anual` / `dim_educ_anual`) | Séries temporais e KPIs com **uma linha por ano**; evita duplicar o mesmo ano em centenas de linhas nos gráficos agregados. |

**Alvo principal do baseline implementado:** `taxa_abandono_em` (precursor operacional, variável entre escolas).

**Contexto municipal:** `taxa_evasao_em`, `taxa_promocao_*`, `taxa_repetencia_*` etc. vêm da série socio e são **constantes entre escolas do mesmo ano** na `fato_integrado`.

---

## 4. Variável-alvo (target)

| Nome | Descrição | Uso no baseline |
|------|-----------|-----------------|
| **`taxa_abandono_em`** | Abandono no Ensino Médio (%) na escola | **Alvo de regressão** no módulo `ml/baseline_municipio.py` |
| **`taxa_evasao_em`** | Evasão municipal EM (%) | **Covariável** (não alvo no baseline atual); repetida por escola no mesmo ano |

A taxa de abandono escolar do Ensino Médio foi utilizada como variável-alvo devido à sua **disponibilidade em nível escola–ano**, à **variação entre escolas** e à **forte relação com o risco de evasão escolar** na literatura de fluxo escolar — como **indicador preditivo associado** à evasão, não como substituto conceitual da evasão municipal.

Modelos futuros voltados estritamente à **evasão municipal** devem usar **`dim_integrado_anual`** ou agregar por ano; usar `fato_integrado` com `taxa_evasao_em` como alvo por linha **duplica o mesmo rótulo** em todas as escolas do ano.

---

## 5. Dados de entrada (features)

### 5.1 Princípios

- Não usar **`indice_risco_evasao`** como preditor de um alvo que já incorpora partes da mesma informação sem critério de causalidade temporal (risco de vazamento conceitual).
- Excluir **`id_linha_educacional`** do conjunto de features (apenas identificador).

### 5.2 Conjunto ilustrativo na `fato_integrado`

Indicadores educacionais por escola (ex.: `tdi_em`, `taxa_reprovacao_em`, `atu_em`, …) e indicadores socio municipais replicados (`taxa_promocao_em`, `taxa_evasao_em`, …), mais `periodo` / `ano` conforme o notebook.

---

## 6. Métricas de avaliação e justificativa

(inalterado em espírito — MAE / RMSE / R² em pontos percentuais.)

**Validação:** manter **split temporal por ano** (treino em anos iniciais, teste em anos finais). Com **muitas linhas escola–ano**, o teste contém várias escolas por ano; métricas refletem erro **micro** agregado no período de teste.

---

## 7. Limitações explícitas do escopo

1. **Sem código INEP de escola** no CSV disponível — generalização para “escola X” é limitada à linha do extrato.  
2. **`taxa_evasao_em` municipal repetida** no mesmo ano: não confundir com evasão específica da escola.  
3. **Inferência causal** não reivindicada.

---

## 8. Coerência com o produto (dashboard / integração futura)

O dashboard usa **`dim_integrado_anual`** para séries **município–ano** e **`fato_integrado`** onde faz sentido analisar **dispersão escola–ano** (ex.: correlações micro).

---

## Referências internas ao repositório

- ETL: `etl/etl_pipeline.py` — gera `fato_integrado`, `dim_integrado_anual`, `fato_educacional`, etc.  
- Dicionário de variáveis: `README.md`.  
- Interface: `dashboard/app.py`.
