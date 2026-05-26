# Plano técnico de dados

**Projeto:** Evasão escolar — Recife (PE)  
**Documento:** Requisito 2 — fontes, formatos, volume, desafios e viabilidade de ML  
**Versão:** 1.1 — alinhamento ao target **`taxa_abandono_em`** em granularidade escola–ano.  

---

## 1. Objetivo deste documento

Registrar **de onde vêm os dados**, em **que formato** chegam, **quanto volume** existe no acervo atual do repositório, **quais obstáculos** surgem na coleta/agrupamento e se **técnicas de aprendizado de máquina são viáveis** no escopo definido em `definicao_problema_e_escopo.md`.

---

## 2. Fontes de dados

| Origem | Descrição | Tipo de acesso |
|--------|-----------|----------------|
| **INEP / MEC** — indicadores educacionais e censos escolares | Dados agregados para o **município de Recife** (código IBGE **2611606**), incluindo indicadores de fluxo, infraestrutura e desempenho por estabelecimento/ano conforme disponibilização nas extrações utilizadas. | **Dados abertos** (uso conforme licenças e normas do INEP). |
| **INEP / MEC** — série socioeconômica / fluxo escolar | Taxas de promoção, repetência e **evasão** em nível compatível com o recorte municipal. | **Dados abertos**. |

**Observação:** O repositório não armazena URLs de download versionadas; a **rastreabilidade** deve ser complementada na entrega acadêmica (data da extração, nome do conjunto no portal de dados abertos, filtros aplicados).

---

## 3. Arquivos brutos no repositório

| Arquivo (relativo a `data/raw/`) | Conteúdo | Formato | Codificação |
|----------------------------------|----------|---------|-------------|
| `dados_socioeconomicos_recife.csv` | Taxas de promoção, repetência e evasão (EF e EM), entre outras colunas do dicionário | CSV delimitado por vírgula | UTF-8 (recomendado; validar ao abrir) |
| `dados_educacionais_recife.csv` | ATU, HAD, TDI, aprovação, reprovação, abandono (EF e EM), etc. | CSV | UTF-8 |

**Metadados explícitos nas colunas:** `ano`, `id_municipio`, `id_municipio_nome` em ambos os arquivos (chaves para junção e filtro por Recife).

---

## 4. Volume estimado (acervo atual)

Medidas obtidas a partir dos CSV presentes em `data/raw/` (inclui cabeçalho nas contagens de linhas de arquivo):

| Métrica | Valor |
|---------|------:|
| Linhas totais `dados_educacionais_recife.csv` | 247 |
| Linhas totais `dados_socioeconomicos_recife.csv` | 65 |
| **Registros aproximados (excl. cabeçalho)** educacional | 246 |
| **Registros aproximados (excl. cabeçalho)** socioeconômico | 64 |

**Granularidade:**

- **Educacional:** múltiplas linhas por **ano** (uma por estabelecimento — ordem de centenas de registros no conjunto atual).
- **Socioeconômico:** múltiplas linhas por **ano** (estrutura compatível com série municipal/temporal).

**Sobreposição temporal (anos presentes em ambos os arquivos, deduplicados):**

- **15 anos** em interseção: **2008, 2009, …, 2022** (alinhado ao fechamento da base socioeconômica disponível).

Após o join **`escola–ano`**, a tabela **`fato_integrado`** passa a ter **uma linha por registro da base educacional** em cada ano presente na interseção com as taxas municipais; o volume típico é da ordem de **centenas de linhas** (detalhes nos logs do ETL). Para séries **uma linha por ano**, usar **`dim_integrado_anual`**.

| Métrica | Valor (referência ao CSV bruto atual) |
|---------|--------------------------------------|
| Linhas educacional | ~246 |
| Linhas socioeconômica | ~64 |
| **Amostras `fato_integrado` após inner join por ano** | Ver log do ETL (`Carregando... shape`) |

**Nota:** O arquivo educacional deste repositório **não inclui código nem nome de escola** do INEP; o ETL atribui **`id_linha_educacional`** para identificar unicamente cada observação escola–ano após limpeza.

---

## 5. Formatos após processamento (ETL)

O pipeline `etl/etl_pipeline.py` produz:

| Saída | Formato | Local |
|-------|---------|------|
| Várias tabelas em CSV | `.csv` em `data/processed/` | Ex.: `fato_integrado.csv`, `fato_educacional.csv`, etc. |
| Banco relacional leve | SQLite | `data/processed/evasao_escolar.db` |

Isso padroniza tipos numéricos, remove duplicatas iniciais, aplica *clip* em taxas (0–100), **mantém a granularidade escola–ano na educacional**, agrega por ano apenas onde necessário (`dim_socio_anual`, `dim_educ_anual`, **`dim_integrado_anual`**), faz **`merge` por ano** entre educacional e taxas municipais para gerar **`fato_integrado`**, e calcula índices derivados.

---

## 6. Desafios na coleta, preparação e uso

| Desafio | Descrição | Mitigação prevista |
|---------|-----------|-------------------|
| **Desalinhamento de anos** entre arquivos | Inner join reduz o universo aos anos comuns (~15). | Documentar anos perdidos; considerar **imputação** ou fontes adicionais apenas com critério explícito (evitar inventar dados). |
| **Duplicatas** | O ETL remove duplicatas completas (quantidades registradas em logs do pipeline na execução). | Manter ETL como única porta de entrada dos dados processados. |
| **Valores ausentes** | Colunas de fluxo podem vir vazias em certos anos/escolas (ex.: aprovação/reprovação). | Análise de completude no notebook; estratégias: exclusão de features, imputação simples ou modelos que aceitem missing (com justificativa). |
| **Agregação municipal** | As médias **por ano** em `dim_educ_anual` resumem o sistema; já **`fato_integrado`** preserva dispersão entre escolas. | README e dashboard distinguem tabela escola–ano vs `dim_integrado_anual`. |
| **Vazamento (*leakage*)** | Índices compostos que já incorporam informação do próprio constructo predito ou do alvo não devem ser features (ex.: `indice_risco_evasao` excluído no baseline). | Lista explícita de exclusões em `ml/baseline_municipio.py`. |
| **Série curta no nível município–ano** | n ≈ 15 para regressão multivariada é **delicado**. | Preferir validação **temporal**, modelos **parsimoniosos**, **regularização**, e honestidade nas conclusões sobre generalização. |

---

## 7. Análise preliminar de viabilidade de machine learning

### 7.1 Viável com ressalvas

- **Sim**, é viável treinar **regressão supervisionada** em **`fato_integrado`** com alvo **`taxa_abandono_em`** (disponível por escola–ano e variando entre escolas), usando covariáveis educacionais e municipais alinhadas por ano.
- A **evasão municipal (`taxa_evasao_em`)** pode **contextualizar** o cenário da cidade como **feature**, mas **não** deve ser o **alvo** em nível escola–ano (valor repetido por ano em todas as escolas).
- No nível **município–ano**, uma regressão com **`taxa_evasao_em`** como alvo permanece conceituável (`dim_integrado_anual`), mas com **n baixo** (~15 anos) — métricas devem ser interpretadas com cautela.

### 7.2 Complementaridade do nível escola–ano

- O arquivo educacional oferece **centenas de observações escola–ano**, o que **aumenta o n** para modelos com **`taxa_abandono_em`** como variável dependente.
- A **evasão municipal** **não** está disponível por escola neste conjunto — por isso o baseline **não** estima “evasão por escola”; estima-se **abandono EM**, proxy operacional ligado ao risco de evasão.

### 7.3 Conclusão

A aplicação de **ML é preliminarmente viável** para:

1. **Regressão da taxa de abandono no EM** em granularidade **escola–ano** (baseline em `ml/baseline_municipio.py`), com validação temporal e métricas MAE / RMSE / R² em pontos percentuais.  
2. **Análises municipais** da evasão em séries temporais e no dashboard — **sem confundir** com o alvo do modelo escola–ano.

Qualquer evolução para **microdados individuais** ou **outras regiões** exigiria **nova rodada** deste plano técnico (fontes, volume, LGPD, consistência).

---

## 8. Dependências de software (extração e pipeline)

Conforme `requirements.txt`: **pandas**, **numpy**, **SQLAlchemy** (uso no ecossistema), **streamlit** / **plotly** (produto de visualização) e **scikit-learn** para a etapa de ML já implementada no repositório (pipelines, regressão, KMeans, `RandomizedSearchCV`, `TimeSeriesSplit`). **Optuna** permanece opcional para evolução futura.

---

## Referências cruzadas

- Definição do problema e métricas: `docs/definicao_problema_e_escopo.md`  
- Pipeline: `etl/etl_pipeline.py`  
- Visão geral do projeto: `README.md`  
- Política de dados ausentes / imputação: `docs/politica_dados_ausentes.md`
