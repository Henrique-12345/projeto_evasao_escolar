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

## Integração das Bases de Dados

As duas bases possuem granularidades diferentes: a base socioeconômica já vem com uma linha por ano, enquanto a base educacional tem uma linha por escola por ano — o que significa que um mesmo ano pode ter dezenas de registros, um para cada escola do município.

Para que as duas possam ser analisadas juntas, o ETL executa dois passos. Primeiro, a base educacional é agregada por ano: todos os indicadores de cada escola (TDI, ATU, aprovação, reprovação, abandono) são calculados como a média entre todas as escolas do Recife naquele ano. Isso transforma os registros individuais por escola em um único valor representativo por ano. Segundo, as duas bases — agora ambas com uma linha por ano — são unidas por meio de um `INNER JOIN` usando o campo `ano` como chave. Isso significa que apenas os anos presentes nas duas fontes aparecem na tabela integrada final.

O resultado é a tabela `fato_integrado`, que concentra em uma linha por ano todas as variáveis socioeconômicas (taxas de evasão, abandono, promoção, repetência) lado a lado com os indicadores educacionais agregados (TDI, ATU, HAD, aprovação, reprovação). A partir dessa tabela unificada, o ETL calcula também o **Índice de Risco de Evasão**: um score de 0 a 100 que combina evasão no Ensino Médio (peso 40%), distorção idade-série (peso 30%) e repetência no Ensino Médio (peso 30%).

Vale destacar uma limitação dessa abordagem: ao agregar por média, perde-se a variabilidade entre escolas. Uma escola com 0% de abandono e outra com 40% resultam na média de 20%, sem que a dispersão fique visível na tabela integrada. Por isso, o ETL também mantém a tabela `fato_educacional` com os dados brutos por escola, que é utilizada no dashboard para análises de distribuição individuais.

---

## Dashboard

O dashboard é organizado como uma **narrativa em 5 seções**, projetada para comunicar insights de forma clara para qualquer pessoa — técnica ou não. Cada visualização é acompanhada de texto explicativo antes e depois do gráfico.

| Seção | Pergunta respondida | Conteúdo principal |
|---|---|---|
| 1. Contexto Geral | Qual é o problema? | KPIs do ano mais recente, Score de Risco com gauge, diagnóstico automático, variação do período |
| 2. Evolução ao Longo do Tempo | Como o problema mudou ano a ano? | Série temporal de evasão e abandono, variação anual, comparação EF × EM, boxplots por período histórico |
| 3. Impacto da Pandemia | Por que 2020–2021 foram tão ruins? | Explicação dos 3 mecanismos (fechamento, ensino remoto, crise econômica), score antes/durante/após, comparação de indicadores por período |
| 4. Por que os Alunos Evadem? | Quais são as causas? | Cadeia causal (reprovação → TDI → abandono → evasão), correlações com scatter e tendência, diagnóstico por indicador, mapa de correlação |
| 5. Conclusões e Modelo Preditivo | O que fazer? Quais variáveis usar no modelo? | 5 insights consolidados, tabela de preditores para ML, plano de ação por urgência, projeção simplificada |

**Princípios do dashboard:**
- **Texto antes de cada gráfico** — explica o que será analisado e qual é o insight principal
- **Texto depois de cada gráfico** — interpreta os dados, explica as causas e destaca pontos críticos
- **Insights automáticos** — pior ano, melhor ano, maior alta e maior queda calculados automaticamente dos dados
- **Score de Risco 0–100** — Abandono EM (40%) + TDI (30%) + Reprovação EM (30%)
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
| Score de Risco | Indicador 0–100: Abandono EM (40%) + TDI EM (30%) + Reprovação EM (30%) |

---

## Principais Insights

1. **Ensino Médio é 2–3× mais crítico** que o EF — causas estruturais: pressão econômica, currículo percebido como distante e maior impacto de crises externas
2. **Cadeia causal confirmada pelos dados:** Reprovação → TDI (defasagem escolar) → Abandono → Evasão definitiva. Qualquer intervenção que quebre essa cadeia reduz a evasão
3. **Queda consistente de 2008 a 2019** — políticas educacionais e programas sociais produziram avanços reais; **pandemia de 2020–2021 reverteu parte desse progresso** em meses
4. **TDI e taxa de abandono são os preditores mais fortes** da evasão — variáveis prioritárias para o modelo de Machine Learning
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

## Tecnologias

| Biblioteca | Uso |
|---|---|
| `pandas` / `numpy` | Manipulação e transformação de dados |
| `matplotlib` / `seaborn` | Visualizações estáticas (notebook) |
| `plotly` | Gráficos interativos (dashboard) |
| `streamlit` | Interface do dashboard web |
| `sqlite3` | Banco de dados local |
| `jupyter` | Análise exploratória |
