# Política de valores ausentes (missing values)

Este documento descreve como o projeto **preserva**, **documenta** e **trata** dados ausentes em cada etapa, para evitar distorções metodológicas e vazamento de informação entre treino e teste.

## Princípios

1. **No ETL**, valores não observados permanecem como **`NaN`** nas tabelas integradas (`fato_integrado`, fatos dimensionais etc.), salvo transformações que derivam novas colunas a partir de regras explícitas. **Não** se aplica imputação genérica no carregamento para CSV/SQLite.
2. **Índices e classificações de risco** não devem tratar ausência como **zero**. O **índice de risco de evasão** combina componentes com pesos que são **renormalizados** quando algum indicador está ausente (média ponderada apenas sobre componentes observados).
3. **Classificações ordinais** de risco (`risco_ef`, `risco_em`, níveis do score no dashboard) usam rótulo explícito **`Sem dado`** quando a taxa ou o score não pode ser calculado, em vez de empurrar o valor para um bin pelo uso implícito de zero.
4. **Na modelagem supervisionada**, a imputação é **sistemática** e **restrita ao conjunto de treino**: `SimpleImputer` com **mediana** (numéricas) e **valor mais frequente** (categóricas) permanece **dentro** do `sklearn.pipeline.Pipeline`, de modo que estatísticas de imputação são aprendidas apenas com `X_train` e aplicadas a `X_test` via `predict`, evitando **data leakage**.

## O que permanece ausente após o ETL

- Indicadores **não divulgados** ou **omitidos** nos microdados agregados do INEP/MEC em determinado ano ou escola.
- Campos derivados de divisões ou médias quando **faltam insumos**; onde o pipeline não define regra de substituição, o resultado segue ausente.
- Após **merge** escola–ano com série municipal, pode haver ausência **só no lado educacional** ou **só no socioeconômico** conforme o ano.

Esses `NaN` são **intencionalmente mantidos** para máxima rastreabilidade e para que análises de qualidade (relatório de ausentes, notebooks) quantifiquem o problema antes de qualquer modelo.

## O que só é tratado na modelagem

- Preenchimento de covariáveis para algoritmos que não aceitam `NaN` na matriz de desenho — feito pelo **Pipeline** no treino.
- Estudos de sensibilidade (por exemplo, **remover colunas com alta taxa de ausência no treino**) são opcionais e documentados em `ml/baseline_municipio.run_missing_impact_analysis`.

## Evidências e rastreabilidade

- Após cada execução bem-sucedida do ETL: **`outputs/relatorio_missing_values.csv`** com:
  - resumo global de células ausentes por tabela;
  - **formas** (linhas × colunas) **bruto vs processado**;
  - detalhe por coluna (contagem, percentual, ranking).
- Figuras opcionais em **`outputs/figures/`** (barras e heatmap amostral), se `matplotlib`/`seaborn` estiverem disponíveis.

## Limitações das bases públicas e pandemia

- **Reformulações censitárias e mudanças de instrumento** podem gerar lacunas ou quebras de série.
- **2020–2021**: distúrbios no censo escolar, ensino remoto e fechamentos afetam consistência e completude de alguns indicadores; ausências podem concentrar-se nesses anos.
- Interpretações preditivas devem considerar que **imputação** recupera covariáveis para o modelo, mas **não substitui** informação não coletada na fonte.

## Referências no código

| Componente | Arquivo |
|------------|---------|
| Relatório de ausentes | `etl/missing_report.py` |
| Índice de risco (pesos renormalizados) | `etl/etl_pipeline.py` (`_indice_risco_evasao`) |
| Score do dashboard (mesma lógica) | `dashboard/app.py` (`calcular_score`) |
| Pipeline ML + imputação | `ml/baseline_municipio.py` |
