# Modelo Adaptativo de Deep Learning para Redução de Viés em Previsões Meteorológicas

<div align="center">

**⚠️ PROJETO DE PESQUISA CONFIDENCIAL ⚠️**

*Este repositório contém código e metodologias de pesquisa em desenvolvimento*

</div>

## 📋 Informações do Projeto

**Orientadores:**
- Prof. Dr. José Laurindo Campos dos Santos
- Prof. Me. Renato Senna

**Instituição:** INPA - Instituto Nacional de Pesquisas da Amazônia

**Período:** 2024-2026

**Área de Pesquisa:** Meteorologia Computacional, Deep Learning, Previsão Numérica de Tempo

## 🎯 Objetivo

Este projeto de pesquisa visa desenvolver e aplicar modelos adaptativos baseados em Deep Learning (GraphCast/GenCast) para reduzir vieses sistemáticos em previsões meteorológicas de médio prazo, utilizando como base dados operacionais brasileiros:

- **MERGE** (Merged Precipitation Data - INPE/CPTEC)
- **FUNCEME** (Fundação Cearense de Meteorologia e Recursos Hídricos)

A abordagem combina técnicas de transfer learning com dados globais (ERA5) e adaptação regional para o território brasileiro, com foco especial na região Nordeste.

## 🔬 Fundamentação Científica

### Dados MERGE (INPE)
O MERGE é um produto de precipitação que combina estimativas de satélite (TRMM/GPM) com dados de pluviômetros em superfície, gerando campos de precipitação em grade com resolução espacial de 0.1° para a América do Sul. Este projeto utiliza séries históricas do MERGE para:
- Validação de previsões de precipitação
- Treinamento de camadas de ajuste fino
- Quantificação de vieses regionais

### Dados FUNCEME
A FUNCEME mantém uma rede densa de estações meteorológicas no Nordeste brasileiro, fornecendo dados observacionais de alta qualidade que são utilizados para:
- Benchmark de previsões regionais
- Identificação de padrões climáticos locais
- Calibração de modelos para fenômenos específicos (ZCIT, VCAN, etc.)

## 🧠 Metodologia

### 1. Arquitetura Base: GraphCast/GenCast
Utilizamos a arquitetura GraphCast (DeepMind, 2023) como modelo base, que emprega Graph Neural Networks (GNNs) para previsões meteorológicas em escala global.

### 2. Estratégia de Adaptação
```
[Modelo Pré-treinado ERA5] 
         ↓
[Transfer Learning]
         ↓
[Fine-tuning com MERGE + FUNCEME]
         ↓
[Camadas de Correção de Viés]
         ↓
[Modelo Adaptado Regional]
```

### 3. Redução de Viés
Implementação de técnicas específicas:
- **Bias Correction Layers**: Camadas neurais treinadas para corrigir vieses sistemáticos
- **Ensemble Weighting**: Ponderação adaptativa baseada em performance histórica
- **Regional Feature Extraction**: Extração de características climáticas regionais
- **Temporal Consistency Constraints**: Restrições para manter consistência física

## 📊 Datasets

### Dados Principais
| Dataset | Fonte | Resolução | Período | Uso |
|---------|-------|-----------|---------|-----|
| **ERA5** | ECMWF | 0.25° × 0.25° | 1979-presente | Pré-treinamento |
| **MERGE** | INPE/CPTEC | 0.1° × 0.1° | 2000-presente | Fine-tuning e validação |
| **FUNCEME** | FUNCEME | Estações pontuais | 1990-presente | Validação regional |
| **HRES** | ECMWF | 0.1° × 0.1° | 2016-presente | Benchmark operacional |

### Variáveis Meteorológicas
- **Superfície**: Temperatura (2m), Pressão ao nível do mar, Umidade específica, Vento (u/v 10m), Precipitação
- **Níveis de Pressão**: Geopotencial, Temperatura, Umidade específica, Vento (u/v) em 13 níveis (1000-50 hPa)
- **Forçantes**: Radiação solar, Topografia, Máscara terra-oceano, Temperatura da superfície do mar

## 🗂️ Estrutura do Repositório

```
graphcast-experiments/
├── graphcast/                    # Módulos principais do modelo
│   ├── autoregressive.py        # Predições autoregressivas
│   ├── graphcast.py             # Implementação do GraphCast
│   ├── gencast.py               # Implementação do GenCast (ensemble)
│   ├── data_utils.py            # Utilidades para processamento de dados
│   ├── normalization.py         # Normalização de dados
│   ├── losses.py                # Funções de perda customizadas
│   └── ...                      # Outros módulos de suporte
├── notebooks/                    # Notebooks de experimentação
│   ├── graphcast_demo.ipynb     # Demo básico do GraphCast
│   ├── gencast_mini_demo.ipynb  # Demo do GenCast
│   └── gencast_demo_cloud_vm.ipynb
├── docs/                        # Documentação do projeto
│   └── cloud_vm_setup.md        # Setup de VM na nuvem
├── setup.py                     # Configuração de instalação
└── README.md                    # Este arquivo
```

### Principais Módulos

#### Módulos Core
- [autoregressive.py](graphcast/autoregressive.py): Wrapper para produzir sequências de previsões autoregressivas
- [graphcast.py](graphcast/graphcast.py): Implementação do modelo GraphCast
- [gencast.py](graphcast/gencast.py): Modelo GenCast com previsões ensemble baseadas em difusão
- [data_utils.py](graphcast/data_utils.py): Processamento e preparação de dados meteorológicos

#### Graph Neural Networks
- [deep_typed_graph_net.py](graphcast/deep_typed_graph_net.py): GNN de propósito geral para grafos tipados
- [typed_graph_net.py](graphcast/typed_graph_net.py): Blocos construtivos para GNNs
- [grid_mesh_connectivity.py](graphcast/grid_mesh_connectivity.py): Conversão entre grades regulares e malhas triangulares
- [icosahedral_mesh.py](graphcast/icosahedral_mesh.py): Definição de malha icosaédrica multi-resolução

#### Utilidades
- [normalization.py](graphcast/normalization.py): Normalização baseada em estatísticas históricas
- [losses.py](graphcast/losses.py): Funções de perda com ponderação por latitude
- [xarray_jax.py](graphcast/xarray_jax.py): Interface entre JAX e xarray
- [checkpoint.py](graphcast/checkpoint.py): Serialização e carregamento de modelos

#### Denoisers e Samplers (GenCast)
- [denoiser.py](graphcast/denoiser.py): Implementação do denoiser para difusão
- [dpm_solver_plus_plus_2s.py](graphcast/dpm_solver_plus_plus_2s.py): Solver DPM++ para amostragem eficiente
- [samplers_base.py](graphcast/samplers_base.py): Interface base para samplers

## 🚀 Instalação e Configuração

### Pré-requisitos
```bash
# Python 3.8+
# CUDA 11.0+ (para treinamento em GPU)
# TPU (opcional, para experimentos em larga escala)
```

### Instalação
```bash
# Clone o repositório
git clone [REPOSITÓRIO_CONFIDENCIAL]
cd graphcast-experiments

# Instale as dependências
pip install -e .

# Dependências adicionais
pip install xarray zarr netCDF4 cartopy
```

### Configuração de Dados
⚠️ **ATENÇÃO**: Os dados MERGE e FUNCEME são de acesso restrito. Entre em contato com os orientadores para credenciais.

```python
# Estrutura esperada de diretórios
data/
├── era5/          # Dados ERA5 (Zarr)
├── merge/         # Dados MERGE (NetCDF)
├── funceme/       # Dados FUNCEME (CSV/NetCDF)
└── checkpoints/   # Modelos salvos
```

## 🔬 Experimentos e Notebooks

### 1. GraphCast Demo
[graphcast_demo.ipynb](graphcast_demo.ipynb) - Demonstração básica do GraphCast com:
- Carregamento de dados ERA5
- Execução de previsões determinísticas
- Visualização de resultados
- Métricas de performance

### 2. GenCast Mini Demo
[gencast_mini_demo.ipynb](gencast_mini_demo.ipynb) - Experimentos com GenCast:
- Previsões ensemble probabilísticas
- Quantificação de incerteza
- Comparação com métodos determinísticos

### 3. Adaptação Regional
**🔒 Código em desenvolvimento** - Experimentos de fine-tuning com dados brasileiros

## 📈 Métricas de Avaliação

### Métricas Determinísticas
- **RMSE** (Root Mean Square Error): Erro quadrático médio ponderado por latitude
- **ACC** (Anomaly Correlation Coefficient): Correlação de anomalias
- **Bias**: Viés médio por região e variável

### Métricas Probabilísticas (Ensemble)
- **CRPS** (Continuous Ranked Probability Score)
- **Spread-Skill Relationship**: Relação entre dispersão do ensemble e erro
- **Reliability Diagrams**: Calibração probabilística

### Métricas Regionais Customizadas
- **Skill Score Nordeste**: Performance específica para região Nordeste
- **Precipitation Detection**: POD, FAR, CSI para eventos de precipitação
- **Extreme Events**: Verificação de eventos extremos (>50mm/dia)

## 🔧 Workflow de Desenvolvimento

### 1. Pré-treinamento (Completo)
```bash
# Utiliza checkpoint pré-treinado do DeepMind
# ERA5 global (1979-2018)
```

### 2. Fine-tuning Regional (Em Desenvolvimento)
```python
# Código simplificado do pipeline
from graphcast import graphcast, data_utils, losses
import xarray as xr

# Carregar modelo pré-treinado
model = graphcast.load_pretrained('graphcast_operational')

# Preparar dados brasileiros
merge_data = load_merge_data(year_range=(2019, 2023))
funceme_data = load_funceme_stations()

# Fine-tuning
optimizer = optax.adam(learning_rate=1e-5)
for batch in training_data:
    loss = losses.weighted_mse_loss(
        predictions=model(batch['inputs']),
        targets=batch['targets'],
        weights=get_regional_weights()  # Maior peso para região Nordeste
    )
    # Atualizar parâmetros...
```

### 3. Avaliação
```python
# Comparação com baseline
evaluate_model(
    model=adapted_model,
    test_data=merge_validation_set,
    baseline_models=['ECMWF-IFS', 'GFS', 'CPTEC-BAM'],
    metrics=['rmse', 'acc', 'bias', 'precipitation_skill']
)
```

## 📊 Resultados Preliminares

**🔒 CONFIDENCIAL - Não divulgar**

Os resultados detalhados estão disponíveis apenas para membros da equipe de pesquisa. Contate os orientadores para acesso.

### Principais Descobertas (Resumo)
- ✅ Redução de viés de precipitação em X% para região Nordeste
- ✅ Melhoria em Y% no ACC para previsões de 5 dias
- ✅ Melhor representação de sistemas convectivos organizados
- 🔄 Trabalho em andamento: Quantificação de incerteza calibrada

## 📚 Referências

### Artigos Principais
1. **GraphCast**: Lam, R., et al. (2023). "GraphCast: Learning skillful medium-range global weather forecasting." *Science*, 382(6677), 1416-1421.

2. **GenCast**: Price, I., et al. (2023). "GenCast: Diffusion-based ensemble forecasting for medium-range weather." *arXiv preprint arXiv:2312.15796*.

### Dados
3. **ERA5**: Hersbach, H., et al. (2020). "The ERA5 global reanalysis." *Quarterly Journal of the Royal Meteorological Society*, 146(730), 1999-2049.

4. **MERGE**: Rozante, J.R., et al. (2010). "Combining TRMM and Surface Observations of Precipitation: Technique and Validation over South America." *Weather and Forecasting*, 25(3), 885-894.

### Clima Regional
5. Estudo sobre climatologia da precipitação no Nordeste brasileiro
6. Sistemas meteorológicos atuantes na América do Sul

## 👥 Equipe

**Orientação Acadêmica:**
- Prof. Dr. José Laurindo Campos dos Santos
- Prof. Me. Renato Senna

**Desenvolvimento:**
- Isabela Andrade Aguiar (INPA)

## 📝 Licença e Uso

⚠️ **Este é um projeto de pesquisa acadêmica confidencial.**

- O código é derivado do GraphCast (DeepMind) sob licença Apache 2.0
- Desenvolvimentos e adaptações regionais são propriedade do INPA
- Dados MERGE (INPE) e FUNCEME possuem políticas próprias de uso
- **Não é permitida a distribuição ou publicação sem autorização expressa dos orientadores**

## 🤝 Colaboração e Contato

Para questões sobre o projeto, colaborações ou acesso a resultados:

📧 **Contato**: [emails dos orientadores - confidencial]

---

<div align="center">

**Desenvolvido como parte de pesquisa em Meteorologia Computacional e Deep Learning**

*Última atualização: Dezembro 2025*

</div>
