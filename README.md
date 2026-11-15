# Seminario_Gradient_Boosting_Desempenho
Projeto acadêmico de Inteligência Artificial para previsão do desempenho de partidas utilizando Gradient Boosting (XGBoost), com análise de dados, ajuste de hiperparâmetros e comparação entre modelos de aprendizado de máquina.

O sistema classifica os resultados pela perspectiva do mandante em três categorias: Vitória, Empate e Derrota.

## Problema

Investigar se características históricas das equipes podem contribuir para a previsão dos resultados das partidas, utilizando dados de temporadas anteriores.

## Integrantes

Rafaela Oliveira Viol

Henrique Meneses Catanho da Silva

Luis Henrique Balduino Neto Comini

Felipe Yuji Sato Silva


## Dataset
São utilizados dados das temporadas de 2023, 2024 e 2025 do Campeonato Brasileiro.

Treinamento: temporadas de 2023 e 2024.

Teste: temporada de 2025.

Total: 1.140 partidas.

Os dados incluem datas, equipes participantes, gols e resultados. A partir dessas informações, são calculadas características históricas, como rating Elo, média de pontos e médias de gols marcados e sofridos.

Fonte dos dados: https://github.com/adaoduque/Brasileirao_Dataset

## Tecnologias e Dependências

Python

Pandas

NumPy

Matplotlib

Scikit-learn

XGBoost

Joblib

## Como executar o Projeto

### Pré-requisitos

Python instalado no computador.

As dependências necessárias para executar a aplicação.

### Dependências

O projeto utiliza Python e as seguintes bibliotecas:

- **Flask:** criação e execução da API.
- **Joblib:** carregamento do modelo treinado.
- **NumPy:** operações numéricas.
- **Pandas:** manipulação dos datasets.
- **Scikit-learn:** ferramentas de aprendizado de máquina.
- **XGBoost:** algoritmo utilizado para a previsão dos resultados.
- **Matplotlib:** geração de gráficos, caso necessário no treinamento.
- **OpenPyXL:** leitura e escrita de arquivos Excel, caso necessário.

### Instalação das dependências

Com o Python instalado, abra o terminal na pasta principal do projeto e execute:

```bash
pip install flask joblib numpy pandas scikit-learn xgboost matplotlib openpyxl
```

### Execução da aplicação

Após instalar as dependências, execute:

```bash
python app.py
```

Em seguida, abra o navegador e acesse:

http://127.0.0.1:5000

**Observação:** o arquivo `modelo_xgboost.joblib` deve estar presente no caminho esperado pela API. Não é necessário executar o treinamento novamente para utilizar o modelo já treinado.


## Resultados Iniciais

Na avaliação realizada com as partidas de 2025, o XGBoost apresentou acurácia de 52,11% e F1-macro de 0,3894.

Esses valores representam os resultados da execução atual e não garantem o mesmo desempenho em outras temporadas.

## Objetivo Acadêmico
O projeto busca aplicar e avaliar técnicas de aprendizado de máquina, comparando o XGBoost com modelos de referência e analisando suas limitações na previsão de resultados esportivos.

## Funcionalidades da Aplicação

A aplicação foi desenvolvida utilizando Flask para disponibilizar o modelo treinado por meio de uma interface web.

O sistema conta com as seguintes funcionalidades:

- **Arena de Simulação:** permite selecionar as equipes mandante e visitante para realizar uma previsão do resultado da partida.
- **Previsão de Resultados:** apresenta as probabilidades de Vitória, Empate e Derrota pela perspectiva do mandante.
- **Análise das Equipes:** apresenta informações históricas utilizadas pelo modelo, como rating Elo, média de pontos e médias de gols marcados e sofridos.
- **Visualização Gráfica:** disponibiliza gráficos para facilitar a interpretação das informações e dos resultados da previsão.
- **Laboratório de Modelos:** permite comparar diferentes modelos de aprendizado de máquina e realizar experimentos com ajustes de hiperparâmetros.

## Modelos de Aprendizado de Máquina

Foram utilizados modelos de referência para avaliar o desempenho do XGBoost:

- **Dummy Classifier:** modelo baseline utilizado como referência inicial.
- **Regressão Logística:** modelo utilizado para comparação com o algoritmo principal.
- **XGBoost:** modelo baseado em Gradient Boosting, responsável pela previsão das três classes de resultados.

A comparação considera métricas como acurácia e F1-macro, permitindo analisar não apenas a quantidade de previsões corretas, mas também o desempenho do modelo em cada classe.

## Laboratório de Experimentos

A aplicação possui uma seção dedicada à realização de experimentos com os modelos.

Nessa seção, é possível:

- Comparar o desempenho da Regressão Logística e do XGBoost.
- Visualizar métricas de avaliação, como acurácia e F1-macro.
- Ajustar hiperparâmetros do XGBoost.
- Executar experimentos com diferentes configurações.
- Analisar o tempo de execução e a matriz de confusão.

Os experimentos permitem observar como as alterações nos hiperparâmetros influenciam o desempenho do modelo.

## Estrutura da Aplicação

O projeto utiliza uma API desenvolvida com Flask para conectar o modelo de aprendizado de máquina à interface web.

Entre as rotas disponibilizadas estão:

- `/api/teams`: disponibiliza as equipes utilizadas na simulação.
- `/api/predict`: realiza a previsão do resultado de uma partida.
- `/api/model-comparison`: disponibiliza os resultados da comparação entre modelos.
- `/api/experiment`: executa experimentos com configurações de hiperparâmetros.

A interface permite interagir com essas funcionalidades sem precisar executar diretamente os códigos de treinamento pelo navegador.
