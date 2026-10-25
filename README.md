# Seminario_Gradient_Boosting_Desempenho
Projeto acadêmico de Inteligência Artificial para previsão do desempenho de partidas utilizando Gradient Boosting (XGBoost), com análise de dados, ajuste de hiperparâmetros e comparação entre modelos de aprendizado de máquina.


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
