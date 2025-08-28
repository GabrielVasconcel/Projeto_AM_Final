# 🤖 Projeto de Classificação Supervisionada com Scikit-Learn

Este repositório apresenta um projeto de aprendizado supervisionado focado na **classificação de dados tabulares**. O objetivo principal foi aplicar e comparar diferentes algoritmos de machine learning utilizando a biblioteca `scikit-learn`, com um pipeline completo de preparação de dados e avaliação cruzada. Adicionalmente, foi desenvolvido uma interface para interação com os modelos e explicação da metodologia aplicada, passando por assuntos desde a concepção do entendimento da regra de negócio, avaliação e preparação dos dados, aplicação de modelos de aprendizado de máquina e busca por hiperparâmetros, até avaliação dos resultados obtidos.

## 🚀 Como Rodar o Projeto

Com o repositório clonado na sua máquina:

### 1. Criar e Ativar o Ambiente Virtual

Para manter as dependências do projeto isoladas, vamos criar um **ambiente virtual**.

```bash
# Cria um ambiente virtual na pasta 'venv'
python -m venv venv
```

Depois de criado, você precisa ativá-lo. O comando varia dependendo do seu sistema operacional:

- **No Windows:**

  ```bash
  .\venv\Scripts\activate
  ```

- **No macOS ou Linux:**
  `bash
  source venv/bin/activate
  `
  Ao ativar, você verá `(venv)` no início da linha do seu terminal, indicando que o ambiente está ativo.

### 2. Instalar as Dependências

Com o ambiente virtual ativo, instale todas as bibliotecas necessárias, que estão listadas no arquivo `requirements.txt`.

```bash
pip install -r requirements.txt
```

Este comando irá instalar o `scikit-learn`, `gradio`, `pandas` e todas as outras dependências utilizadas no projeto.

### 3. Executar a Aplicação com Gradio

Finalmente, inicie a interface interativa do Gradio. O servidor de desenvolvimento local será iniciado, permitindo que você interaja com o modelo diretamente no seu navegador.

```bash
gradio app.py
```

Após executar o comando, o terminal exibirá um endereço local, geralmente algo como:

`Running on local URL: http://127.0.0.1:7860`

Abra essa URL no seu navegador para ver a aplicação funcionando e interagir com os modelos de classificação.


## 📚 Estrutura do projeto
O projeto consiste em 2 arquivos principais, main.ipynb e app.py.

### main.ipynb
Neste arquivo encontra-se todo o processo de ciência de dados realizado. Importação dos dados, manipulações, análise exploratória, construção dos modelos e otimização, aplicação e avaliação dos resultados.

### app.py
Arquivo que utiliza a biblioteca gradio para montagem da interface de interação com os modelos e visualização geral do processo de desenvolvimento do projeto.

## 📚 Etapas Desenvolvidas

### 🔎 1. Análise e Pré-processamento de Dados

- Carregamento do dataset e análise das características dos atributos.
- Normalização dos dados utilizando `RobustScaler`.
- Separação dos conjuntos de treino e teste com `train_test_split`, tomando precauções para evitar vazamento de dados.

### 🧪 2. Treinamento e Avaliação de Modelos

Foram aplicados diversos algoritmos de classificação, incluindo:

- **K-Nearest Neighbors (KNN)**
- **Support Vector Machine (SVM)**
- **Random Forest**
- **Decision Tree**
- **Comitês de Redes Neurais**
- **Comitês Heterogêneos**
- **XGBoost**
- **LightGBM**
- **Multi-Layer Perceptron (MLPClassifier)**

### 🧰 3. Comparação e Organização

- Criação de uma estrutura automatizada para treinar e avaliar múltiplos modelos em sequência.
- Organização dos resultados em tabelas para facilitar a comparação e a análise final.

## 🛠️ Tecnologias Utilizadas

- Python
- scikit-learn
- pandas
- numpy
- matplotlib / seaborn

---

Este projeto demonstra um fluxo completo de desenvolvimento de modelos supervisionados, desde o tratamento dos dados até a avaliação estruturada de múltiplas abordagens clássicas e modernas de machine learning.
