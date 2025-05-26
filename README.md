# 🤖 Projeto de Classificação Supervisionada com Scikit-Learn

Este repositório apresenta um projeto de aprendizado supervisionado focado na **classificação de dados tabulares**. O objetivo principal foi aplicar e comparar diferentes algoritmos de machine learning utilizando a biblioteca `scikit-learn`, com um pipeline completo de preparação de dados e avaliação cruzada.

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
- **Gradient Boosting**
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
- xgboost
- pandas
- numpy
- matplotlib / seaborn

---

Este projeto demonstra um fluxo completo de desenvolvimento de modelos supervisionados, desde o tratamento dos dados até a avaliação estruturada de múltiplas abordagens clássicas e modernas de machine learning.
