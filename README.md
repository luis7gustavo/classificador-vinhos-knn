# Classificação de Vinhos com KNN Otimizado

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas">
  <img src="https://img.shields.io/badge/Matplotlib-007ACC?style=for-the-badge&logo=matplotlib&logoColor=white" alt="Matplotlib">
  <img src="https://img.shields.io/badge/Seaborn-094158?style=for-the-badge&logo=seaborn&logoColor=white" alt="Seaborn">
</p>

## 📊 Sobre o Projeto

Este projeto demonstra um pipeline completo de Machine Learning para classificar vinhos em três categorias distintas com base em suas características químicas. A solução utiliza o algoritmo **K-Nearest Neighbors (KNN)** e enfatiza a importância do pré-processamento de dados e da otimização de hiperparâmetros para alcançar alta performance.

O objetivo principal é construir um modelo preditivo robusto, partindo da análise exploratória até a avaliação final, e estruturá-lo como um projeto de software modular e replicável.

## 🔬 Metodologia

O pipeline do projeto foi estruturado nas seguintes etapas principais:

### 1️⃣ **Carregamento e Pré-processamento**
- Carregamento do dataset `wine.data`.
- Divisão dos dados em conjuntos de treino e teste (75% / 25%).
- **Escalonamento de Features**: Aplicação do `StandardScaler` para normalizar as escalas das variáveis, um passo crucial para algoritmos baseados em distância como o KNN.

### 2️⃣ **Otimização de Hiperparâmetros**
- Utilização de **Grid Search com Validação Cruzada (5-fold)** para encontrar o valor ótimo de `k` (número de vizinhos).
- O espaço de busca para `k` foi definido de 1 a 30.

### 3️⃣ **Treinamento e Avaliação**
- Treinamento do modelo KNN final com o melhor valor de `k` encontrado.
- Avaliação completa no conjunto de teste, utilizando métricas como Acurácia, Precisão, Recall, F1-Score e a Matriz de Confusão.

### 4️⃣ **Visualização de Resultados**
- Geração de gráficos para analisar a distribuição das features, o impacto do escalonamento, a relação entre `k` e a acurácia, e um heatmap da matriz de confusão para interpretar os erros de classificação.

## 🚀 Resultados

O modelo otimizado demonstrou uma performance excelente, com uma **acurácia de 95.56%** no conjunto de teste. O valor ótimo de **K=18** foi identificado como o que melhor generaliza para novos dados, conforme a curva de acurácia.

| Métrica               | Valor  |
| --------------------- | ------ |
| Acurácia              | 0.9556 |
| Precisão (Ponderada)  | 0.9587 |
| Recall (Ponderado)    | 0.9556 |
| F1-Score (Ponderado)  | 0.9551 |

<p align="center">
  <img src="https://raw.githubusercontent.com/seu-usuario/classificador_de_vinhos_knn/main/visualizations/confusion_matrix.png" alt="Matriz de Confusão" width="400"/>
  <img src="https://raw.githubusercontent.com/seu-usuario/classificador_de_vinhos_knn/main/visualizations/accuracy_vs_k.png" alt="Acurácia vs. K" width="400"/>
</p>

**Observação:** Para que as imagens acima apareçam, você precisa fazer o upload do repositório para o GitHub e substituir `seu-usuario/classificador_de_vinhos_knn` pelo caminho real do seu repositório.

## 📂 Estrutura do Projeto
