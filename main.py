import argparse
import pandas as pd
import os
from src.data_processing import load_and_preprocess_data, scale_features
from src.model_training import tune_knn_hyperparameters, train_and_evaluate_model
from src.visualization import (
    plot_feature_distributions,
    plot_scaling_impact,
    plot_accuracy_vs_k,
    plot_confusion_matrix_heatmap
)

def main(data_path):
    """
    Executa o pipeline completo de classificação de vinhos com KNN.
    """
    # Garante que o diretório de visualizações exista
    os.makedirs("visualizations", exist_ok=True)

    # 1. Carregamento e Pré-processamento
    print("--- Etapa 1: Carregando e pré-processando os dados ---")
    X_train, X_test, y_train, y_test, df = load_and_preprocess_data(data_path)
    print("Dados carregados com sucesso.")

    # 2. Escalonamento das Features
    print("\n--- Etapa 2: Escalonando as features ---")
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    print("Features escalonadas com StandardScaler.")

    # 3. Otimização de Hiperparâmetros
    print("\n--- Etapa 3: Otimizando o valor de K via Grid Search ---")
    best_k, grid_search_results = tune_knn_hyperparameters(X_train_scaled, y_train)
    print(f"-> Melhor valor de K encontrado: {best_k}")
    print(f"-> Acurácia de validação cruzada para o melhor K: {grid_search_results.best_score_:.4f}")

    # 4. Treinamento e Avaliação do Modelo Final
    print("\n--- Etapa 4: Treinando e avaliando o modelo final ---")
    model, metrics = train_and_evaluate_model(X_train_scaled, X_test_scaled, y_train, y_test, best_k)

    print("\nMétricas de Performance do Modelo Otimizado:")
    for metric, value in metrics.items():
        if metric != 'confusion_matrix':
            print(f"- {metric.replace('_', ' ').title()}: {value:.4f}")
    print("\nMatriz de Confusão:")
    print(metrics['confusion_matrix'])

    # 5. Geração de Visualizações
    print("\n--- Etapa 5: Gerando e salvando visualizações ---")
    plot_feature_distributions(df, save_path="visualizations/feature_distributions.png")
    plot_scaling_impact(df, X_train_scaled, save_path="visualizations/scaling_impact.png")
    plot_accuracy_vs_k(grid_search_results, save_path="visualizations/accuracy_vs_k.png")
    plot_confusion_matrix_heatmap(metrics['confusion_matrix'], save_path="visualizations/confusion_matrix.png")
    print("\nPipeline concluído com sucesso!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pipeline de Classificação de Vinhos com KNN.")
    parser.add_argument(
        '--data',
        type=str,
        default='data/wine.data',
        help='Caminho para o arquivo de dados (wine.data).'
    )
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"Erro: Arquivo de dados não encontrado em '{args.data}'.")
        print("Certifique-se de que o arquivo 'wine.data' está no diretório 'data/'.")
    else:
        main(args.data)