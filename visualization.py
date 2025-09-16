import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Define um estilo visual mais moderno e profissional
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')
sns.set_context("talk")

def plot_feature_distributions(df: pd.DataFrame, save_path: str):
    """Cria e salva boxplots da distribuição de features por classe."""
    features_to_plot = ['alcohol', 'color_intensity', 'flavanoids']
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    fig.suptitle('Distribuição de Features por Classe de Vinho', fontsize=20, fontweight='bold')

    for i, feature in enumerate(features_to_plot):
        sns.boxplot(x='class', y=feature, data=df, ax=axes[i], width=0.5)
        axes[i].set_title(f'{feature.replace("_", " ").title()}', fontsize=16)
        axes[i].set_xlabel('Classe do Vinho', fontsize=14)
        axes[i].set_ylabel(feature.replace("_", " ").title(), fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"-> Gráfico de distribuição salvo em: {save_path}")

def plot_scaling_impact(df: pd.DataFrame, X_train_scaled, save_path: str):
    """Cria e salva histogramas comparando distribuições antes e depois do escalonamento."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Impacto do Escalonamento nas Distribuições', fontsize=20, fontweight='bold')

    # Antes
    sns.histplot(df['alcohol'], kde=True, ax=axes[0, 0], color='skyblue', bins=20)
    axes[0, 0].set_title('Álcool (Antes do Escalonamento)', fontsize=16)

    sns.histplot(df['color_intensity'], kde=True, ax=axes[0, 1], color='salmon', bins=20)
    axes[0, 1].set_title('Intensidade da Cor (Antes)', fontsize=16)

    # Depois
    sns.histplot(X_train_scaled[:, 0], kde=True, ax=axes[1, 0], color='skyblue', bins=20)
    axes[1, 0].set_title('Álcool (Depois do Escalonamento)', fontsize=16)

    sns.histplot(X_train_scaled[:, 10], kde=True, ax=axes[1, 1], color='salmon', bins=20)
    axes[1, 1].set_title('Intensidade da Cor (Depois)', fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"-> Gráfico de impacto do escalonamento salvo em: {save_path}")

def plot_accuracy_vs_k(grid_search_results, save_path: str):
    """Cria e salva o gráfico da acurácia média vs. valor de K."""
    k_values = grid_search_results.cv_results_['param_n_neighbors'].data
    mean_scores = grid_search_results.cv_results_['mean_test_score']
    best_k = grid_search_results.best_params_['n_neighbors']
    best_score = grid_search_results.best_score_

    plt.figure(figsize=(14, 8))
    plt.plot(k_values, mean_scores, marker='o', linestyle='--', color='dodgerblue', label='Acurácia Média CV', markersize=8)
    plt.axvline(best_k, color='red', linestyle=':', linewidth=2, label=f'Melhor K = {best_k} (Acurácia: {best_score:.4f})')

    plt.title('Performance do KNN vs. Número de Vizinhos (K)', fontsize=20, fontweight='bold')
    plt.xlabel('Número de Vizinhos (K)', fontsize=14)
    plt.ylabel('Acurácia Média (Validação Cruzada)', fontsize=14)
    plt.xticks(list(range(1, 31)))
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"-> Gráfico de acurácia vs. K salvo em: {save_path}")

def plot_confusion_matrix_heatmap(cm, save_path: str):
    """Cria e salva um heatmap da matriz de confusão."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Classe 1', 'Classe 2', 'Classe 3'],
        yticklabels=['Classe 1', 'Classe 2', 'Classe 3'],
        annot_kws={"size": 16}
    )
    plt.title('Matriz de Confusão do Modelo Otimizado', fontsize=20, fontweight='bold')
    plt.xlabel('Classe Prevista', fontsize=14)
    plt.ylabel('Classe Verdadeira', fontsize=14)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"-> Gráfico da matriz de confusão salvo em: {save_path}")