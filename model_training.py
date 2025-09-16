from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def tune_knn_hyperparameters(X_train_scaled, y_train):
    """
    Utiliza GridSearchCV para encontrar o melhor valor de 'k' (n_neighbors) para o KNN.

    Args:
        X_train_scaled: Features de treino escalonadas.
        y_train: Alvo de treino.

    Returns:
        Tuple: Melhor valor de k e o objeto grid_search ajustado.
    """
    param_grid = {'n_neighbors': list(range(1, 31))}
    knn = KNeighborsClassifier()

    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)

    return grid_search.best_params_['n_neighbors'], grid_search

def train_and_evaluate_model(X_train_scaled, X_test_scaled, y_train, y_test, best_k: int):
    """
    Treina o modelo KNN com o melhor k e o avalia no conjunto de teste.

    Args:
        X_train_scaled: Features de treino escalonadas.
        X_test_scaled: Features de teste escalonadas.
        y_train: Alvo de treino.
        y_test: Alvo de teste.
        best_k (int): Número ótimo de vizinhos.

    Returns:
        Tuple: Modelo treinado e um dicionário com as métricas de avaliação.
    """
    model = KNeighborsClassifier(n_neighbors=best_k)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted'),
        "recall": recall_score(y_test, y_pred, average='weighted'),
        "f1_score": f1_score(y_test, y_pred, average='weighted'),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }

    return model, metrics