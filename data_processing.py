import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path: str):
    """
    Carrega o dataset de vinhos, define os nomes das colunas, separa as features
    do alvo e divide os dados em conjuntos de treino e teste.

    Args:
        file_path (str): Caminho para o arquivo .data.

    Returns:
        Tuple: X_train, X_test, y_train, y_test, df (dataframe completo).
    """
    column_names = [
        'class', 'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash',
        'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
        'proanthocyanins', 'color_intensity', 'hue',
        'od280_od315_of_diluted_wines', 'proline'
    ]
    df = pd.read_csv(file_path, header=None, names=column_names)

    y = df['class']
    X = df.drop('class', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    return X_train, X_test, y_train, y_test, df

def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Aplica o escalonamento StandardScaler nas features de treino e teste.

    Args:
        X_train (pd.DataFrame): Features de treino.
        X_test (pd.DataFrame): Features de teste.

    Returns:
        Tuple: Features de treino e teste escalonadas (numpy arrays).
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled