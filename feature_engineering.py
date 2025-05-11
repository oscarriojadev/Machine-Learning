# feature_engineering.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """
    Load data from a CSV file.

    Parameters:
    file_path (str): Path to the CSV file.

    Returns:
    pd.DataFrame: Loaded data.
    """
    data = pd.read_csv(file_path)
    return data

def handle_missing_values(data, strategy='mean'):
    """
    Handle missing values in the dataset.

    Parameters:
    data (pd.DataFrame): Input data.
    strategy (str): Strategy to handle missing values ('mean', 'median', 'mode', 'drop').

    Returns:
    pd.DataFrame: Data with missing values handled.
    """
    if strategy == 'mean':
        data.fillna(data.mean(), inplace=True)
    elif strategy == 'median':
        data.fillna(data.median(), inplace=True)
    elif strategy == 'mode':
        data.fillna(data.mode().iloc[0], inplace=True)
    elif strategy == 'drop':
        data.dropna(inplace=True)
    return data

def encode_categorical_variables(data, columns=None):
    """
    Encode categorical variables using Label Encoding.

    Parameters:
    data (pd.DataFrame): Input data.
    columns (list): List of columns to encode. If None, all categorical columns are encoded.

    Returns:
    pd.DataFrame: Data with categorical variables encoded.
    """
    if columns is None:
        columns = data.select_dtypes(include=['object']).columns
    label_encoder = LabelEncoder()
    for column in columns:
        data[column] = label_encoder.fit_transform(data[column])
    return data

def create_new_features(data):
    """
    Create new features from existing ones.

    Parameters:
    data (pd.DataFrame): Input data.

    Returns:
    pd.DataFrame: Data with new features.
    """
    # Example: Create a new feature by combining existing features
    if 'feature1' in data.columns and 'feature2' in data.columns:
        data['new_feature'] = data['feature1'] * data['feature2']
    return data

def scale_features(data, scaler='standard'):
    """
    Scale features using StandardScaler or MinMaxScaler.

    Parameters:
    data (pd.DataFrame): Input data.
    scaler (str): Scaler to use ('standard', 'minmax').

    Returns:
    pd.DataFrame: Data with scaled features.
    """
    if scaler == 'standard':
        scaler = StandardScaler()
    elif scaler == 'minmax':
        scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    return pd.DataFrame(scaled_data, columns=data.columns)

def select_features(data, target_column, k=10):
    """
    Select the best features using SelectKBest.

    Parameters:
    data (pd.DataFrame): Input data.
    target_column (str): Name of the target column.
    k (int): Number of features to select.

    Returns:
    pd.DataFrame: Data with selected features.
    """
    X = data.drop(columns=[target_column])
    y = data[target_column]
    selector = SelectKBest(score_func=f_classif, k=k)
    X_new = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    return data[selected_features]

def reduce_dimensions(data, n_components=2):
    """
    Reduce dimensions using PCA.

    Parameters:
    data (pd.DataFrame): Input data.
    n_components (int): Number of components to keep.

    Returns:
    pd.DataFrame: Data with reduced dimensions.
    """
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)
    return pd.DataFrame(reduced_data, columns=[f'PC{i}' for i in range(1, n_components + 1)])

def plot_feature_importance(model, feature_names):
    """
    Plot feature importance from a trained model.

    Parameters:
    model: Trained model with feature_importances_ attribute.
    feature_names (list): List of feature names.
    """
    feature_importance = model.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), np.array(feature_names)[sorted_idx])
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance')
    plt.show()

def main():
    # Example usage
    file_path = 'path_to_your_data.csv'
    target_column = 'target'

    data = load_data(file_path)
    data = handle_missing_values(data, strategy='mean')
    data = encode_categorical_variables(data)
    data = create_new_features(data)
    data = scale_features(data, scaler='standard')
    data = select_features(data, target_column, k=10)
    data = reduce_dimensions(data, n_components=2)

    # Example: Train a model and plot feature importance
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    plot_feature_importance(model, X.columns)

if __name__ == '__main__':
    main()
