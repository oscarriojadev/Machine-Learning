# regression_models.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

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

def preprocess_data(data, target_column):
    """
    Preprocess the data by splitting into features and target, and then scaling the features.

    Parameters:
    data (pd.DataFrame): Input data.
    target_column (str): Name of the target column.

    Returns:
    tuple: Features (X) and target (y) arrays.
    """
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets.

    Parameters:
    X (array): Features.
    y (array): Target.
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int): Seed for the random number generator.

    Returns:
    tuple: Training and testing sets (X_train, X_test, y_train, y_test).
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def train_model(model, X_train, y_train):
    """
    Train a regression model.

    Parameters:
    model: Regression model to train.
    X_train (array): Training features.
    y_train (array): Training target.

    Returns:
    model: Trained regression model.
    """
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a regression model.

    Parameters:
    model: Trained regression model.
    X_test (array): Testing features.
    y_test (array): Testing target.

    Returns:
    tuple: Mean Squared Error (MSE) and R-squared (R2) scores.
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2

def plot_results(y_test, y_pred, title='Actual vs Predicted'):
    """
    Plot actual vs predicted values.

    Parameters:
    y_test (array): Actual values.
    y_pred (array): Predicted values.
    title (str): Title of the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(title)
    plt.show()

def main():
    # Example usage
    file_path = 'path_to_your_data.csv'
    target_column = 'target'

    data = load_data(file_path)
    X, y = preprocess_data(data, target_column)
    X_train, X_test, y_train, y_test = split_data(X, y)

    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso(),
        'Decision Tree': DecisionTreeRegressor(),
        'Random Forest': RandomForestRegressor(),
        'Gradient Boosting': GradientBoostingRegressor(),
        'Support Vector Regression': SVR()
    }

    results = {}

    for name, model in models.items():
        trained_model = train_model(model, X_train, y_train)
        mse, r2 = evaluate_model(trained_model, X_test, y_test)
        results[name] = {'MSE': mse, 'R2': r2}
        print(f'{name}: MSE = {mse:.4f}, R2 = {r2:.4f}')

        y_pred = trained_model.predict(X_test)
        plot_results(y_test, y_pred, title=f'{name} - Actual vs Predicted')

    # Display results
    results_df = pd.DataFrame(results).T
    print('\nModel Evaluation Results:')
    print(results_df)

if __name__ == '__main__':
    main()
