# classification_models.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

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
    Train a classification model.

    Parameters:
    model: Classification model to train.
    X_train (array): Training features.
    y_train (array): Training target.

    Returns:
    model: Trained classification model.
    """
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a classification model.

    Parameters:
    model: Trained classification model.
    X_test (array): Testing features.
    y_test (array): Testing target.

    Returns:
    tuple: Accuracy, precision, recall, and F1 scores.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    return accuracy, precision, recall, f1

def plot_confusion_matrix(y_test, y_pred, title='Confusion Matrix'):
    """
    Plot the confusion matrix.

    Parameters:
    y_test (array): Actual values.
    y_pred (array): Predicted values.
    title (str): Title of the plot.
    """
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def main():
    # Example usage
    file_path = 'path_to_your_data.csv'
    target_column = 'target'

    data = load_data(file_path)
    X, y = preprocess_data(data, target_column)
    X_train, X_test, y_train, y_test = split_data(X, y)

    models = {
        'Logistic Regression': LogisticRegression(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'Support Vector Machine': SVC(),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB()
    }

    results = {}

    for name, model in models.items():
        trained_model = train_model(model, X_train, y_train)
        accuracy, precision, recall, f1 = evaluate_model(trained_model, X_test, y_test)
        results[name] = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1': f1}
        print(f'{name}: Accuracy = {accuracy:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}, F1 = {f1:.4f}')

        y_pred = trained_model.predict(X_test)
        plot_confusion_matrix(y_test, y_pred, title=f'{name} - Confusion Matrix')

        print(classification_report(y_test, y_pred))

    # Display results
    results_df = pd.DataFrame(results).T
    print('\nModel Evaluation Results:')
    print(results_df)

if __name__ == '__main__':
    main()
