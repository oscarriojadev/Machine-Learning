# automl_comparison.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# AutoML libraries
from auto_sklearn import AutoSklearnClassifier
from tpot import TPOTClassifier
from h2o.automl import H2OAutoML

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

def train_auto_sklearn(X_train, y_train, time_left_for_this_task=120, per_run_time_limit=30):
    """
    Train an Auto-sklearn model.

    Parameters:
    X_train (array): Training features.
    y_train (array): Training target.
    time_left_for_this_task (int): Time limit in seconds for the search.
    per_run_time_limit (int): Time limit in seconds for a single model run.

    Returns:
    AutoSklearnClassifier: Trained Auto-sklearn model.
    """
    automl = AutoSklearnClassifier(time_left_for_this_task=time_left_for_this_task, per_run_time_limit=per_run_time_limit)
    automl.fit(X_train, y_train)
    return automl

def train_tpot(X_train, y_train, generations=5, population_size=20):
    """
    Train a TPOT model.

    Parameters:
    X_train (array): Training features.
    y_train (array): Training target.
    generations (int): Number of generations to run the optimization.
    population_size (int): Number of individuals in the population.

    Returns:
    TPOTClassifier: Trained TPOT model.
    """
    automl = TPOTClassifier(generations=generations, population_size=population_size, verbosity=2)
    automl.fit(X_train, y_train)
    return automl

def train_h2o_automl(X_train, y_train, max_runtime_secs=120):
    """
    Train an H2O AutoML model.

    Parameters:
    X_train (array): Training features.
    y_train (array): Training target.
    max_runtime_secs (int): Maximum runtime in seconds.

    Returns:
    H2OAutoML: Trained H2O AutoML model.
    """
    # Convert data to H2O frame
    import h2o
    h2o.init()
    train = h2o.H2OFrame(pd.concat([pd.DataFrame(X_train), pd.DataFrame(y_train)], axis=1))

    # Define target column
    target_column = y_train.name if hasattr(y_train, 'name') else 'target'
    train[target_column] = train[target_column].asfactor()

    # Train AutoML model
    automl = H2OAutoML(max_runtime_secs=max_runtime_secs)
    automl.train(y=target_column, training_frame=train)

    return automl

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

    automl_models = {
        'Auto-sklearn': train_auto_sklearn,
        'TPOT': train_tpot,
        'H2O AutoML': train_h2o_automl
    }

    results = {}

    for name, train_func in automl_models.items():
        print(f'Training {name}...')
        model = train_func(X_train, y_train)
        accuracy, precision, recall, f1 = evaluate_model(model, X_test, y_test)
        results[name] = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1': f1}
        print(f'{name}: Accuracy = {accuracy:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}, F1 = {f1:.4f}')

        y_pred = model.predict(X_test)
        plot_confusion_matrix(y_test, y_pred, title=f'{name} - Confusion Matrix')

        print(classification_report(y_test, y_pred))

    # Display results
    results_df = pd.DataFrame(results).T
    print('\nAutoML Model Evaluation Results:')
    print(results_df)

if __name__ == '__main__':
    main()
