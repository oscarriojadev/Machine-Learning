# forecasting_arima_prophet.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from prophet import Prophet
from math import sqrt

def load_data(file_path):
    """
    Load time series data from a CSV file.

    Parameters:
    file_path (str): Path to the CSV file.

    Returns:
    pd.DataFrame: Loaded time series data.
    """
    data = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
    return data

def plot_time_series(data, title='Time Series Data'):
    """
    Plot the time series data.

    Parameters:
    data (pd.DataFrame): Time series data.
    title (str): Title of the plot.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(data, label='Time Series')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

def test_stationarity(data):
    """
    Test the stationarity of the time series using the Augmented Dickey-Fuller test.

    Parameters:
    data (pd.DataFrame): Time series data.

    Returns:
    tuple: ADF test results.
    """
    result = adfuller(data)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:', result[4])
    return result

def plot_acf_pacf(data, lags=None):
    """
    Plot the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF).

    Parameters:
    data (pd.DataFrame): Time series data.
    lags (int): Number of lags to include in the plot.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(data, lags=lags, ax=ax1)
    plot_pacf(data, lags=lags, ax=ax2)
    plt.show()

def fit_arima_model(data, order=(1, 1, 1)):
    """
    Fit an ARIMA model to the time series data.

    Parameters:
    data (pd.DataFrame): Time series data.
    order (tuple): ARIMA order (p, d, q).

    Returns:
    statsmodels.tsa.arima.model.ARIMAResults: Fitted ARIMA model.
    """
    model = ARIMA(data, order=order)
    fitted_model = model.fit()
    print(fitted_model.summary())
    return fitted_model

def forecast_arima(fitted_model, steps=10):
    """
    Forecast future values using the fitted ARIMA model.

    Parameters:
    fitted_model (statsmodels.tsa.arima.model.ARIMAResults): Fitted ARIMA model.
    steps (int): Number of steps to forecast.

    Returns:
    pd.DataFrame: Forecasted values.
    """
    forecast = fitted_model.forecast(steps=steps)
    return forecast

def evaluate_arima_model(fitted_model, data):
    """
    Evaluate the ARIMA model using Mean Squared Error (MSE) and Root Mean Squared Error (RMSE).

    Parameters:
    fitted_model (statsmodels.tsa.arima.model.ARIMAResults): Fitted ARIMA model.
    data (pd.DataFrame): Time series data.

    Returns:
    tuple: MSE and RMSE values.
    """
    predictions = fitted_model.predict()
    mse = mean_squared_error(data, predictions)
    rmse = sqrt(mse)
    print('MSE:', mse)
    print('RMSE:', rmse)
    return mse, rmse

def fit_prophet_model(data, column_name='value'):
    """
    Fit a Prophet model to the time series data.

    Parameters:
    data (pd.DataFrame): Time series data.
    column_name (str): Name of the column containing the values to forecast.

    Returns:
    prophet.forecaster.Prophet: Fitted Prophet model.
    """
    # Prepare data for Prophet
    prophet_data = data.reset_index()
    prophet_data.columns = ['ds', 'y']

    # Initialize and fit the Prophet model
    model = Prophet()
    model.fit(prophet_data)
    return model

def forecast_prophet(model, periods=365):
    """
    Forecast future values using the fitted Prophet model.

    Parameters:
    model (prophet.forecaster.Prophet): Fitted Prophet model.
    periods (int): Number of periods to forecast.

    Returns:
    pd.DataFrame: Forecasted values.
    """
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast

def evaluate_prophet_model(model, data, column_name='value'):
    """
    Evaluate the Prophet model using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

    Parameters:
    model (prophet.forecaster.Prophet): Fitted Prophet model.
    data (pd.DataFrame): Time series data.
    column_name (str): Name of the column containing the values to forecast.

    Returns:
    tuple: MAE and RMSE values.
    """
    # Prepare data for Prophet
    prophet_data = data.reset_index()
    prophet_data.columns = ['ds', 'y']

    # Make predictions
    future = model.make_future_dataframe(periods=len(prophet_data))
    forecast = model.predict(future)

    # Calculate errors
    predictions = forecast['yhat'].values[:len(prophet_data)]
    mae = mean_absolute_error(prophet_data['y'], predictions)
    rmse = sqrt(mean_squared_error(prophet_data['y'], predictions))
    print('MAE:', mae)
    print('RMSE:', rmse)
    return mae, rmse

def plot_forecast(data, forecast, title='Forecast'):
    """
    Plot the forecasted values.

    Parameters:
    data (pd.DataFrame): Time series data.
    forecast (pd.DataFrame): Forecasted values.
    title (str): Title of the plot.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(data, label='Actual')
    plt.plot(forecast, label='Forecast', color='red')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

def main():
    # Example usage
    file_path = 'path_to_your_data.csv'
    data = load_data(file_path)

    plot_time_series(data)

    # ARIMA model
    print('ARIMA Model:')
    test_stationarity(data)
    plot_acf_pacf(data, lags=20)
    arima_model = fit_arima_model(data, order=(1, 1, 1))
    arima_forecast = forecast_arima(arima_model, steps=10)
    print('ARIMA Forecast:', arima_forecast)
    arima_mse, arima_rmse = evaluate_arima_model(arima_model, data)
    plot_forecast(data, arima_forecast, title='ARIMA Forecast')

    # Prophet model
    print('Prophet Model:')
    prophet_model = fit_prophet_model(data)
    prophet_forecast = forecast_prophet(prophet_model, periods=365)
    print('Prophet Forecast:', prophet_forecast[['ds', 'yhat']].tail())
    prophet_mae, prophet_rmse = evaluate_prophet_model(prophet_model, data)
    plot_forecast(data, prophet_forecast['yhat'].tail(len(data)), title='Prophet Forecast')

if __name__ == '__main__':
    main()
