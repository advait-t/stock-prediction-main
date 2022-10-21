import csv
import json
import time
import prophet
import warnings
import numpy as np
import pandas as pd
from dateutil.parser import parse
from datetime import datetime, timedelta, date
from prophet.serialize import model_to_json, model_from_json
from sklearn import metrics
from dateutil.parser import parse
from prophet import Prophet
import warnings
warnings.filterwarnings("ignore")


#! Store the model performance metrics in a file to compare the performance of different models

def model_building_for_new_company(company, company_prices, holidays_list, h, eliminate_weekends, model_path, error_df_path, error_metrics_path):

    if holidays_list is not None:

        # variables for the model building and their meaning:
        '''
        holidays: list, list of holidays
        n_changepoints: int, number of changepoints. Change points are abrupt variations in time series data. (n_changepoints = 1 means there is only one changepoint.)
        n_changepoints_scale: float, scale of the number of changepoints 
        changepoint_prior_scale: float, scale of the changepoint prior
        yearly_seasonality: boolean, True if yearly seasonality is to be used, False otherwise
        weekly_seasonality: boolean, True if weekly seasonality is to be used, False otherwise
        daily_seasonality: boolean, True if daily seasonality is to be used, False otherwise
        holidays_prior_scale: float, scale of the holiday prior
        holidays_yearly_prior_scale: float, scale of the yearly holiday prior
        fourier_order: int, order of the fourier series. How quickly the seasonility of the time series can change.
        '''

        m = Prophet(growth="linear",
            holidays= holidays_list,
            seasonality_mode="multiplicative",
            changepoint_prior_scale=30,
            seasonality_prior_scale=35,
            holidays_prior_scale=20,
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=False,
            ).add_seasonality(
                name='monthly',
                period=30.5,
                fourier_order=55
            ).add_seasonality(
                name="daily",
                period=1,
                fourier_order=15
            ).add_seasonality(
                name="weekly",
                period=7,
                fourier_order=20
            ).add_seasonality(
                name="yearly",
                period=365.25,
                fourier_order=20
            ).add_seasonality(
                name="quarterly",
                period = 365.25/4,
                fourier_order=5,
                prior_scale = 15)
    else:
        m = Prophet(growth = 'linear')

    # make last 30 days of the data as the test data and remove them from the training data
    test_data = company_prices[-30:]

    company_prices = company_prices[:-30]

    

    model = m.fit(company_prices)

    future_dates = model.make_future_dataframe(periods = h)

    if eliminate_weekends is not None:
        future_dates['day'] = future_dates['ds'].dt.weekday
        future_dates = future_dates[future_dates['day']<=4]

    
    #! saving the model
    with open(model_path + company + '.json', 'w') as fout:
        json.dump(model_to_json(model), fout)  # Save model

    #! Creating a dataframe for the new company which will log all the values for prediction and track errors
    error_df = pd.DataFrame(columns=['Date', 'Actual_Close', 'Predicted_Close', 'Predicted_Close_Minimum', 'Predicted_Close_Maximum', 'Percent_Change_from_Close', 'Actual_Up_Down', 'Predicted_Up_Down', 'Company'])
    error_df = error_df.append({'Date': '07-04-2022'}, ignore_index=True)
    error_df.to_csv(error_df_path + company + '.csv', index=False)

    # Testing the model on the test set and calculating the error metrics and saving to a csv file
    # The error metrics are calculated for the last 30 days of the test set and comparing between the actual and predicted values
    
    forecast = model.predict(test_data)
    forecast = forecast[['ds', 'yhat']]
    forecast = pd.merge(forecast, test_data, how='left', left_on='ds', right_on='ds')
    forecast = forecast.rename(columns={'ds': 'Date', 'yhat': 'Predicted_Close', 'y': 'Actual_Close'})

    # calculate the RMSE
    rmse = np.sqrt(metrics.mean_squared_error(forecast['Actual_Close'], forecast['Predicted_Close']))
    # calculate the MAPE
    mape = np.mean(np.abs(forecast['Actual_Close'] - forecast['Predicted_Close'])/np.abs(forecast['Actual_Close']))
    # calculate the MAE
    mae = metrics.mean_absolute_error(forecast['Actual_Close'], forecast['Predicted_Close'])
    # calculate the R2
    r2 = metrics.r2_score(forecast['Actual_Close'], forecast['Predicted_Close'])

    error_metrics = pd.DataFrame(columns=['RMSE', 'MAPE', 'MAE', 'R2', 'Company'])
    error_metrics = error_metrics.append({'RMSE': rmse, 'MAPE': mape, 'MAE': mae, 'R2': r2, 'Company': company}, ignore_index=True)
    error_metrics.to_csv(error_metrics_path + 'error_metrics.csv', index=False)

    prediction = model.predict(future_dates)

    return model, prediction, future_dates
