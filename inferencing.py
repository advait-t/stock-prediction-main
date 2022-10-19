import os
import warnings
import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta, date
from dateutil.parser import parse
import json
warnings.filterwarnings("ignore")
from prophet.serialize import model_to_json, model_from_json


def inferencing(holiday_list_path, training_data_path, error_df_path, model_path):

    today = date.today()

    company_list = pd.read_csv(training_data_path)["Company"].unique()

    for company in company_list:
        
        error_df = pd.read_csv(error_df_path + company + '.csv')

        #! Checking if there were any missed days in between
        if error_df.iloc[-1]['Date'] >= str(today):
            error_df = pred_vs_real_comparision(real_stock_price(company, next_day_prediction(model_path + company + '.json', False)), next_day_prediction(model_path + company + '.json', False), error_df, company)
        else:
            print ("Missed days")
            error_df = filling_missing_dates(error_df, company, holiday_list_path, model_path)
            error_df = pred_vs_real_comparision(real_stock_price(company, next_day_prediction(model_path + company + '.json', False)), next_day_prediction(model_path + company + '.json', False), error_df, company)

        #! Check for null values in actual close and get its date
        error_df = update_actual_close(error_df, company)

        if is_holiday(today, holiday_list_path) == True or today.weekday() == 5 or today.weekday() == 6:
            error_df = error_df[error_df['Date'] != str(today)]

        # convert the dates to one format
        # error_df['Date'] = pd.to_datetime(error_df['Date'])
        # error_df['Date'] = error_df['Date'].dt.strftime('%Y-%m-%d')
        #! saving the df to a csv file
        error_df.to_csv(error_df_path + company + '.csv', index=False)


#! Loading Model
def load_model(model_path):
    with open(model_path, 'r') as fin:
        saved_model = model_from_json(json.load(fin))  # Load model
    return saved_model

#! check for holiday
def is_holiday(today, holiday_list_path):
    holidays_list = pd.read_csv(holiday_list_path)
    for i in range(len(holidays_list['Day'])):
        holidays_list['Day'][i] = pd.to_datetime(parse(holidays_list['Day'][i]))
    for i in range(len(holidays_list['Day'])):
        if holidays_list['Day'][i].date() == today:
            return True
    return False

#! get real stock price
def real_stock_price(company, predicted):

    now = datetime.now()
    weekday_weekend = datetime.strptime(str(predicted['ds'][0]), '%Y-%m-%d %H:%M:%S')
    
    if weekday_weekend.weekday() <= 5 and weekday_weekend.weekday() != 0:
        days = 1
    elif weekday_weekend.weekday() == 6:
        days = 2
    elif weekday_weekend.weekday() == 0:
        days = 3

    past = datetime.strptime(str(predicted['ds'][0]), '%Y-%m-%d %H:%M:%S') - timedelta(days)
    past = past.replace(hour = now.hour, minute = now.minute, second = now.second, microsecond = now.second)
    past = int(time.mktime(past.timetuple()))
    
    interval = '1d'

    # defining the query to get historical stock data
    query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{company}?period1={past}&period2={past}&interval={interval}&events=history&includeAdjustedClose=true'
    
    try:
        company_stock_price = pd.read_csv(query_string)
        company_stock_price = company_stock_price[['Date', 'Close']]
        return company_stock_price
    except:
        company_stock_price = pd.DataFrame(np.nan, index = [0], columns=['Date'])
        return company_stock_price

#! for next day prediction
def next_day_prediction(model_path, missing_dates, missing_dates_df = 0):

    saved_model = load_model(model_path)

    if missing_dates == False:
        next_day = date.today() + timedelta(days=1)
        future_date = pd.DataFrame(pd.date_range(start = next_day, end = next_day, freq ='D'), columns = ['ds'])
        predicted = saved_model.predict(future_date)
        return (predicted[['ds','yhat', 'yhat_upper', 'yhat_lower']])
    else:
        missing_dates_df.rename(columns={'Date':'ds'}, inplace=True)
        predicted = saved_model.predict(missing_dates_df)
        return (predicted[['ds','yhat', 'yhat_upper', 'yhat_lower']])

#! Fetch stock price when there is a null value in actual price column
def fetch_stock_price(company, date):
    now = datetime.now()

    past = datetime.strptime(str(date), '%Y-%m-%d %H:%M:%S')
    past = past.replace(hour = now.hour, minute = now.minute, second = now.second, microsecond = now.second)
    past = int(time.mktime(past.timetuple()))
    
    interval = '1d'

    # defining the query to get historical stock data
    query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{company}?period1={past}&period2={past}&interval={interval}&events=history&includeAdjustedClose=true'
    
    try:
        company_stock_price = pd.read_csv(query_string)
        company_stock_price = company_stock_price[['Date', 'Close']]
        return company_stock_price['Close'][0]
    except:
        return None

def update_actual_close(df, company):
    #! Check for null values in actual close and get its date
    null_values = df[df['Actual_Close'].isnull()]
    print(null_values)
    # add %H:%M:%S to get time as well
    null_values['Date'] = pd.to_datetime(null_values['Date'], format = '%Y-%m-%d %H:%M:%S')
    # convert to string
    null_values['Date'] = null_values['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    null_values = null_values['Date'].to_list()

    for date in null_values:
        stock_price = fetch_stock_price(company, date)
        # remove time from date
        date = date.split(' ')[0]
        # append to dataframe
        df.loc[df['Date'] == date, 'Actual_Close'] = stock_price
        # calculate percent change from close for the date in null_values
        df.loc[df['Date'] == date, 'Percent_Change_from_Close'] = (df.loc[df['Date'] == date, 'Predicted_Close'] - df.loc[df['Date'] == date, 'Actual_Close'])/df.loc[df['Date'] == date, 'Actual_Close']

        df['Actual_Up_Down'] = np.where(df['Actual_Close'].isna(), np.nan, np.where(df['Actual_Close'] > df['Actual_Close'].shift(1), 'Up', 'Down'))
        df['Predicted_Up_Down'] = np.where(df['Predicted_Close'].isna(), np.nan, np.where(df['Predicted_Close'] > df['Predicted_Close'].shift(1), 'Up', 'Down'))

        # df['Actual_Up_Down'] = np.where((df['Actual_Close'] > df['Actual_Close'].shift(-1)), 'Up', 'Down')
        # df['Predicted_Up_Down'] = np.where((df['Predicted_Close'] > df['Actual_Close'].shift(-1)), 'Up', 'Down')

        # convert the dates to one format
        df['Date'] = pd.to_datetime(df['Date'])
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

    return df


def real_stock_price_missing_date(company, predicted):
    now = datetime.now()
    predicted['Close'] = None
    for i in range(len(predicted['ds'])):
        past = datetime.strptime(str(predicted['ds'][i]), '%Y-%m-%d %H:%M:%S')
        past = past.replace(hour = now.hour, minute = now.minute, second = now.second, microsecond = now.second)
        print(past)
        past = int(time.mktime(past.timetuple()))
        interval = '1d'
        
        query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{company}?period1={past}&period2={past}&interval={interval}&events=history&includeAdjustedClose=true'
        company_stock_price = pd.read_csv(query_string)
        company_stock_price = company_stock_price[['Date', 'Close']]
        predicted['Close'][i] = company_stock_price['Close'].values[0]
    return predicted

#! Filling missing dates
def filling_missing_dates(error_df, company, holiday_list_path, model_path):
    Date = date.today()
    
    date_range = pd.date_range(start = error_df.iloc[-1]['Date'], end = Date, freq ='B')

    date_range_df = pd.DataFrame(columns = error_df.columns)
    date_range_df['Date'] = date_range
    date_range_df['Date'] = date_range_df['Date'].dt.date

    for i in range(len(date_range_df['Date'])):
        if is_holiday(date_range_df['Date'][i], holiday_list_path) == True:
            date_range_df = date_range_df[date_range_df['Date'] != date_range_df['Date'][i]]
            
    missing_dates_df = next_day_prediction(model_path + company + '.json',True, date_range_df)
    missing_dates_df = real_stock_price_missing_date(company, missing_dates_df)

    # convert ds from datetime to date
    missing_dates_df['ds'] = missing_dates_df['ds'].dt.date

    missing_dates_df.rename(columns = {'ds':'Date', 'Close':'Actual_Close', 'yhat':'Predicted_Close', 'yhat_upper':'Predicted_Close_Maximum', 'yhat_lower':'Predicted_Close_Minimum'}, inplace = True)
    missing_dates_df['Percent_Change_from_Close'] = ((missing_dates_df['Actual_Close'] - missing_dates_df['Predicted_Close'])/missing_dates_df['Actual_Close'])*100

    missing_dates_df['Actual_Up_Down'] = np.where((missing_dates_df['Actual_Close'] > missing_dates_df['Actual_Close'].shift(-1)), 'Up', 'Down')
    missing_dates_df['Predicted_Up_Down'] = np.where((missing_dates_df['Predicted_Close'] > missing_dates_df['Actual_Close'].shift(-1)), 'Up', 'Down')

    error_df = error_df.append(missing_dates_df, ignore_index= True)
    error_df = error_df.drop_duplicates(subset = 'Date', keep = 'last')
    error_df['Company'] = company

    return error_df

def pred_vs_real_comparision(real_stock_price, predicted, error_df, company):

    df = pd.DataFrame([[np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]], columns=error_df.columns)
    error_df = pd.concat([error_df, df], ignore_index =True)

    error_df['Date'].iloc[-1] = str(predicted['ds'].iloc[-1].strftime('%Y-%m-%d'))
    error_df['Date'] = pd.to_datetime(error_df['Date'])
    error_df = error_df.set_index('Date')

    error_df['Predicted_Close'].loc[predicted['ds']] = predicted['yhat'].iloc[-1]
    error_df['Predicted_Close_Minimum'].loc[predicted['ds']] = predicted['yhat_lower'].iloc[-1]
    error_df['Predicted_Close_Maximum'].loc[predicted['ds']] = predicted['yhat_upper'].iloc[-1]
    
    # add company name to the dataframe
    error_df['Company'] = company

    error_df.insert(0, 'Date', error_df.index)

    if pd.isna(real_stock_price['Date'])[0] == False:
        if predicted['ds'].iloc[-1].weekday() == 0:
            days = 3 #default days = 1
        elif predicted['ds'].iloc[-1].weekday() == 6:
            days = 2
        else:
            days = 1
            
        error_df['Actual_Close'].loc[predicted['ds']-timedelta(days)] = real_stock_price['Close'].iloc[-1]
        percent_change = ((error_df['Actual_Close'].loc[predicted['ds']-timedelta(days)] - error_df['Predicted_Close'].loc[predicted['ds']-timedelta(days)])/error_df['Actual_Close'].loc[predicted['ds']-timedelta(days)]*100)
        error_df['Percent_Change_from_Close'].loc[predicted['ds']-timedelta(days)] = percent_change

        up_or_down_original = error_df['Actual_Close'].loc[predicted['ds']][0]-error_df['Actual_Close'].loc[predicted['ds']-timedelta(days)][0]

        if up_or_down_original > 0:
            error_df['Actual_Up_Down'].loc[predicted['ds']] = 'Up'

        elif up_or_down_original == 0:
            error_df['Actual_Up_Down'].loc[predicted['ds']] = 'Same'

        else:
            error_df['Actual_Up_Down'].loc[predicted['ds']] = 'Down'

        up_or_down_predicted = error_df['Predicted_Close'].loc[predicted['ds']][0]-error_df['Predicted_Close'].loc[predicted['ds']-timedelta(days)][0]

        if up_or_down_predicted > 0:
            error_df['Predicted_Up_Down'].loc[predicted['ds']] = 'Up'

        elif up_or_down_predicted == 0:
            error_df['Predicted_Up_Down'].loc[predicted['ds']] = 'Same'

        else:
            error_df['Predicted_Up_Down'].loc[predicted['ds']] = 'Down'
        

        error_df = error_df[~error_df.index.duplicated(keep='first')]

    else:
        pass

    return error_df