import os
import time
from dateutil.parser import parse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

#! Function to check if there is any new company in the list or an old company has been removed from the list
def check_for_changes_in_companies(training_data_path, companies_list_path):
    existing_company_list = pd.read_csv(training_data_path)["Company"].unique()
    with open(companies_list_path, "r") as f:
        new_companies_list=[i for line in f for i in line.split(',')]

    new_company = list(set(new_companies_list) - set(existing_company_list))
    delete_company = list(set(existing_company_list) - set(new_companies_list))
    return new_company, delete_company

#! Function to fetch data for new company from yahoo finance
def YahooFinanceHistory(company, previous_days, training_data_path):
    '''
    
    This function takes the company name and the number of previous days as input and returns the dataframe of the company history.

    Variables:

    company: string, name of the company
    previous_days: int, number of days to extract data from
    today: date, today's date
    past: date, date of the past
    query_string: string, query string to extract data from yahoo finance
    company_prices: dataframe, dataframe containing the prices of the company
    company_data: dataframe, dataframe containing the data of the company
    valuation_measures: list, list containing the valuation measures interested in
    company_valuation: dataframe, dataframe containing the valuation measures of the company
    path_save_as_csv: boolean, True if the dataframe is to be saved as a csv file, False otherwise
    
    '''
    
    # today = int(time.mktime((datetime.now()).timetuple()))
    # past = int(time.mktime((datetime.now() - timedelta(previous_days)).timetuple()))
    
    # interval = '1d'

    # # defining the query to get historical stock data
    # query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{company}?period1={past}&period2={today}&interval={interval}&events=history&includeAdjustedClose=true'
    
    # company_prices = pd.read_csv(query_string)


    today = date.today()
    past = today - timedelta(previous_days)
    company_prices = yf.download(company, start = past, end = today)
    company_prices = company_prices.reset_index()
    # company_prices = company_prices[['Date', 'Close']]
    # company_prices.columns = ['Date', 'Close']
    company_prices['Date'] = pd.to_datetime(company_prices['Date'])
    company_prices = company_prices.sort_values(by = 'Date')
    company_prices = company_prices.reset_index(drop = True)

    company_prices['Company'] = company
    training_data = pd.read_csv(training_data_path)

    training_data = training_data.append(company_prices)

    training_data1 = training_data[training_data['Company'] == company]
    training_data = training_data[training_data['Company'] != company]

    if training_data1['Date'].tail(1).values[0] != company_prices['Date'].tail(1).values[0]: 
        training_data1 = training_data1.append(company_prices.tail(1))
    else:
        pass

    training_data = training_data.append(training_data1)
    data = training_data[training_data['Company'] == company]
    data1 = training_data[training_data['Company'] != company]
    data.drop_duplicates(subset = 'Date', inplace = True, keep = 'last')
    data.reset_index(inplace = True, drop = True)
    training_data = data1.append(data)
    training_data.to_csv(training_data_path, index = False)

    return company_prices


#! Function to read data from csv file
def read_data(company, previous_days, training_data_path, holidays_list_path = 0):

    company_prices = YahooFinanceHistory(company, previous_days, training_data_path)
    company_prices = company_prices[:-1]
    company_prices = company_prices[['Date', 'Close']]
    company_prices.columns = ['ds', 'y']
    company_prices['ds'] = pd.to_datetime(company_prices['ds'])

    holidays_list = pd.read_csv(holidays_list_path)

    for i in range(len(holidays_list['Day'])):
        holidays_list['Day'][i] = pd.to_datetime(parse(holidays_list['Day'][i]))

    holidays_list = holidays_list[['Holiday','Day']]
    holidays_list = holidays_list.rename({'Day':'ds', 'Holiday':'holiday'}, axis = 1)   

    return company_prices, holidays_list

def fetch_data_new_company(new_company, training_data_path, holidays_list_path):
    new_company = ','.join(new_company)
    new_company_prices = read_data(new_company, 365*5, training_data_path,holidays_list_path) # read data for 5 years
    
    return new_company_prices

def data_delete_old_company(old_company, training_data_path, error_df_path, model_path):
    # old_company = ','.join(old_company)
    training_data = pd.read_csv(training_data_path)
    training_data = training_data[training_data['Company'] != old_company]
    training_data.to_csv(training_data_path, index=False)
    os.remove(error_df_path + old_company + '.csv')
    os.remove(model_path + old_company + '.json')