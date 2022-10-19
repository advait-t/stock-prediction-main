import os
import warnings
import pandas as pd
from datetime import date
from data_fetching import *
from model_building import *
from functions_for_inferencing import *
from inferencing import *
import warnings
warnings.filterwarnings("ignore")

def main():
    holiday_list_path = 'https://raw.githubusercontent.com/advait-t/Stock-Prediction/master/data/raw/2017-2022_Holidays_NSE_BSE_EQ_EQD.csv'
    training_data_path = 'https://raw.githubusercontent.com/advait-t/Stock-Prediction/master/data/raw/training_data.csv'
    error_df_path = 'https://raw.githubusercontent.com/advait-t/Stock-Prediction/master/data/Final/error_df1'
    model_path = 'https://raw.githubusercontent.com/advait-t/Stock-Prediction/master/models/Model_JSON/'
    companies_list_path = 'https://raw.githubusercontent.com/advait-t/Stock-Prediction/master/config/process/companies_config.txt'
    error_metrics_path  = 'https://raw.githubusercontent.com/advait-t/Stock-Prediction/master/models/Model_Metrics/error_metrics.csv'

    # holiday_list_path = '/Users/advait_t/Desktop/Jio/Stock-Prediction/data/raw/2017-2022_Holidays_NSE_BSE_EQ_EQD.csv'
    # training_data_path = '/Users/advait_t/Desktop/Jio/Stock-Prediction/data/raw/training_data.csv'
    # error_df_path = '/Users/advait_t/Desktop/Jio/Stock-Prediction/data/Final/error_df1'
    # model_path = '/Users/advait_t/Desktop/Jio/Stock-Prediction/models/Model_JSON/'
    # companies_list_path = "/Users/advait_t/Desktop/Jio/Stock-Prediction/config/process/companies_config.txt"
    # error_metrics_path = "/Users/advait_t/Desktop/Jio/Stock-Prediction/models/Model_Metrics/error_metrics.csv"

    #! Checking if there is an addition or deletion of companies in the configs file
    new_company, delete_company = check_for_changes_in_companies(training_data_path, companies_list_path)
    print('Checked for changes in list of companies')

    if new_company:
        for new_company in new_company:
            print('Found new company: %s'%new_company)

            #! fetch data for new company
            company_prices, holidays_list = fetch_data_new_company([new_company], training_data_path, holiday_list_path)
            print('Fetched data for new company: %s'%new_company)

            #! train model for new company
            model_building_for_new_company(new_company, company_prices, holidays_list, 1, True, model_path, error_df_path, error_metrics_path)
            print('Trained model for new company: %s'%new_company)

    if delete_company:
        for i in delete_company:
        #! delete data for old company
            data_delete_old_company(i,  training_data_path, error_df_path, model_path)
            print('Deleted data for old company: %s'%i)

    #! inferencing the model
    inferencing(holiday_list_path, training_data_path, error_df_path, model_path)
    print('Inferencing done')

if __name__ == "__main__":
    main()
