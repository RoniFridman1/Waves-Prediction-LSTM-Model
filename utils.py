import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import scipy
from ShallowRegressionLSTM import ShallowRegressionLSTM
import psycopg2
import os, sys
from dotenv import load_dotenv
load_dotenv()

def load_data(data, features_mask='all', forecast_lead=4, target_variable='hs', new_data=False):
    if isinstance(data, str):
        data = pd.read_csv(data, index_col='datetime')
    if new_data:
        data.index = pd.to_datetime(data.index,utc=True)
        datetime_column = data.reset_index()['datetime'].apply(lambda x: int(x.strftime("%H")) * 2 * np.pi / 24)
    else:
        datetime_column = data.reset_index()['datetime'].apply(lambda x:int(x.strftime("%H")) * 2 * np.pi / 24 )
    data['hour_x'] = [np.cos(i) for i in datetime_column]
    data['hour_y'] = [np.sin(i) for i in datetime_column]
    # Transform degrees using sin/cos
    data['direction_x'] = data['direction'].apply(lambda x: np.cos(x * 2 * np.pi / 360))
    data['direction_y'] = data['direction'].apply(lambda x: np.sin(x * 2 * np.pi / 360))
    data['wind_direction_x'] = data['wind_direction'].apply(lambda x: np.cos(x* 2 * np.pi / 360))
    data['wind_direction_y'] = data['wind_direction'].apply(lambda x: np.sin(x * 2 * np.pi/ 360))
    data.drop(columns=['direction', "wind_direction"], inplace=True)
    num_of_rows = len(data)
    def print_tp(data,rows=100):
        plt.plot(data['tp'][-rows:])
        plt.show()

    # Rain is sparse - convert to binary column
    data['rain'] = data['rain'].apply(lambda x: 1.0 if x > 0 else 0.0)
    # print_tp(data)
    if target_variable == 'tp':
        data['tp'] = data['tp'].rolling(8, 1, center=True, closed='both').median()
    if target_variable == 'hs':
        data['hs'] = data['hs'].rolling(5, 1, center=True, closed='both').mean()

    # print_tp(data)
    # Do we want all features or filtering some of them. Default is "all".
    if features_mask != 'all':
        features_mask = list(features_mask) + [target_variable]
        data = data.loc[:, features_mask]
    # Notice that the target variable is also a feature now.
    features = list(data.columns.difference([target_variable]))
    new_target_col_name = f"{target_variable}_lead{forecast_lead}"

    if not new_data:
        data[new_target_col_name] = data[target_variable].shift(-forecast_lead)
        data = data.iloc[:-forecast_lead]
    else:
        data[new_target_col_name] = data[target_variable].shift(-forecast_lead)

    features_excluded_from_cleaning = ["rain",'direction_y',"direction_x"]
    # print(f"Rows in data: {num_of_rows}")
    for column in features + [new_target_col_name]:
        # Cleaning outliers except for the target variable.
        if target_variable in column or column in features_excluded_from_cleaning:
            data[column].interpolate(inplace=True)
            data[column].fillna(method='bfill', inplace=True)
            continue
        iqr = scipy.stats.iqr(data[column], nan_policy='omit')

        q1, q3 = np.nanquantile(data[column], 0.25), np.nanquantile(data[column], 0.75)
        if column in ['hmax', 'h_onethird']:
            lb, ub = q1 - 3 * iqr, q3 + 3 * iqr
        else:
            lb, ub = q1 - 1.5 * iqr, q3 + 1.5 * iqr

        percent_to_remove = len(data.loc[(data[column] < lb) | (data[column] > ub), column]) / num_of_rows

        data.loc[(data[column] < lb) | (data[column] > ub), column] = np.nan
        data[column].interpolate(inplace=True)
        data[column].fillna(method='bfill', inplace=True)
        # print(f"column: {column}, lower bound: {lb}, upper bound: {ub}")
        # print(f"percent of cells to remove:{round(percent_to_remove * 100, 2)}%")
    return data, features, new_target_col_name


def train_test_split(data, ratio=0.7, test_start_ts=None):
    splitting_edge = int(len(data) * ratio) if test_start_ts is None else test_start_ts

    if test_start_ts is not None:
        df_train = data.loc[:test_start_ts].copy()
        df_test = data.loc[test_start_ts:].copy()
    else:
        df_train = data.iloc[:splitting_edge].copy()
        df_test = data.iloc[splitting_edge:].copy()
    return df_train, df_test


def configure_new_model(features, learning_rate, num_hidden_units, num_layers, dropout,weight_decay,target):
    model = ShallowRegressionLSTM(num_sensors=len(features), hidden_units=num_hidden_units, num_layers=num_layers,
                                  dropout=dropout)
    loss_function = nn.L1Loss() if target == 'tp' else nn.MSELoss() # L1Loss or MSELoss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=weight_decay)
    return model, loss_function, optimizer


def create_short_data_csv(full_csv_path,wind_data_path,swell_data_path,ims_data_path, new_data_training_path, empirical_test_data_path,
                          forecast, seq_number, predict_latest=False, years_of_data_to_use=3):
    wind_data = pd.read_csv(wind_data_path, index_col='datetime')
    wind_data.index = pd.to_datetime(wind_data.index,utc=True)
    wind_data = wind_data.resample('30min').mean()
    swell_data = pd.read_csv(swell_data_path, index_col='datetime')
    swell_data.index = pd.to_datetime(swell_data.index,utc=True)
    swell_data = swell_data.resample('30min').mean()
    ims_meteo_data = pd.read_csv(ims_data_path, index_col='datetime')
    ims_meteo_data.index = pd.to_datetime(ims_meteo_data.index, utc=True)
    ims_meteo_data = ims_meteo_data.resample('30min').mean()

    if predict_latest:
        data = pd.read_csv(full_csv_path).set_index('datetime').iloc[-1 * forecast * seq_number:]
        data.index = pd.to_datetime(data.index)
        data = pd.DataFrame.join(data, wind_data,on='datetime', how='left').join(other=swell_data,on='datetime', how='left').join(other=ims_meteo_data,on='datetime', how='left')
        return data
    cutoff_index= round(-17520 * years_of_data_to_use)
    data = pd.read_csv(full_csv_path,index_col='datetime').iloc[cutoff_index:]
    data.index = pd.to_datetime(data.index)
    data = pd.DataFrame.join(data, wind_data, on='datetime', how='left').join(other=swell_data,on='datetime', how='left').join(other=ims_meteo_data,on='datetime', how='left')
    data[-1 * forecast * seq_number:-1 * forecast].to_csv(new_data_training_path)
    data[-1 * forecast:].to_csv(empirical_test_data_path)
    data = data.iloc[:-1 * forecast]
    return data


def upload_predictions_to_db(predictions_df: pd.DataFrame, location='haifa',target_variable='hs'):
    """
    saves the prediction made into a dedicated table
    order in table should be: datetime, "predicted_<target variable name>"
    """
    try:

        connection = psycopg2.connect(user=os.getenv('db_user'),
                                      password=os.getenv('db_password'),
                                      host=os.getenv('db_host'),
                                      port=os.getenv('db_port'),
                                      database=os.getenv('db_name'))
        table_name = f"waves_data.cameri_{target_variable}_predictions_{location}"
        target_variable_table_name = f"predicted_{target_variable}"
        cursor = connection.cursor()
        general_command = "INSERT INTO " + table_name + " VALUES (%s, %s) on conflict (datetime) do update set "+\
                f"{target_variable_table_name} = excluded.{target_variable_table_name}"
        for l in predictions_df:
            command = cursor.mogrify(general_command, (l[0],l[1])).decode("utf-8")
            cursor.execute(command)
        connection.commit()
    except (Exception, psycopg2.Error) as error:
        print(" saving to db is not successful : " + str(error))

    finally:
        if connection:
            cursor.close()
            connection.close()


def update_latest_data_from_db(full_data_path, location='haifa'):
    """
    saves the prediction made into a dedicated table
    order in table should be: datetime, location, target variable predicted value
    """
    try:
        location_id = {'haifa':1, "ashdod":2}
        connection = psycopg2.connect(user=os.getenv('db_user'),
                                      password=os.getenv('db_password'),
                                      host=os.getenv('db_host'),
                                      port=os.getenv('db_port'),
                                      database=os.getenv('db_name'))
        cursor = connection.cursor()

        waves_data_table_name = "backend_buoysmsr"
        command = f"COPY (select datetime,hs,direction,tz,tp,temperature,h_onethird,hmax,tav from {waves_data_table_name} where location_id = {location_id[location]} order by datetime) " \
                  f"to STDOUT with CSV delimiter ',' header"
        command = cursor.mogrify(command)
        connection.commit()
        with open(full_data_path,"w+") as file:
            cursor.copy_expert(command, file)
            file.close()

        wind_data_table_name = "wind_data.wind_10min"
        command = f"COPY (select datetime_utc as datetime ,wind_speed_10m, wind_direction from {wind_data_table_name} where location = {location_id[location]} order by datetime) " \
                  f"to STDOUT with CSV delimiter ',' header"
        command = cursor.mogrify(command)
        connection.commit()
        wind_csv_output_path = full_data_path[:-4]+"_wind.csv"
        with open(wind_csv_output_path, "w+") as file:
            cursor.copy_expert(command, file)
            file.close()

        swell_data_table_name = "waves_data.swell"
        command = f"COPY (select datetime, hs_swell from {swell_data_table_name} where location = '{location}' order by datetime desc) " \
                  f"to STDOUT with CSV delimiter ',' header"
        command = cursor.mogrify(command)
        connection.commit()
        swell_csv_output_path = full_data_path[:-4] + "_swell.csv"
        with open(swell_csv_output_path, "w+") as file:
            cursor.copy_expert(command, file)
            file.close()

        ims_data_table_name = "meteo_data.haifa_technion"
        command = f"COPY (select utc_datetime as datetime,td, rh, tdmax, tdmin, ws1mm, ws10mm,rain from {ims_data_table_name} order by datetime desc) " \
                  f"to STDOUT with CSV delimiter ',' header"
        command = cursor.mogrify(command)
        connection.commit()
        ims_csv_output_path = full_data_path[:-4] + "_ims.csv"
        with open(ims_csv_output_path, "w+") as file:
            cursor.copy_expert(command, file)
            file.close()

    except (Exception, psycopg2.Error) as error:
        print(" exporting to CSV was not successful : " + str(error))

    finally:
        if connection:
            cursor.close()
            connection.close()