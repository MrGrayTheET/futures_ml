import os
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score


def clean_arrays(data, feats, target_col, sequence=False, periods_in=50, periods_out=20,
                 train_split=True, train_size=0.80, scale_x=True, scale_y=False, x_scale_type='standard',y_scale_type='minmax',
                 minmax_settings=(0, 1),
                 to_tensor=False, return_y_scaler=True):
    # Cleans data and converts to array

    data = data.replace(np.inf, np.nan)
    data = data.dropna()
    x_data = data[feats]
    y_data = data[target_col] # Target data

    if scale_x:
        if x_scale_type == 'minmax':

            scale = MinMaxScaler(minmax_settings)

        else:
            scale = StandardScaler()

        scale.fit(x_data)
        x_data = scale.transform(x_data)

        if scale_y:
            if y_scale_type == 'minmax':
                y_scaler = MinMaxScaler(minmax_settings)
            else: y_scaler = StandardScaler()

            y_scaler.fit(data[[target_col]])
            y_data = y_scaler.fit_transform(data[[target_col]])

    if sequence:
        X, y = [], []  # Sequences for deep learning arrays

        for i in range(len(x_data)):
            end_idx = periods_in + i
            out_end_idx = end_idx + periods_out - 1
            if out_end_idx > len(x_data): break
            seq_x, seq_y = x_data[i:end_idx], y_data[end_idx - 1:out_end_idx]
            X.append(seq_x), y.append(seq_y)

        X_arr = np.array(X)
        y_arr = np.array(y)

    else:
        X_arr, y_arr = x_data, y_data  # Return previous data

    if train_split:
        train_len = round(len(X_arr) * train_size)  # Split arrays for train/test
        x_train = X_arr[:train_len]
        y_train = y_arr[:train_len]
        x_test = X_arr[train_len:]
        y_test = y_arr[train_len:]

        if to_tensor:  # Tensors for deep learning
            x_train = torch.Tensor(x_train)
            x_test = torch.Tensor(x_test)
            y_train = torch.Tensor(y_train)
            y_test = torch.Tensor(y_test)

        if scale_y and return_y_scaler:

            return [x_train, y_train], [x_test, y_test], y_scaler

        else:

            return [x_train, y_train], [x_test, y_test]

    else:

        return X_arr, y_arr


def plot_predictions(train_predict, test_predict, y_train, y_test, y_scaler):
    train_pred, test_pred = [], []
    train_true, test_true = [], []
    train_predictions = y_scaler.inverse_transform(train_predict.data.numpy())
    test_predictions = y_scaler.inverse_transform(test_predict.data.numpy())
    train_y = y_scaler.inverse_transform(y_train.data.numpy())
    test_y = y_scaler.inverse_transform(y_test.data.numpy())

    for i in range(len(train_predictions)):
        train_pred.append(train_predictions[i][0])
        train_true.append(train_y[i][0])

    for i in range(len(test_predictions)):
        test_pred.append(test_predictions[i][0])
        test_true.append(test_y[i][0])

    fig, ax= plt.subplots(2)
    ax[0].plot(train_pred, label='predicted')
    ax[0].plot(train_true, label='Actual')
    ax[1].plot(test_true, label='Actual')
    ax[1].plot(test_pred, label='Predicted')
    plt.show()


def evaluate_model(test_predict, y_test, features, log_file='rfr_log.csv', sorted_features=False, ):

    pred_std_dev = np.std(test_predict)

    evaluation_results = \
        {
            'eval_datetime': [pd.Timestamp.today()],
            'r2_score': [r2_score(y_test, test_predict)],
            'mse': [mean_squared_error(y_test, test_predict)],
            'rmse': [mean_squared_error(y_test, test_predict,squared=False)],
            'rmse/sd': [mean_squared_error(y_test, test_predict, squared=False) / pred_std_dev],
            'mape': [mean_absolute_percentage_error(y_test, test_predict) / 1000000],
            'sorted_features': sorted_features,
            'features': [features]
        }
    if not os.path.isfile(log_file):
        log_df = pd.DataFrame(columns=evaluation_results.keys()).set_index('eval_datetime')

    else:
        log_df = pd.read_csv(log_file, index_col=['eval_datetime'], infer_datetime_format=True)

    log_df.loc[pd.Timestamp.now()] = evaluation_results
    log_df.to_csv(log_file)

    print(evaluation_results)

    return evaluation_results


class sierra_charts:

    def __init__(self):
        self.resample_logic = {'Open'  : 'first',
         'High'  : 'max',
         'Low'   : 'min',
         'Last' : 'last',
         'Volume': 'sum',
         'BidVolume':'sum',
         'AskVolume': 'sum',
         'NumberOfTrades':'sum'}


    def format_sierra(self, df, date_col='Date', time_col='Time'):
        df.columns = df.columns.str.replace(' ', '')
        datetime_series = df[date_col].astype(str)+df[time_col].astype(str)
        datetime_series = pd.to_datetime(datetime_series, format='mixed')
        df = df.drop(columns=[date_col, time_col]).set_index(datetime_series)
        return df
