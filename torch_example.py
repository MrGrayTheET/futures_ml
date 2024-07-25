import torch_dl as dl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils import clean_arrays, plot_predictions, sierra_charts as sc
from chart_data import file_dict as fd
import datetime as dt


reader = sc()

es_f = reader.format_sierra(pd.read_csv(fd['ES_F'])).resample('1h', offset=dt.timedelta(hours=-9)).apply(reader.resample_logic)


ohlc = ['Open', 'High', 'Low', 'Last']
target = 'Last'

[train_x, train_y], [test_x, test_y], y_scaler = clean_arrays(data=es_f, feats=ohlc, target_col=target, sequence=True, periods_in=50, periods_out=20,
                                                    train_split=True, scale_x=True, x_scale_type='standard', scale_y=True,y_scale_type='minmax', to_tensor=True, return_y_scaler=True)

train_x = torch.reshape(train_x, (train_x.shape[0], 50, train_x.shape[2]))
test_x = torch.reshape(test_x, (test_x.shape[0], 50, test_x.shape[2]))
n_epochs = 200
learning_rate = 0.001
input_size = train_x.shape[2]
hidden_size = 2
num_layers = 1
num_classes = 20
train_y = torch.reshape(train_y, (train_y.shape[0], train_y.shape[1]))
test_y = torch.reshape(test_y, (test_y.shape[0], test_y.shape[1]))
lstm = dl.lstm(num_classes, 4, hidden_size,num_layers)
loss = nn.MSELoss()
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

training = dl.training_loop(n_epochs=400, lstm=lstm,loss_func=loss,optimizer=optimizer, X_train=train_x, y_train=train_y, X_test=test_x, y_test=test_y)


training_preds = lstm(train_x)
testing_preds = lstm(test_x)

plot_predictions(training_preds, testing_preds, train_y, test_y, y_scaler)