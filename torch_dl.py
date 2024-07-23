import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import clean_arrays
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class lstm(nn.Module):

    def __init__(self, num_classes, input_size,hidden_size, num_layers):
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc_1 = nn.Linear(hidden_size, 128)
        self.fc_2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        hid_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        cel_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        output, (hn, cn) = self.lstm(x, (hid_0, cel_0))
        hn = hn.view(-1, self.hidden_size)
        out = self.relu(hn)
        out = self.fc_1(out)
        out = self.relu(out)
        return self.fc_2(out)


def training_loop(n_epochs, lstm, optimizer, loss_func, X_train, y_train, X_test, y_test):

    for epoch in range(n_epochs):
        lstm.train()
        outputs = lstm.forward(X_train)
        optimizer.zero_grad()
        loss = loss_func(outputs, y_train)
        loss.backward()
        optimizer.step()

        # Test Loss
        lstm.eval()
        test_preds = lstm(X_test)
        test_loss = loss_func(test_preds, y_test)

        if epoch % 100 == 0:
            print(f'Epoch: {epoch}, train_loss: {loss.item()} test_loss: {test_loss.item()}')

        if epoch ==  n_epochs:

            return lstm(X_train), lstm(X_test)



def plot_final_prediction(test_x, test_y, y_scaler, lstm):
    '''
        test_predict : LSTM prediction result
         test_true: torch tensor containing true values
         '''
    forecast = lstm(test_x[-1].unsqueeze(0))
    forecast = y_scaler.inverse_transform(forecast.detach().numpy())
    forecast = forecast[0].tolist()
    final_true = test_y[-1].detach().numpy()
    final_true = y_scaler.inverse_transform(final_true.reshape(1,-1))
    final_true = final_true[0].tolist()
    plt.plot(final_true, label='Actual')
    plt.plot(forecast, label='Predicted')

