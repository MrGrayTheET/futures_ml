import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error

rfr_params = {'max_depth': [5], 'max_features': [4, 5, 6], 'n_estimators': [200]}


class ml_models:

    def __init__(self, data):
        self.data = data.replace(np.inf, np.nan)
        self.data = self.data.dropna()

        self.params = {}
        if type(self.data.index) == pd.DatetimeIndex:
            self.data.index = range(0, self.data.shape[0])  # Cleans NAs and drops DateTimeIndex

        self.x_train, self.y_train, self.x_test, self.y_test, self.train_predict, self.test_predict = \
            [None] * 6  # Vars for future use

    def ml_training_dfs(self, feats, target_col, train_size=0.80):
        train_len = int(self.data.shape[0] * train_size)
        self.feats = feats
        x_data = np.array(self.data[feats])
        y_data = np.array(self.data[target_col])  # Target data
        self.x_train = x_data[:train_len]
        self.y_train = y_data[:train_len]
        self.x_test = x_data[train_len:]
        self.y_test = y_data[train_len:]

        return [self.x_train, self.y_train], [self.x_test, self.y_test]  # returns two lists containing
        # train data, test data

    def linear_model(self, x_col, y_col):  # Linear Regression cuz why not, it might be useful later
        x = np.array(self.data[x_col].reshape(-1, 1))
        y = np.array(self.data[y_col].reshape(-1, 1))
        lin_reg = LinearRegression()
        fitted = lin_reg.fit(x, y)
        score = lin_reg.score(x, y)
        y_pred = lin_reg.predict(x)
        print(f'Coef: {lin_reg.coef_}'
              f'Score: {score}')

        return ([x, y], y_pred)

    def rfr_model(self, parameter_dict, plot_pred=True, plot_importances=True, save_params=True,
                  file_name='rfr_params.csv'):
        test_scores = []
        rfr = RandomForestRegressor(n_estimators=200)
        for g in ParameterGrid(parameter_dict):
            rfr.set_params(**g)
            rfr.fit(self.x_train, self.y_train)
            test_scores.append(rfr.score(self.x_train, self.y_train))
        best_idx = np.argmax(test_scores)
        print(f'best score: {test_scores[best_idx]}\n')
        print(f'best settings {ParameterGrid(parameter_dict)[best_idx]}\n')
        print(f'remaining scores:{test_scores}')
        self.params.update({'rfr': ParameterGrid(parameter_dict)[best_idx]})
        self.train_predict = rfr.predict(self.x_train)
        self.test_predict = rfr.predict(self.x_test)

        if (plot_pred == True):
           if (plot_importances == True):
                fig, ax = plt.subplots(3)
                importance = rfr.feature_importances_
                sorted_idx = np.argsort(importance)[::-1]
                x = range(len(importance))
                labels = np.array(self.feats)[sorted_idx]
                ax[2].bar(x, importance[sorted_idx], tick_label=labels)
                ax[0].scatter(self.y_train, self.train_predict, label='train')
                ax[1].scatter(self.y_test, self.test_predict, label='test')

           else:
                fig,ax=plt.subplots(2)
                ax[0].scatter(self.y_train, self.train_predict, label='train')
                ax[1].scatter(self.y_test, self.test_predict, label='test')
        else:
            if (plot_importances == True):

                importance = rfr.feature_importances_
                sorted_idx = np.argsort(importance)[::-1]
                x = range(len(importance))
                labels = np.array(self.feats)[sorted_idx]
                plt.bar(x,importance,tick_label=labels)



        if save_params == True:
            params = pd.DataFrame(data=self.params['rfr'], index=[0]).to_csv(file_name)



        rfr.set_params(**self.params['rfr'])

        return rfr
