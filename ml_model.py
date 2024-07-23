import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import clean_arrays, evaluate_model
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score, root_mean_squared_error

rfr_params = {'max_depth': [3,5], 'max_features': [5,6,7], 'n_estimators': [200]}

gbr_params = {'max_features': [5,6,7],
              'learning_rate':[0.01],
              'n_estimators':[200],
              'subsample':[0.6],
              'random_state':[42]}

class ml_model:

    def __init__(self, data, features,target_column,train_test_size=0.8,scale_x=False, scale_y=False, log_results=False, log_file_path='ml_log'):
        self.feats = features
        self.model = None
        self.train_predict, self.test_predict = \
            [None] * 2  # Vars for future use

        self.params = {}

        if type(data.index) == pd.DatetimeIndex:
            data.index = range(0, data.shape[0])  # Cleans NAs and drops DateTimeIndex

        [self.x_train, self.y_train],  [self.x_test, self.y_test] =\
            clean_arrays(data, features, target_column, scale_x=scale_x, scale_y=scale_y, train_split=True, train_size=train_test_size)


    def plot_predictions(self, training_pred=False, test_pred=True):
        if training_pred and test_pred:
            fig, ax = plt.subplots(2)
            ax[0].scatter(self.y_train, self.train_predict, label='train')
            ax[1].scatter(self.y_test, self.test_predict, label='test')

        elif training_pred and not test_pred:
            plt.scatter(self.y_train, self.train_predict)

        elif test_pred and not training_pred: plt.scatter(self.y_test, self.test_predict)

        plt.show()

        return None

    def tree_model(self, parameter_dict, gbr=False, plot_pred=True, plot_importances=True, save_params=False,
                   params_file='rfr_params.csv', evaluate=True, eval_log='model_eval.csv'):
        test_scores = []
        rmse_scores = []

        if not gbr:
            model = RandomForestRegressor(n_estimators=200)
            name = 'rfr'
        else:
            model = GradientBoostingRegressor(n_estimators=200)
            name = 'gbr'

        for g in ParameterGrid(parameter_dict):
            model.set_params(**g)
            model.fit(self.x_train, self.y_train)
            test_scores.append(model.score(self.x_train, self.y_train))
            rmse_scores.append(root_mean_squared_error(self.y_test, model.predict(self.x_test)))

        best_idx = np.argmax(test_scores)
        best_rmse = np.argmin(rmse_scores)
        print(f'best score: {test_scores[best_idx]}\n')
        print(f'best settings {ParameterGrid(parameter_dict)[best_idx]}\n')
        print(f'remaining scores:{test_scores}')
        print(f'Best RMSE: {rmse_scores[best_rmse]}')
        print(f'best rmse settings: {ParameterGrid(parameter_dict)[best_rmse]}')
        self.params.update({name: ParameterGrid(parameter_dict)[best_idx]})

        model.set_params(**self.params[name])
        self.model = model
        self.train_predict = model.predict(self.x_train)
        self.test_predict = model.predict(self.x_test)
        importance = model.feature_importances_
        sorted_idx = np.argsort(importance)[::-1]
        x = range(len(importance))
        labels = np.array(self.feats)[sorted_idx]
        self.feats = [labels]

        if plot_pred:
            if plot_importances:
                fig, ax = plt.subplots(3)
                ax[2].bar(x, importance[sorted_idx], tick_label=labels)
                ax[0].scatter(self.y_train, self.train_predict, label='train')
                ax[1].scatter(self.y_test, self.test_predict, label='test')

            else:
                self.plot_predictions(training_pred=True, test_pred=True)
        else:
            if plot_importances:
                importance = model.feature_importances_
                sorted_idx = np.argsort(importance)[::-1]
                x = range(len(importance))
                labels = np.array(self.feats)[sorted_idx]
                plt.bar(x, importance, tick_label=labels)

        if save_params:
            params = pd.DataFrame(data=self.params[name], index=[pd.Timestamp.today()]).to_csv(params_file)

        if evaluate:
            evaluate_model(self.test_predict, self.y_test, self.feats,sorted_features=True, log_file=eval_log)

        return self.model

    def neighbors_model(self, n_start, n_end, plot_pred=True,
                        evaluate=True, eval_log='model_eval.csv'):
        results_df = pd.DataFrame(columns=['train_r2', 'test_r2', 'n_neighbors'], index=range(n_start, n_end))
        for n in range(n_start, n_end):
            knn = KNeighborsRegressor(n_neighbors=n)
            knn.fit(self.x_train, self.y_train)
            results_df['n_neighbors'].loc[n] = n
            results_df['train_r2'].loc[n] = knn.score(self.x_train, self.y_train)
            results_df['test_r2'].loc[n] = knn.score(self.x_test, self.y_test)

        sorted_models = np.argsort(results_df['test_r2'])[::-1]
        best_neighbors = results_df['n_neighbors'].iloc[sorted_models].iloc[0]

        knn.set_params(**{'n_neighbors': best_neighbors})

        self.model = knn
        self.train_predict = self.model.predict(self.x_train)
        self.test_predict = self.model.predict(self.x_test)

        if evaluate:

            evaluate_model(self.test_predict, self.y_test, self.feats, eval_log)

        if plot_pred:
            self.plot_predictions(test_pred=True, training_pred=True)

        return self.model


