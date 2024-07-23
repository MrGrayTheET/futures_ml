from chart_data import file_dict as fd
from feature_builder import model_prep as mp
import ml_model as ml
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
from utils import clean_arrays, sierra_charts as sc
from backtesting import Backtest, Strategy

reader = sc()

es_f = reader.format_sierra(pd.read_csv(fd['ES_F'])).resample('1h').apply(reader.resample_logic)
es_f['vx_norm'] = (reader.format_sierra(pd.read_csv(fd['VX_F'])).resample('1h').apply(reader.resample_logic))['Last']
volume_features = [ 'Delta', 'avg_size','vx_norm'] # Separate features
supply_demand_features = ['D2_norm', 'S2_norm']
sma_features = ['20SMA_norm']
ohlc_feats = ['Open', 'High', 'Low', 'Last']
feats = sma_features + volume_features + supply_demand_features
ohlc_feats = ohlc_feats+feats
es_pre_mod = mp(es_f, ohlc = ohlc_feats)
es_pre_mod.create_targets(2)
es_pre_mod.add_SMA(period=20)
es_pre_mod.bid_ask_vol_features()
es_pre_mod.supply_demand_zone(drop_non_normals=True)
es_pre_mod.make_time_specific()
es_model = ml.ml_model(es_pre_mod.data, feats, '2TARGET')

x_data = es_pre_mod.data[ohlc_feats]
N_TRAIN = len(es_model.x_train)
gbr_model = es_model.tree_model(ml.gbr_params, gbr=True, evaluate=True, plot_pred=True, plot_importances=True)

def get_X(data):
    """Return model design matrix X"""
    return data.filter(like='norm').values


def get_y(data):
    """Return dependent variable y"""
    y = data.Close.pct_change(2).shift(-2)  # Returns after roughly two days
    y[y.between(-.0003, .0003)] = 0             # Devalue returns smaller than 0.4%
    y[y > 0] = 1
    y[y < 0] = -1
    return y


def get_clean_Xy(df):
    """Return (X, y) cleaned of NaN values"""
    X = get_X(df)
    y = get_y(df).values
    isnan = np.isnan(y)
    X = X[~isnan]
    y = y[~isnan]
    return X, y


class MLTrainStrategy(Strategy):
    price_delta = 0.001

    def init(self):
        self.clf = gbr_model
        df = self.data.df.iloc[:N_TRAIN]
        X, y = get_clean_Xy(df)
        self.I(get_y, self.data.df, name='y_true')

        self.forecasts = self.I(lambda: np.repeat(np.nan, len(self.data)))

    def next(self):
        if len(self.data) < N_TRAIN:
            return
        high, low, close = self.data.High, self.data.Low, self.data.Close
        current_time = self.data.index[-1]
        X = get_X(self.data.df.iloc[-1:])
        forecast = self.clf.predict(X)[0]
        self.forecasts[-1] = forecast
        upper, lower = close[-1] * (1 + np.r_[1, -1] * self.price_delta)

        if forecast > 0.0015 and not self.position.is_long:
            self.buy(size=.2, tp=upper, sl=lower)
        elif forecast < -0.0015 and not self.position.is_short:
            self.sell(size=.2, tp=lower, sl=upper)


data = x_data
data['Close']=data['Last'].rename('Close')
data['delta_norm'] = data['Delta'].rename('delta_norm')
data['avg_size_norm'] = data['avg_size'].rename('avg_size_norm')
data = data.dropna()
bt = Backtest( data, MLTrainStrategy, commission=0.0002, margin=.05)
bt.run()
