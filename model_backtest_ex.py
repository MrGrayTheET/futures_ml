from chart_data import file_dict as fd, cot_data as cot, sentiment
from feature_builder import model_prep as mp
import ml_model as ml
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
from utils import clean_arrays, sierra_charts as sc
from backtesting import Backtest, Strategy


reader = sc()

es_f = reader.format_sierra(pd.read_csv(fd['ES_F'])).resample('4h').apply(reader.resample_logic)

ohlc = ['Open', 'High', 'Low', 'Last']

es_pre_mod = mp(es_f, ohlc = ohlc)
es_pre_mod.create_targets(period=2)

sentiment_names = ['Bullish_norm', 'Bearish_norm']
pos_names = ['long_pos_norm', 'short_pos_norm', 'long_chg_norm', 'short_chg_norm', 'pct_long_norm', 'pct_short_norm']

es_pre_mod.add_new_series(sentiment, ["Bullish", 'Bearish'], sentiment_names)

cot = cot()
futs_noncom_positioning = cot.contract_data('ES_F')[cot.non_commercials]
es_pre_mod.add_new_series(futs_noncom_positioning, cot.non_commercials[2:],pos_names)


es_pre_mod.bid_ask_vol_features()
es_pre_mod.add_SMA(20)
es_pre_mod.add_daily_SMA(20, normalize=True)
es_pre_mod.supply_demand_zone()

volume_features = [ 'Delta', 'avg_size', 'BidVolume', 'AskVolume'] # Separate features
supply_demand_features = ['D1_norm', 'S1_norm','D2_norm', 'S2_norm']
sma_features = ['20SMA_norm', '20d_SMA_norm']
technical_feats = sma_features + volume_features + supply_demand_features

all_feats = pos_names + sentiment_names + technical_feats

es_ml_model = ml.ml_model(es_pre_mod.data, all_feats, target_column='3TARGET')

x_data = es_pre_mod.data[all_feats]

gbr_model = es_ml_model.tree_model(parameter_dict=ml.gbr_params, gbr=True, eval_log='positioning_model')
N_TRAIN = len(es_ml_model.x_train)

def get_X(data):
    """Return model design matrix X"""
    return data.filter(like='norm').values


def get_y(data):
    """Return dependent variable y"""
    y = data.Close.pct_change(2).shift(-2)  # Returns after roughly two days
    y[y.between(-.003, .003)] = 0             # Devalue returns smaller than 0.4%
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
    price_delta = 0.005

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

        if forecast > 0.005 and not self.position.is_long:
            self.buy(size=.2, tp=upper+0.002*upper, sl=lower)
        elif forecast < -0.005 and not self.position.is_short:
            self.sell(size=.2, tp=lower-0.002*lower, sl=upper)

data = es_pre_mod.data[all_feats + ohlc]

data['Close']=data['Last'].rename('Close')
data['delta_norm'] = data['Delta'].rename('delta_norm')
data['avg_size_norm'] = data['avg_size'].rename('avg_size_norm')
data['BidVolume_norm'] = data['BidVolume'].rename('BidVolume_norm')
data['AskVolume_norm'] = data['AskVolume'].rename('AskVolume_norm')

data = data.dropna()
bt = Backtest( data, MLTrainStrategy, commission=0.0002, margin=.05)
bt.run()
