import pandas as pd
import numpy as np
import datetime as dt
from formatting import format_sierra as fs
import seaborn as sns
import feature_builder as fb
from chart_data import file_dict
import ml_model as ml
import matplotlib.pyplot as plt

bar_data_PATH = "C:\\Users\\nicho\\charts\\"

logic = {'Open'  : 'first',
         'High'  : 'max',
         'Low'   : 'min',
         'Last' : 'last',
         'Volume': 'sum',
         'BidVolume':'sum',
         'AskVolume': 'sum',
         'NumberOfTrades':'sum'}

rfr_params = {'max_depth': [5, 6],
              'max_features': [3, 4, 5],
              'n_estimators': [200],
              'random_state': [42]}

ng_f = fs(pd.read_csv(file_dict['NG_F']))
es_f = fs(pd.read_csv(file_dict['ES_F']))
cl_f = fs(pd.read_csv(file_dict['CL_F']))
ohlc = ['Open', 'High', 'Low', 'Last']
volume_features = ['Bid_chg', 'Ask_chg', 'Bid%chg', 'Ask%chg', 'Delta']
supply_demand_features = ['D1_norm', 'D2_norm', 'S1_norm', 'S2_norm','H-L']
feats = volume_features + supply_demand_features

es_f =  es_f.resample('15min').apply(logic).dropna()
cl_f = cl_f.resample('15min').apply(logic).dropna()

cl_model = fb.model_prep(cl_f, ohlc=ohlc)
cl_model.supply_demand_zone(drop_non_normals=True)
cl_model.bid_ask_vol_features()
cl_model.dt_features()