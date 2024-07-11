import pandas as pd



def format_qt_dts(str_series):
    time_format = '%m/%d/%Y %H:%M:%S %p'
    dt_series = pd.to_datetime(str_series, format='%m/%d/%Y %I:%M:%S %p -06:00',)
    return dt_series

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