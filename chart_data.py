import pandas as pd
import os

bar_data_PATH = "C:\\Users\\nicho\\charts\\"
files = [bar_data_PATH+i for i in os.listdir(bar_data_PATH)]
ticker_names = ['CL_F', 'ES_F', 'NG_F','ZC_F', 'ZS_F']
file_dict = dict(zip(ticker_names, files))
