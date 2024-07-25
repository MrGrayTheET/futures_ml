import pandas as pd

sentiment = pd.read_csv('/content/futures_ml/data/sentiment.csv', index_col='Date', date_format='%YY-%mm-%dd', parse_dates=True)
sentiment = sentiment.replace('%', '', regex=True)
sentiment = sentiment.replace(',', '', regex=True)

sentiment.index = pd.to_datetime(sentiment.index)

sentiment.to_csv('sentiment_2')


class cot_data:

    def __init__(self):
        self.aggregated_reports = pd.read_csv('/content/futures_ml/data/agg_legacy_reports', index_col='As of Date in Form YYYY-MM-DD')

        self.filtered_df = self.aggregated_reports[["Market and Exchange Names", 'CFTC Contract Market Code',
                                                    "Noncommercial Positions-Long (All)",
                                                    "Noncommercial Positions-Short (All)",
                                                    "Change in Noncommercial-Long (All)",
                                                    "Change in Noncommercial-Short (All)",
                                                    "% of OI-Noncommercial-Long (All)",
                                                    "% of OI-Noncommercial-Short (All)",
                                                    'Change in Commercial-Long (All)',
                                                    'Change in Commercial-Short (All)', '% of OI-Commercial-Long (All)',
                                                    '% of OI-Commercial-Short (All)', '% of OI-Commercial-Long (Old)',
                                                    '% of OI-Commercial-Short (Old)',
                                                    '% of OI-Commercial-Long (Other)', ]]

        self.non_commercials = ["Market and Exchange Names", 'CFTC Contract Market Code',
                                                    "Noncommercial Positions-Long (All)",
                                                    "Noncommercial Positions-Short (All)",
                                                    "Change in Noncommercial-Long (All)",
                                                    "Change in Noncommercial-Short (All)",
                                                    "% of OI-Noncommercial-Long (All)",
                                                    "% of OI-Noncommercial-Short (All)",]


        self.filtered_df.index = pd.to_datetime(self.filtered_df.index)

        self.code_dict = {'CL_F': ['067651'], 'ES_F': ['13874A'], 'NQ_F': ['209742'], 'LE': ['057642']}

        None

    def contract_data(self, contract, by_ticker=True):
        filter = self.filtered_df['CFTC Contract Market Code'].isin(self.code_dict.get(contract))

        return self.filtered_df[filter].sort_index()

    def download_cot_year_list(self, year_list, report_type='legacy_fut'):
        for i in year_list:
            cot_i = cot_reports.cot_year(i)
            self.aggregated_reports = pd.concat([self.aggregated_reports, cot_i])

        return self.aggregated_reports


