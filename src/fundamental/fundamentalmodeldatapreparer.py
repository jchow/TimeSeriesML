import quandl as quandl
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler


def convert2Stationary(data_table):
    return data_table  # TODO make it time stationary by differencing maybe


def isStationary(data_frame_table):
    """ Test for stationary
    :rtype: Boolean
    """
    for column in data_frame_table.columns:
        data = data_frame_table[column]
        seriesToTest = data.values
        ''' Note that maxlag should be < nobs (num of observations), if not specified it is calculated as
        12*(nobs/100)^{1/4}. Need to specify it if nobs is too small'''
        pTestResult = adfuller(seriesToTest, maxlag=4)
        if pTestResult[1] > 0.05:
            return False


class FundamentalModelDataPreparer(object):
    quandl.ApiConfig.api_key = "kEEQaKt7AbyJ4yRLDHRg"

    def __init__(self, predict_single=True):
        self.single = predict_single

    def get_dataset(self, tickers=['MSFT', 'AAPL', 'INTC', 'IBM']):
        all_data_table = pd.DataFrame()
        all_labels = []
        for ticker in tickers:
            data_table, labels = self.get_ticker_data(ticker)
            all_data_table = all_data_table.append(data_table, ignore_index=True)
            self.add_labels(all_labels, labels)

        data_value_arrays = all_data_table.values
        data_value_arrays = data_value_arrays.astype('float32')

        # normalize features
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data_value_arrays = scaler.fit_transform(data_value_arrays)
        sequence_length = scaled_data_value_arrays.shape[0] / len(tickers)  # make each ticker its own sequence
        return self.to_sequence_data(scaled_data_value_arrays, int(sequence_length)), all_labels

    def add_labels(self, all_labels, labels):
        if self.single:
            all_labels.append(labels[-1])  # naive for the moment
        else:
            all_labels.append([labels])

    @staticmethod
    def to_sequence_data(data_arrays, seq_num=1):
        return data_arrays.reshape((data_arrays.shape[0]//seq_num, seq_num, data_arrays.shape[1]))

    def get_ticker_data(self, ticker, ticker_baseline='NASDAQOMX/COMP'):
        tickers = [ticker]
        '''
        columns = ['accoci', 'assets', 'assetsc',
       'assetsnc', 'assetturnover', 'bvps', 'capex', 'cashneq',
       'cashnequsd', 'cor', 'consolinc', 'currentratio', 'de', 'debt',
       'debtc', 'debtnc', 'debtusd', 'deferredrev', 'depamor', 'deposits',
       'divyield', 'dps', 'ebit', 'ebitda', 'ebitdamargin', 'ebitdausd',
       'ebitusd', 'ebt', 'eps', 'epsdil', 'epsusd', 'equity', 'equityavg',
       'equityusd', 'ev', 'evebit', 'evebitda', 'fcf', 'fcfps', 'fxusd',
       'gp', 'grossmargin', 'intangibles', 'intexp', 'invcap', 'invcapavg',
       'inventory', 'investments', 'investmentsc', 'investmentsnc',
       'liabilities', 'liabilitiesc', 'liabilitiesnc', 'marketcap', 'ncf',
       'ncfbus', 'ncfcommon', 'ncfdebt', 'ncfdiv', 'ncff', 'ncfi',
       'ncfinv', 'ncfo', 'ncfx', 'netinc', 'netinccmn', 'netinccmnusd',
       'netincdis', 'netincnci', 'netmargin', 'opex', 'opinc', 'payables',
       'payoutratio', 'pb', 'pe', 'pe1', 'ppnenet', 'prefdivis', 'price',
       'ps', 'ps1', 'receivables', 'retearn', 'revenue', 'revenueusd',
       'rnd', 'roa', 'roe', 'roic', 'ros', 'sbcomp', 'sgna', 'sharefactor',
       'sharesbas', 'shareswa', 'shareswadil', 'sps', 'tangibles',
       'taxassets', 'taxexp', 'tbvps', 'workingcapital']
        '''
        columns = ['calendardate', 'pe', 'pb', 'gp', 'currentratio']

        data_table = quandl.get_table('SHARADAR/SF1', qopts={"columns": columns}, dimension='MRY', ticker=tickers)

        ''' Process data '''
        data_table = data_table.dropna(axis=0, how='any')

        ''' Assign price performance - labels for supervised learning '''

        '''Get the prices - start date align with fundamental start date minus 10 days for the avg later. End date is 1 year after the last date of 
        the fundamental data '''
        start = pd.to_datetime(data_table['calendardate'].values[0]) - np.timedelta64(10, 'D')
        end = pd.to_datetime(data_table['calendardate'].values[-1]) + np.timedelta64(1, 'Y')
        prices = quandl.get('EOD/' + ticker, start_date=start.strftime('%Y-%m-%d'), end_date=end.strftime('%Y-%m-%d'))
        baseline_prices = quandl.get(ticker_baseline, start_date=start.strftime('%Y-%m-%d'),
                                     end_date=end.strftime('%Y-%m-%d'))
        all_prices = prices.assign(Baseline=baseline_prices['Index Value'])
        all_close_prices = all_prices.loc[:, ['Adj_Close', 'Baseline']]

        target_dates = data_table['calendardate'].values
        # To slice prices according to a set of dates
        # target_prices = adj_close_prices[adj_close_prices.index.isin(target_dates)]

        ''' Iterate target prices to take average price for 10 days before the indicator date'''
        data_table.assign(avg_price='')
        data_table.assign(baseline_price='')
        # data_table.assign(performance='')

        performance_label = []
        for date in target_dates:
            end_date = date
            start_date = end_date - np.timedelta64(10, 'D')
            prices_to_avg = all_close_prices.loc[start_date:end_date]
            data_table.loc[data_table['calendardate'] >= pd.to_datetime(date), 'avg_price'] = prices_to_avg.mean()[0]
            data_table.loc[data_table['calendardate'] >= pd.to_datetime(date), 'baseline_price'] = prices_to_avg.mean()[
                1]

            '''Labels - how much does the price after 3 mths approximately out perform an index/etf tracking index 
            '''
            start_date_3mths = date + np.timedelta64(90, 'D')
            end_date_3mths = start_date_3mths + np.timedelta64(10, 'D')
            prices_3mths_to_avg = all_close_prices.loc[start_date_3mths:end_date_3mths]
            if not prices_3mths_to_avg.empty:
                data = (prices_3mths_to_avg.mean()[0] - prices_to_avg.mean()[0]) / prices_to_avg.mean()[0] - \
                   (prices_3mths_to_avg.mean()[1] - prices_to_avg.mean()[1]) / prices_to_avg.mean()[1]
                performance_label.append(data)

        ''' Get rid of dates column as we do not need it '''
        data_table = data_table.drop(['calendardate'], axis=1)
        data_table = data_table.dropna(axis=0, how='any')

        if not isStationary(data_table):
            data_table = convert2Stationary(data_table)

        return data_table, performance_label
