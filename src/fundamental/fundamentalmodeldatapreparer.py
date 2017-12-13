import quandl as quandl
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller


def convert2Stationary(data_table):
    return None


def isStationary(dataFrameTable):
    """ Test for stationary
    :rtype: Boolean
    """
    for column in dataFrameTable.columns:
        data = dataFrameTable[column]
        seriesToTest = data.values
        pTestResult = adfuller(seriesToTest)
        if pTestResult[1] > 0.05:
            return False


class FundamentalModelDataPreparer(object):
    quandl.ApiConfig.api_key = "kEEQaKt7AbyJ4yRLDHRg"

    def get_data(self, ticker):
        tickers = []
        tickers.append(ticker)
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
        columns = ['calendardate', 'ncf', 'pe', 'pb', 'gp', 'currentratio']

        data_table = quandl.get_table('SHARADAR/SF1', qopts={"columns": columns}, dimension='MRQ', ticker=tickers)

        ''' Process data '''
        data_table = data_table.dropna(axis=0, how='any')
        if not isStationary(data_table):
            data_table = convert2Stationary(data_table)

        ''' Assign price performance - labels for supervised learning '''

        ''' Get the prices'''
        start = pd.to_datetime(data_table['calendardate'].values[0])
        end = pd.to_datetime(data_table['calendardate'].values[0])
        prices = quandl.get('EOD/' + ticker, start_date=start.strftime('%Y-%m-%d'), end_date=end.strftime('%Y-%m-%d'))
        adj_close_prices = prices.loc[:, "Adj_Close"]

        target_dates = data_table['calendardate'].values
        # To slice prices according to a set of dates
        # target_prices = adj_close_prices[adj_close_prices.index.isin(target_dates)]

        ''' Iterate target prices to look take average price for 10 days before the indicator date'''
        data_table.assign(avg_price='')
        for date in target_dates:
            end_date = date
            start_date = end_date - np.timedelta64(10, 'D')
            prices_to_avg = adj_close_prices.loc[start_date:end_date]
            data_table.loc[data_table['calendardate'] == date, ['avg_price']] = prices_to_avg.mean()

        return data_table
