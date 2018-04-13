from requests.auth import HTTPBasicAuth
import requests
import pandas as pd
import quandl as quandl
import datetime


def get_next_period(period):
    quarter_next = {'Q1': 'Q2', 'Q2': 'Q3', 'Q3': 'Q4', 'Q4': 'Q1_next'}
    return quarter_next[period]


def get_next_quarterly(period, prices_json):
    next_period = get_next_period(period)
    return prices_json[next_period]['date']


class DataPersistor(object):
    def __init__(self):
        quandl.ApiConfig.api_key = "kEEQaKt7AbyJ4yRLDHRg"
        self.user_intrinio = 'f5a8db553025c1f766ddc045f0e21b0f'
        self.password_intrinio = '0f1b9293cbf822568e7ca9d7d65086c2'

    def get_quarterly_statement_data(self, ticker, statement, year, period, prices_json,
                                     ticker_baseline='NASDAQOMX/COMP'):
        fundamental_json = self.get_fundamental_intrinio(period, statement, ticker, year)

        fundamental_dict = {}
        for item in fundamental_json:
            fundamental_dict[item['tag']] = item['value']

        quarterly_date = prices_json[period]['date']
        next_quarterly_date = get_next_quarterly(period, prices_json)

        avg_baseline_price = self.get_baseline_prices(quarterly_date, ticker_baseline)
        avg_next_baseline_price = self.get_baseline_prices(next_quarterly_date, ticker_baseline)

        fundamental_dict['ticker'] = ticker
        fundamental_dict['date'] = quarterly_date

        fundamental_dict['close_price'] = prices_json[period]['value']
        fundamental_dict['baseline_price'] = avg_baseline_price

        # price growth and baseline for next quarter - performance calculation
        fundamental_dict['close_price_next_quarter'] = prices_json[get_next_period(period)]['value']
        fundamental_dict['baseline_price_next_quarter'] = avg_next_baseline_price

        print('---- quarter prices -----')
        print("Quarter: {}. Price: {}. Base line: {}. Price next:{}. Base line next:{}".format(period,
                                                                                               fundamental_dict[
                                                                                                   'close_price'],
                                                                                               fundamental_dict[
                                                                                                   'baseline_price'],
                                                                                               fundamental_dict[
                                                                                                   'close_price_next_quarter'],
                                                                                               fundamental_dict[
                                                                                                   'baseline_price_next_quarter']))
        return fundamental_dict

    def get_baseline_prices(self, quarterly_date, ticker_baseline):
        quarterly_start_date = datetime.datetime.strptime(quarterly_date, '%Y-%m-%d') - datetime.timedelta(days=10)
        baseline_price = quandl.get(ticker_baseline, start_date=quarterly_start_date.strftime('%Y-%m-%d'),
                                    end_date=quarterly_date)
        avg_baseline_price = baseline_price['Index Value'].mean()
        return avg_baseline_price

    def get_prices_intrinio(self, ticker, year):
        next_year = year + 1
        url_price = "https://api.intrinio.com/historical_data?identifier=" + ticker + \
                    "&item=adj_close_price&frequency=quarterly&start_date=" + str(year) + \
                    "-01-01&end_date=" + str(next_year) + "-05-01"
        print(url_price)
        # Get prices
        response = requests.get(url_price, auth=HTTPBasicAuth(self.user_intrinio, self.password_intrinio))
        if not response.status_code == requests.codes.ok:
            raise Exception('Response from Intrinio is: ' + str(response.status_code))
        prices_json = response.json()['data']
        quarters = ['Q1', 'Q2', 'Q3', 'Q4', 'Q1_next']
        sorted_prices = sorted(prices_json, key=lambda k: k['date'])
        quarter_prices = dict(zip(quarters, sorted_prices))

        return quarter_prices

    def get_fundamental_intrinio(self, period, statement, ticker, year):
        url_fundamental = "https://api.intrinio.com/financials/standardized?identifier=" + ticker + \
                          "&statement=" + statement + "&fiscal_year=" + str(year) + "&fiscal_period=" + period
        response = requests.get(url_fundamental, auth=HTTPBasicAuth(self.user_intrinio, self.password_intrinio))
        if not response.status_code == requests.codes.ok:
            raise Exception('Response from Intrinio is: ' + str(response.status_code))
        fundamental_json = response.json()['data']
        print('---- fundamental -----')
        print(url_fundamental)
        return fundamental_json

    def get_all_statement_data(self, ticker, year, period, price_json):
        statements = ['income_statement', 'calculations', 'balance_sheet', 'cash_flow_statement']

        final_fundamental_dict = {}
        for statement in statements:
            result = self.get_quarterly_statement_data(ticker, statement, year, period, price_json)
            final_fundamental_dict.update(result)

        return final_fundamental_dict

    def get_all_time_data(self, ticker):
        quarters = ['Q1', 'Q2', 'Q3', 'Q4']
        result_list = []

        for year in range(2009, 2018):
            prices_json = self.get_prices_intrinio(ticker, year)
            for quarter in quarters:
                print('--------' + ticker + '-------')
                print(year)
                print(quarter)
                result_dict = self.get_all_statement_data(ticker, year, quarter, prices_json)
                result_list.append(result_dict)
                print(len(result_list))

        print('======= Result for ' + ticker + "========")
        # print(result_list)
        print(len(result_list))

        return result_list

    def get_time_series_data_tickers(self, tickers=['MSFT']):
        for ticker in tickers:
            data_list = self.get_all_time_data(ticker)
            df = pd.DataFrame(data_list)
            df.to_csv('/home/jeffchow/Dev/Projects/deepfundamental/resources/fundamental_' + ticker + '.csv',
                      encoding='utf-8', index=False)

        '''
        {'tag':'operatingrevenue', 'value':'0.56'},{...}
        
        This is going to be like
        
                     tag         value
0                  operatingrevenue  2.337150e+11
1                      totalrevenue  2.337150e+11
2            operatingcostofrevenue  1.400890e+11

        how to make the dict value to be columns? e.g.
        
        date operatingrevenue totalrevenue ...
        2018-3-9 2.33715 1.4
        
        '''

    def persistAllDataToCsv(self):
        # tickers = ['MSFT', 'AAPL', 'INTC', 'GOOGL', 'ADBE', 'EA', 'ORCL', 'CRM', 'IBM']
        # Nasdaq tickers in technology sector - 168 symbols
        tickers = ["VNET", "TWOU", "ACIW", "ATVI", "ADBE", "ALRM", "ALLT", "GOOG", "GOOGL", "ALTR", "AMSWA", "ANSS",
                   "APPF", "APPN",
                   "AAPL", "APTI", "ALOT", "TEAM", "ATTU", "ADSK", "AUTO", "AVID", "AWRE", "BOSC", "BIDU", "BAND",
                   "BNFT", "BKYI",
                   "BBOX", "BLKB", "BL", "EPAY", "BLIN", "BVSN", "CA", "CDNS", "CDLX", "CYOU", "CHKP", "CNIT",
                   "CNET", "CSCO", "CTXS", "CHUBA", "CHUBK", "CVLT", "CVON", "CVONW", "CSOD", "CPAH", "COUP", "CRAY",
                   "CYBR", "CYRN", "DWCH",
                   "DTRM", "DGII", "DBX", "ELON", "EDGW", "EGAN", "EA", "EFII", "EIGI", "EVBG", "EXTR", "FFIV", "FB",
                   "FEYE", "FSCT", "FRSX", "FTNT", "FSNN", "GDS", "GIGM", "GEC", "GSUM", "HSTM", "HDP", "ICLK", "INVE",
                   "INFO", "IMMR", "IMPV",
                   "INSE", "LINK", "INAP", "IIJI", "INTU", "KTCC", "LTRX", "LPSN", "LOGI", "LOGM", "MGIC", "MAMS",
                   "MANH", "MTCH", "MTLS", "MTBC", "MTBCP", "MDSO", "MSFT", "MSTR", "MIME", "MITK", "MOBL", "MOMO",
                   "MDB", "MOXC", "MYSZ", "NATI",
                   "NTWK", "NICE", "NUAN", "NTNX", "OKTA", "OMCL", "OSS", "PCTY", "COOL", "IPDN", "PRGS", "PTC", "QADA",
                   "QADB", "QLYS", "QUMU", "RDCM", "RSYS", "RPD", "RP", "SABR", "SPNS", "SCWX", "SHSP", "SSTI", "SIFY",
                   "SILC", "SINA", "SMSI",
                   "SCKT", "SPLK", "SPSC", "SSNC", "SSYS", "SMCI", "SYMC", "SYNC", "SNPS", "TTWO", "TLND", "DSGX",
                   "TSG", "TTD", "ULTI", "TISA", "TACT", "TRIP", "TRUE", "UPLD", "VRNS", "WEB", "WB", "WIX", "XPLR",
                   "XNET", "YNDX", "AMZN", "INTC"]
        self.get_time_series_data_tickers(tickers)
