import datetime as datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader as pdr
from numpy import unique
import yfinance as yf
import os
import sys
from matplotlib import rcParams
pd.set_option('display.max_rows', 15)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 500)


class Stocks():
    _allStocks = []
    investment = 0

    kind = 'ASX Equities'

    def __init__(self, company, code, price, units,
                 date, t_type):
        self.company = company
        self.code = code
        self.price = price
        self.units = units
        self.date = date
        self.t_type = t_type
        self.value = units * price
        if self.t_type == 'Buy':
            Stocks.investment += self.value
        Stocks._allStocks.append(self.code)
        self.random_id = hash(self.code)

    def get_trading_history(self):
        if self.t_type == 'Buy':
            return pd.DataFrame(pdr.DataReader('{}.AX'.format(self.code), 'yahoo', start=self.date)['Adj Close'])
        else:
            pass

    def get_current(self):
        """ Retrieves latest closing price """
        return np.round(pdr.DataReader('{}.AX'.format(self.code), 'yahoo',
                                       start=(datetime.datetime.today() - datetime.timedelta(days=1)))['Adj Close'][-1], 3)

    def percent_profit(self):
        """ Calculates profit of holding """

        if self.t_type == 'Buy':
            return np.round((self.get_current() - self.price)/self.price, 4)
        else:
            print("Cannot return profit for transaction_type = 'Sell'")

    def amount_profit(self):
        if self.t_type == 'Buy':
            return np.round(self.percent_profit() * self.value, 2)
        else:
            print("Cannot return profit for transaction_type = 'Sell'")

    def get_daily_returns(self):
        """ Calculates day on day change """
        if self.t_type == 'Buy':
            yesterday = pdr.DataReader('{}.AX'.format(self.code), 'yahoo',
                                       start=(datetime.datetime.today() - datetime.timedelta(days=1)))['Adj Close'][0]

            return np.round((((self.get_current() - yesterday)) / yesterday), 3)
        else:
            print("Cannot return profit for transaction_type = 'Sell'")

    def get_monthly_returns(self):
        """ Calculated monthly returns """

        if self.t_type == 'Buy':
            month = pdr.DataReader('{}.AX'.format(self.code), 'yahoo',
                                   start=(datetime.datetime.today() - datetime.timedelta(weeks=4)))['Adj Close'][0]

            return np.round((((self.get_current() - month)) / month), 3)
        else:
            print("Cannot return profit for transaction_type = 'Sell'")

    def adjust_purchase_price(self):
        if self.value > 1000:
            return self.price + (19.95/self.units)
        else:
            return self.price + (10/self.units)

    def as_dict(self):
        """ Convert data to dictionary """

        return {'Code': self.code, 'Company': self.company, 'Quantity': self.units,
                'Type': self.t_type, 'Date': self.date, 'Price': self.adjust_purchase_price(),
                'Total': self.units * self.adjust_purchase_price()}

    def __str__(self):
        return self.code

    def __repr__(self):
        return self.code


def recommended_variable_names(path):

    df = pd.read_csv(path, parse_dates=True)
    df['Variable Rec'] = df['Code'] + "_" + [word[0][0]
                                             for word in df['Type'].str.split()]
    counts = df.groupby('Variable Rec').cumcount()+1
    counts.index = df['Variable Rec']
    counts = pd.DataFrame(counts)
    counts.reset_index(inplace=True)
    counts.columns = ['Variable Rec', 'Counts']
    counts['Final Variable Suggestions'] = counts['Variable Rec'] + \
        counts['Counts'].astype(str)

    print("Variable set up is suggested as the following:",
          counts['Final Variable Suggestions'].tolist())


# path_to_transactions = os.path.abspath(
#     "/Users/tomrogers/Desktop/Goody_Transactions.csv")
# recommended_variable_names(path_to_transactions)

# CGF_B1 = Stocks('Challenger', 'CGF', 10.860, 95,
#                 datetime.datetime(2018, 4, 23), 'Buy')
TCL_B1 = Stocks('Transurban', 'TCL', 14.250, 140,
                datetime.datetime(2020, 10, 5), 'Buy')
RMD_B1 = Stocks('ResMed', 'RMD', 23.637, 100,
                datetime.datetime(2020, 10, 5), 'Buy')
LLC_B1 = Stocks('LendLease', 'LLC', 11.950, 165,
                datetime.datetime(2020, 10, 13), 'Buy')
SYD_B1 = Stocks('Sydney Airport', 'SYD', 6.45, 387,
                datetime.datetime(2020, 12, 17), 'Buy')
SYD_B2 = Stocks('Sydney Aiport', 'SYD', 5.625, 176,
                datetime.datetime(2021, 2, 14), 'Buy')
TCL_B2 = Stocks('Transurban', 'TCL', 12.740, 78,
                datetime.datetime(2021, 3, 4), 'Buy')
APX_B1 = Stocks('Appen', 'APX', 15.560, 64,
                datetime.datetime(2021, 3, 4), 'Buy')

stocks = [TCL_B1, RMD_B1, LLC_B1, SYD_B1, SYD_B2, TCL_B2, APX_B1]

transactions_df = pd.DataFrame([x.as_dict() for x in stocks])


def portfolio_plotting(portfolio: pd.Series, benchmark_code='^AXJO'):
    """ collecting the data """
    benchmark_return = pd.DataFrame(pdr.DataReader('{}'.format(
        benchmark_code), 'yahoo', start=portfolio.index[0])['Adj Close'])
    benchmark_return = benchmark_return.pct_change().cumsum().fillna(0)
    benchmark = benchmark_return.iloc[:, 0]
    portfolio = portfolio
    excess_returns = portfolio - benchmark
    excess_returns.fillna(method='pad', inplace=True)
    excess_returns.dropna(inplace=True)
    excess_returns = pd.DataFrame(excess_returns)
    excess_returns.columns = ['Excess Returns']
    print(excess_returns.tail(5))

    """ setting params for table """

    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = 'Tahoma'
    _, ax = plt.subplots(figsize=(12, 6))
    benchmark.plot(c='#E66F6F', label='{}'.format(
        benchmark_code), lw=2, alpha=0.75,)
    portfolio.plot(c='#1880AD', label='Portfolio', lw=2, alpha=0.75)
    # excess_returns.plot(color='red', label='Excess Returns', alpha=0.5)

    """ annotate the data points """

    style = dict(size=10, c='#194765', fontweight='bold')
    props = dict(boxstyle='square', facecolor='#A9BAC6', alpha=0.75)

    # max_date = excess_returns[excess_returns == max(
    #     excess_returns.values)].dropna().index.date[0]
    max_diff = round(excess_returns[excess_returns == max(
        excess_returns.values)].dropna().values[0][0], 3)
    # min_date = excess_returns[excess_returns == min(
    #     excess_returns.values)].dropna().index.date[0]
    min_diff = round(excess_returns[excess_returns == min(
        excess_returns.values)].dropna().values[0][0], 3)
    # last_date = excess_returns.index[-1]
    last_diff = round(excess_returns.iloc[-1].values[0], 3)
    daily_change = round(
        excess_returns.iloc[-1].values[0] - excess_returns.iloc[-2].values[0], 3)

    if daily_change == 0:
        print(excess_returns.tail(5))
        print("Possibly an issue with the data, check again during market hours")

    ax.text(0.025, 0.925, "High: " + str(max_diff),
            transform=ax.transAxes, **style, bbox=props)
    ax.text(0.025, 0.87, "Low: " + str(min_diff),
            transform=ax.transAxes, **style, bbox=props)
    ax.text(0.025, 0.815, "Last: " + str(last_diff),
            transform=ax.transAxes, **style, bbox=props)
    ax.text(0.025, 0.76, "Change: " + str(daily_change),
            transform=ax.transAxes, **style, bbox=props)

    """ finalising the plot """

    ax.set(title='Portfolio Returns vs. Benchmark Returns',
           xlabel='Date', ylabel='Returns')
    plt.xticks(rotation='horizontal', fontsize=8)
    plt.yticks(fontsize=8)
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.xlim(portfolio.index.min(), portfolio.index.max())
    plt.show()


def stock_plotting(transaction_df: pd.DataFrame, benchmark_code='^AXJO'):
    df_codes = transactions_df.groupby('Code').first().Date
    df_codes = pd.DataFrame(df_codes)

    df = pd.DataFrame()

    empty_dict = {}
    for index, start_date in df_codes.iterrows():
        empty_dict[index] = pd.DataFrame('{}.AX'.format(
            index), start=start_date.Date)['Adj Close']

    for key, value in empty_dict.items():
        empty_dict[key] = (value.div(value.iloc[0])) - 1

    for key, value in empty_dict.items():
        empty_dict[key] = pd.DataFrame(value)

    df_list = [v for k, v in empty_dict.items()]
    df = pd.concat(df_list, axis=1)
    df.columns = sorted(empty_dict.keys())

    benchmark_return = pd.DataFrame(pdr.DataReader('{}'.format(
        benchmark_code), 'yahoo', start=df.index[0])['Adj Close'])
    benchmark_return = benchmark_return.pct_change().cumsum().fillna(0)

    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = 'Tahoma'
    df.plot(subplots=True, layout=(4, 3),
            label='Portfolio', lw=2, alpha=0.75, grid=True)
    # benchmark.plot(ax=ax, c='#38ADBB', label='{}'.format(
    #     benchmark_code), lw=2, alpha=0.25)
    # ax.set(title='Individual Stock Returns vs. Benchmark Returns',
    #        xlabel='Date', ylabel='Returns')
    # plt.yticks(fontsize=8)
    # plt.legend(loc='upper right')
    # plt.grid(True)
    plt.tight_layout()
    plt.show()


def calculate_returns(dataframe: pd.DataFrame):

    assert dataframe.columns.tolist() == [
        'Code', 'Company', 'Quantity', 'Type', 'Date', 'Price', 'Total'], "check your dataframe columns"

    dataframe['Date'] = pd.to_datetime(
        dataframe['Date'], dayfirst=True)
    dataframe = dataframe.sort_values('Date')

    df = dataframe

    df['Quantity Movement'] = np.where(df.Type.str.contains(
        'Sell'), df['Quantity']*-1, df['Quantity'])
    df['Quantity CumSum'] = df.groupby('Code')['Quantity Movement'].cumsum()

    print('Program is Running')

    dates_list = []
    for date in df['Date']:
        dates_list.append(date)

    value_list = []
    for date in dates_list:
        value_list.append(df[df['Date'] <= date].sort_values(
            'Date').groupby('Code')[['Code', 'Quantity CumSum']].tail(1))

    daily_prices = pd.DataFrame()
    for x in set(df.Code):
        daily_prices[x] = yf.download('{}.AX'.format(
            x), start=sorted(dates_list)[0])['Close']
    daily_prices.reset_index(inplace=True)

    transaction_idx = []
    for date in dates_list:
        transaction_idx.append(
            daily_prices.index[daily_prices['Date'] == date].tolist())
    print(transaction_idx)
    transaction_idx = sum(transaction_idx, [])

    daily_units = pd.DataFrame(np.zeros(daily_prices.set_index('Date').shape))
    daily_units.columns = daily_prices.set_index('Date').columns

    dict_idx_units = dict(zip(transaction_idx, value_list))

    column_order = sorted(
        max(enumerate(value_list), key=lambda x: len(x[1]))[1].Code.tolist())
    column_order = list(enumerate(column_order))
    column_order = [(a, b) for b, a in column_order]

    for key, _ in dict_idx_units.items():
        mapping_array = np.zeros_like(daily_units.iloc[key])
        for _, row in dict_idx_units[key].iterrows():
            for item in column_order:
                if row.Code == item[0]:
                    mapping_array[item[1]] = row['Quantity CumSum']
                    print(key, ":", 'Portfolio Currently Consists Of:',
                          item[0], str(int(mapping_array[item[1]])))
                else:
                    pass
                daily_units.iloc[key] = mapping_array

    daily_units = daily_units.replace(to_replace=0, method='ffill')
    daily_units.set_index(daily_prices['Date'], inplace=True)
    daily_prices.set_index('Date', inplace=True)
    daily_units.columns = sorted(daily_units.columns)
    dp_cols = daily_prices.columns.tolist()
    dp_cols = sorted(dp_cols)
    daily_prices = daily_prices[dp_cols]

    daily_total = daily_units * daily_prices
    daily_total['Total'] = daily_total.sum(axis=1)

    for col in daily_total.columns:
        daily_total[col] = daily_total[col]/daily_total['Total']

    daily_change = daily_prices.pct_change().cumsum()
    daily_total.drop('Total', inplace=True, axis=1)
    daily_cum_change = daily_change * daily_total
    daily_cum_change['Total'] = daily_cum_change.sum(axis=1)
    portfolio = daily_cum_change['Total']

    return portfolio_plotting(portfolio)


# transactions_df = pd.read_csv(path_to_transactions, parse_dates=True)
calculate_returns(transactions_df)


"""
Known Issues / Future Upgrades
    * International Stocks
    * Mapping Individual Stock Performance (Subplots)
    * Ensuring the Date Lodged in the Spreadsheet is a Traded Day
    * Establishing Profit and/or Effective Price of Units Sold
"""
