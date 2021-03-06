#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 20:12:32 2020
"""

import datetime as dt
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt

from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA


def rsi(array, n):
# This function will only be called once for each instance.

    gain = pd.Series(array).diff()
    loss = gain.copy()
    gain[gain < 0] = 0
    # in the gain series, if gain is negative, make it zero
    loss[loss > 0] = 0
    # in the loss series if loss is positive, make it zero

    #rs = gain.ewm(n).mean() / loss.abs().ewm(n).mean()
    # ewm() -> apply EWMA
    # ewm() appears to be unsuitable here since it could lead to a very high
    # rsi in the very beginning of the period given the lack of previous data.
    rs = gain.rolling(n).mean() / loss.abs().rolling(n).mean()
    # rolling() -> apply SMA
    return 100 - 100 / (1 + rs)
    # According to my understanding, this is operator overloading and the
    # return value is a pd.Series.


class rsi_ls_close(Strategy):

    s_rsi = 6
    l_rsi = 24
    ww_rsi = 14
    oversold_level = 20
    overbought_level = 80

    def init(self):
        Close1 = self.data.Close
        self.rsi_14 = self.I(rsi, Close1, self.ww_rsi)

        self.st_rsi = self.I(rsi, Close1, self.s_rsi)
        self.lt_rsi = self.I(rsi, Close1, self.l_rsi)

    def next(self):
        if not self.position:
            if self.st_rsi < self.oversold_level and self.lt_rsi < self.oversold_level:
                self.buy()
            elif self.st_rsi > self.overbought_level and self.lt_rsi > self.overbought_level:
                self.sell()
        if self.position:
            if self.st_rsi < self.oversold_level or self.lt_rsi < self.oversold_level:
                self.position.close()
            elif self.st_rsi > self.overbought_level or self.lt_rsi > self.overbought_level:
                self.position.close()


class rsi_simple_close(Strategy):

    ww_rsi = 14
    oversold_level = 20
    overbought_level = 80

    def init(self):
        Close1 = self.data.Close
        self.rsi_14 = self.I(rsi, Close1, self.ww_rsi)

    def next(self):
        if not self.position:
            if self.rsi_14 < self.oversold_level:
                self.buy()
            elif self.rsi_14 > self.overbought_level:
                self.sell()
        if self.position:
            if self.rsi_14 < self.oversold_level:
                self.position.close()
            elif self.rsi_14 > self.overbought_level:
                self.position.close()


class rsi_simple(Strategy):

    ww_rsi = 14
    oversold_level = 30
    overbought_level = 70

    def init(self):
        Close1 = self.data.Close
        self.rsi_14 = self.I(rsi, Close1, self.ww_rsi)

    def next(self):
        if self.rsi_14 < self.oversold_level:
            self.buy()
        elif self.rsi_14 > self.overbought_level:
            self.sell()

class ema_cross(Strategy):
    # Appears to me that ema performs constantly better than sma
    # no matter if it is in a bear or a bull market.
    # span1 == 10 and span2 == 20 appears to be the best among 3 combinations
    # tested

    span1 = 10
    span2 = 20

    def init(self):

        Close1 = self.data.Close
        self.ma1 = self.I(func=self.ema, values=Close1, n=self.span1)
        self.ma2 = self.I(func=self.ema, values=Close1, n=self.span2)
        # Declare indicator. An indicator is just an array of values,
        # but one that is revealed gradually in Strategy.next() much like
        # Strategy.data is. Returns np.ndarray of indicator values.
        # https://kernc.github.io/backtesting.py/doc/backtesting/backtesting.html#backtesting.backtesting.Strategy.

        # SMA(): Returns n-period simple moving average of array arr.

    def next(self):
        # If mal crosses above ma2 , buy the asset
        if crossover(series1=self.ma1, series2=self.ma2):
            # Return True if series1 just crossed over series2.
            # https://kernc.github.io/backtesting.py/doc/backtesting/lib.html
            self.buy()
            # https://kernc.github.io/backtesting.py/doc/backtesting/backtesting.html
        # Else , if mal crosses below ma2 , sell it
        elif crossover(series1=self.ma2, series2=self.ma1):
            self.sell()

    def ema(self, values, n):

        return pd.Series(values).ewm(span=n).mean()
        # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ewm.html


class sma_cross(Strategy):
    # Strategy is BaseClassName

    period1 = 10
    period2 = 20

    def init(self):

        Close1 = self.data.Close
        self.ma1 = self.I(func=SMA, arr=Close1, n=self.period1)
        self.ma2 = self.I(func=SMA, arr=Close1, n=self.period2)
        # Declare indicator. An indicator is just an array of values,
        # but one that is revealed gradually in Strategy.next() much like
        # Strategy.data is. Returns np.ndarray of indicator values.
        # https://kernc.github.io/backtesting.py/doc/backtesting/backtesting.html#backtesting.backtesting.Strategy.

        # SMA(): Returns n-period simple moving average of array arr.

    def next(self):
        # If mal crosses above ma2 , buy the asset
        if crossover(series1=self.ma1, series2=self.ma2):
            # Return True if series1 just crossed over series2.
            # https://kernc.github.io/backtesting.py/doc/backtesting/lib.html
            self.buy()
            # https://kernc.github.io/backtesting.py/doc/backtesting/backtesting.html
        # Else , if mal crosses below ma2 , sell it
        elif crossover(series1=self.ma2, series2=self.ma1):
            self.sell()


def draw_moving_average(df, ticker: str, company_name: str):

    short_moving_avg = df.Close.rolling(window=20).mean()
    long_moving_avg = df.Close.rolling(window=20).mean()

    plt.subplots(figsize=(16, 9))
    plt.plot(df.Close.index, df.Close, label='{} ({})'.format(company_name, ticker), alpha=0.8)
    plt.plot(short_moving_avg.index, short_moving_avg, label='20-day MA')
    plt.plot(long_moving_avg.index, long_moving_avg, label='100-day MA')

    import numpy as np
    indicator = np.where(short_moving_avg > long_moving_avg, df['Close'].max() + 1, df['Close'].min() - 1)
    plt.plot(df.Close.index, indicator, alpha=0.3)
    #plt.plot(tsla)
    plt.xlabel('Date')
    plt.ylabel('Closing price ($)')
    plt.legend(loc='lower right')
#    ax =
    plt.show()


def main():

    #df = web.DataReader('IBM', 'yahoo', dt.date(2010, 1, 1), dt.date(2012, 1, 3))
    #draw_moving_average(df, 'IBM', 'IBM')

    df = web.DataReader('0001.hk', 'yahoo',
                        dt.date(2019, 1, 1), dt.date(2020, 12, 31))

    print('\nStrategy I: Simple Moving Average Cross')
    bt_sma_cross = Backtest(data=df, strategy=sma_cross,
                            cash=10000, commission=0.002)

    para2 = [20, 500, 200]
    for i in range(len(para2)):
        print(bt_sma_cross.run(period1=10, period2=para2[i]))
        bt_sma_cross.plot()
    # the plot cannot be shown if run in Spyder.

    print('\nStrategy II: Exponential Moving Average Cross')
    for i in range(len(para2)):
        bt_ema_cross = Backtest(data=df, strategy=ema_cross,
                                cash=10000, commission=0.002)
        print(bt_ema_cross.run(span1=10, span2=para2[i]))
        bt_ema_cross.plot()

    print('\nStrategy III: Simple Relative Strength Index')
    bt_rsi = Backtest(df, rsi_simple, cash=10000, commission=0.002)
    print(bt_rsi.run())
    bt_rsi.plot()
    # the plot cannot be shown if run in Spyder.

    print('\nStrategy IV: Simple Relative Strength Index with stop orders')
    bt_rsi_close = Backtest(df, rsi_simple_close, cash=10000, commission=0.002)
    print(bt_rsi_close.run())
    bt_rsi_close.plot()
    # the plot cannot be shown if run in Spyder.

    print('\nStrategy V: Relative Strength Index with long/short-term')
    bt_rsi_ls_close = Backtest(df, rsi_ls_close, cash=10000, commission=0.002)
    print(bt_rsi_ls_close.run())
    bt_rsi_ls_close.plot()


if __name__ == '__main__':

    main()
