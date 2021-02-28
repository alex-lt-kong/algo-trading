#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 20:12:32 2020

@author: mamsds
"""

import datetime as dt
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt

from backtesting import Backtest , Strategy
from backtesting.lib import crossover
from backtesting.test import SMA

def get_relative_strength_index(array, n):

    gain = pd.Series(array).diff()
    loss = gain.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    rs = gain.rolling(n).mean() / loss.abs().rolling(n).mean()

    return 100 - 100 / (1 + rs)

class rsi_ls_28_close(Strategy):

    s_rsi = 6
    l_rsi = 24
    ww_rsi = 14
    oversold_level = 20
    overbought_level = 80

    def init(self):
        Close1 = self.data.Close
        self.rsi_14 = self.I(get_relative_strength_index, Close1, self.ww_rsi)

        self.st_rsi = self.I(get_relative_strength_index, Close1, self.s_rsi)
        self.lt_rsi = self.I(get_relative_strength_index, Close1, self.l_rsi)

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

class rsi_simple_28_close(Strategy):

    ww_rsi = 14
    oversold_level = 20
    overbought_level = 80

    def init(self):
        Close1 = self.data.Close
        self.rsi_14 = self.I(get_relative_strength_index, Close1, self.ww_rsi)

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

class rsi_simple_28(Strategy):

    ww_rsi = 14
    oversold_level = 20
    overbought_level = 80

    def init(self):
        Close1 = self.data.Close
        self.rsi_14 = self.I(get_relative_strength_index, Close1, self.ww_rsi)

    def next(self):
        if self.rsi_14 < self.oversold_level:
            self.buy()
        elif self.rsi_14 > self.overbought_level:
            self.sell()

class simple_moving_average_Cross(Strategy):

    def init(self):
        Close1 = self.data.Close
        self.ma1 = self.I(SMA, Close1, 10)
        self.ma2 = self.I(SMA, Close1, 20)

    def next(self):
        # If mal crosses above ma2 , buy the asset
        if crossover(self.ma1 , self.ma2):
            self.buy()
        # Else , if mal crosses below ma2 , sell it
        elif crossover(self.ma2, self.ma1):
            self.sell()


def draw_moving_average(df, ticker: str,company_name: str):

    short_moving_avg = df.Close.rolling(window=20).mean()
    long_moving_avg = df.Close.rolling(window=20).mean()

    plt.subplots(figsize=(16, 9))
    plt.plot(df.Close.index, df.Close, label='{} ({})'.format(company_name, ticker), alpha = 0.8)
    plt.plot(short_moving_avg.index, short_moving_avg, label = '20-day MA')
    plt.plot(long_moving_avg.index, long_moving_avg, label = '100-day MA')

    import numpy as np
    indicator = np.where(short_moving_avg > long_moving_avg, df['Close'].max() + 1, df['Close'].min() - 1)
    plt.plot(df.Close.index, indicator, alpha = 0.3)
    #plt.plot(tsla)
    plt.xlabel('Date')
    plt.ylabel('Closing price ($)')
    plt.legend(loc = 'lower right')
#    ax =
    plt.show()

def main():

    #df = web.DataReader('0005.hk', 'yahoo', dt.date(2010, 1, 1), dt.date(2020, 8, 31))
    #draw_moving_average(df, '0005.HK', 'HSBC')
    df = web.DataReader('IBM', 'yahoo', dt.date(2010, 1, 1), dt.date(2012, 1, 3))
    draw_moving_average(df, 'IBM', 'IBM')
  #  print(df.head(20))

    df = web.DataReader('IBM', 'yahoo', dt.date(2012, 1, 3), dt.date(2018, 12, 31))

    print('Trading Strategy I: Simple Moving Average')
    bt_simple_moving_average_Cross = Backtest(df, simple_moving_average_Cross, cash = 10000, commission = 0.002)
    print(bt_simple_moving_average_Cross.run())
    bt_simple_moving_average_Cross.plot() # the plot cannot be shown if run in Spyder.

    print('Trading Strategy II: Simple Relative Strength Index')
    bt_rsi_28 = Backtest(df, rsi_simple_28, cash = 10000, commission = 0.002)
    print(bt_rsi_28.run())
    bt_rsi_28.plot() # the plot cannot be shown if run in Spyder.

    print('Trading Strategy III: Simple Relative Strength Index with stop orders')
    bt_rsi_28_close = Backtest(df, rsi_simple_28_close, cash = 10000, commission = 0.002)
    print(bt_rsi_28_close.run())
    bt_rsi_28_close.plot() # the plot cannot be shown if run in Spyder.

if __name__ == '__main__':
    main()
