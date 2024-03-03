import asyncio

import numpy as np
import pandas as pd
import pandas_ta as ta
import vectorbt as vbt
import os
import yfinance as yf
import talib 
from datetime import datetime, timedelta, timezone
import pytz


def mark_ticker(ticker, holds, buys, sells, entries, exits, tf):
    utc=pytz.UTC

    buy = entries[entries]
    sell = exits[exits]
    delta = int(tf[:-1])
    delta = max(delta, 2)
    interval = tf[-1]
    end_date = datetime.now()
    if (interval == 'm'):
        start_date = end_date - timedelta(minutes=delta)
    elif (interval == 'h'):
        start_date = end_date - timedelta(hours=delta)
    else:
        start_date = end_date - timedelta(days=delta)

    ret  = 0
    if ((len(buy)> 0) and (buy.tail(1).index.item() > start_date) ):
        ret = 1
    if ((len(sell)> 0) and sell.tail(1).index.item() > buy.tail(1).index.item()):
        ret = -1

    if (ret == 1 ):
        print("buy", buy.tail(1).index.item())
        buys.append(ticker)
    elif (ret == -1):
        print("sell", sell.tail(1).index.item())
        sells.append(ticker)
    else:
        holds.append(ticker)

def my_signal(ticker,  entries, exits, tf):
    utc=pytz.UTC

    buy = entries[entries]
    sell = exits[exits]
    delta = int(tf[:-1])
    delta = max(delta, 2)

    interval = tf[-1]
    end_date = datetime.now()
    if (interval == 'm'):
        start_date = end_date - timedelta(minutes=delta)
    elif (interval == 'h'):
        start_date = end_date - timedelta(hours=delta)
    else:
        start_date = end_date - timedelta(days=delta)

    # if (len(buy) > 0):
    #         print("buy", buy.tail(1).index.item(), start_date, delta, end_date)
    # if (len(sell) > 0):
    #         print("sell", sell.tail(1).index.item(), start_date)

    ret = 0
    if ((len(buy)> 0) and (buy.tail(1).index.item() >= start_date) ):
        ret = 1
    if ((len(sell)> 0) and sell.tail(1).index.item() > buy.tail(1).index.item()):
        ret = -1
    return ret

        
def strategy_trends(ticker, startdate, enddate, tf, flag_obv='close', flag_sig='close', straSE='obv', period=3):
    df = yf.download(ticker, 
                      start = startdate, 
                      end = enddate, 
                      interval=tf).fillna(0)
    # print(df.columns)
    df.columns = df.columns.str.lower()
#     print(df[['open','high','low','close']])
    # hammer pattern
    # res = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
    # res_ =talib.CDLHANGINGMAN(df['open'], df['high'], df['low'], df['close'])

    #
    #3candle
    #
    # res = talib.CDL3INSIDE(df['open'], df['high'], df['low'], df['close'])

    if (straSE == 'CDL3O'):
        res = talib.CDL3OUTSIDE(df[flag_obv], df['high'], df['low'], df['close'])
        entries =  res == 100
        exits = res == -100
    elif (straSE == 'PSMA'): # price cross SMA 
        price = df[flag_sig]
        
        pri_sma = talib.SMA(price, timeperiod=period)
        trend = np.where(price > pri_sma, 1, 0)
        trend = np.where(price < pri_sma, -1, trend)
        entries =  trend == 1
        exits = trend == -1               
    elif (straSE == 'RSMA'): # reverse price cross SMA 
        price = df[flag_sig]
        pri_sma = talib.SMA(price, timeperiod=period)
        trend = np.where(price > pri_sma, -1, 0)
        trend = np.where(price < pri_sma, 1, trend)
        entries =  trend == 1
        exits = trend == -1               
    elif (straSE == "SCAL"):
        # Identifying pivot highs
        df['Pivot_High'] = (df['high'] > df['high'].shift(1)) & (df['high'] > df['high'].shift(-1))

        # Identifying pivot lows
        df['Pivot_Low'] = (df['low'] < df['low'].shift(1)) & (df['low'] < df['low'].shift(-1))

        # Optionally, to make the indicator more robust, you can consider the high/low of two bars before and after
        # Adjust the conditions accordingly if you want to include this in your logic

        # Filter to see only the rows where a pivot high or pivot low is identified
        trend = np.where(df['Pivot_Low'], 1, 0)
        trend = np.where(df['Pivot_High'], -1, trend)
        entries =  trend == 1
        exits = trend == -1               

    else: #(straSE == 'obv'
        obv = talib.OBV(df[flag_obv], df['volume'])
        obv_ema = talib.EMA(obv, timeperiod=period)
        entries =  obv > obv_ema
        exits = obv < obv_ema
        
    pf = vbt.Portfolio.from_signals(df[flag_sig], entries, exits)
    
    return pf, entries, exits

# RSI = vbt.IndicatorFactory.from_talib('RSI')
window_delta = 10
def custom_indicator_obv(close, volumn, rsi_window, ma_window_slow,  entry, exit):
    close_5m = close.resample("5T").last()
    print("close_tm",close_5m)
#     rsi = vbt.RSI.run(close,  window = rsi_window).rsi
    obv = vbt.OBV.run(close,  window = rsi_window).rsi
#     print("rsi", rsi)
    ma_window_fast = ma_window_slow - window_delta
    ma_slow = vbt.MA.run(close, ma_window_slow).ma.to_numpy()
    ma_fast = vbt.MA.run(close, ma_window_fast).ma.to_numpy()
    trend = np.where(rsi > exit , -1, 0)
    trend = np.where( (rsi < entry) & (close < ma), 1, trend)
    return trend

def custom_indicator(close, rsi_window, ma_window_slow,  entry, exit):
#     close_5m = close.resample("5T").last()
#     print("close_tm",close_5m)
    rsi = vbt.RSI.run(close,  window = rsi_window).rsi
#     obv = vbt.OBV.run(close,  window = rsi_window).rsi
#     print("rsi", rsi)
    ma_window_fast = ma_window_slow - window_delta
    ma_slow = vbt.MA.run(close, ma_window_slow).ma.to_numpy()
    ma_fast = vbt.MA.run(close, ma_window_fast).ma.to_numpy()    
    trend = np.where((rsi > exit) & (ma_fast < ma_slow), -1, 0)
    trend = np.where( (rsi < entry) & ( ma_fast > ma_slow), 1, trend)
    return trend

def labeling_indicator(close, delta):
    shift_delta = delta
    shift_delta2 = delta *2
    shift_delta3 = delta *3
    value_delta = 1.5

    exits = (((close > close.shift(-shift_delta)) & \
            (close > close.shift(shift_delta))) & \
            ((close > close.shift(-shift_delta2)) & \
            (close > close.shift(shift_delta2)) & \
            (close > close.shift(shift_delta3)) & \
            (close > close.shift(-shift_delta3) )
            ))

    entries = (((close < close.shift(-shift_delta)) & \
            (close < close.shift(shift_delta))) & \
            ((close < close.shift(shift_delta2)) & \
            (close < close.shift(-shift_delta2)) & \
            (close < close.shift(shift_delta3)) & \
            (close < close.shift(-shift_delta3))
            ))     
    trend = np.where(exits, -1, 0)
    trend = np.where(entries, 1, trend)
    return trend

# RSI = vbt.IndicatorFactory.from_talib('RSI')
window_delta = 10
def custom_indicator_obv(close, volumn, rsi_window, ma_window_slow,  entry, exit):
    close_5m = close.resample("5T").last()
    print("close_tm",close_5m)
#     rsi = vbt.RSI.run(close,  window = rsi_window).rsi
    obv = vbt.OBV.run(close,  window = rsi_window).rsi
#     print("rsi", rsi)
    ma_window_fast = ma_window_slow - window_delta
    ma_slow = vbt.MA.run(close, ma_window_slow).ma.to_numpy()
    ma_fast = vbt.MA.run(close, ma_window_fast).ma.to_numpy()
    trend = np.where(rsi > exit , -1, 0)
    trend = np.where( (rsi < entry) & (close < ma), 1, trend)
    return trend

def custom_indicator(close, rsi_window, ma_window_slow,  entry, exit):
#     close_5m = close.resample("5T").last()
#     print("close_tm",close_5m)
    rsi = vbt.RSI.run(close,  window = rsi_window).rsi
#     obv = vbt.OBV.run(close,  window = rsi_window).rsi
#     print("rsi", rsi)
    ma_window_fast = ma_window_slow - window_delta
    ma_slow = vbt.MA.run(close, ma_window_slow).ma.to_numpy()
    ma_fast = vbt.MA.run(close, ma_window_fast).ma.to_numpy()    
    trend = np.where((rsi > exit) & (ma_fast < ma_slow), -1, 0)
    trend = np.where( (rsi < entry) & ( ma_fast > ma_slow), 1, trend)
    return trend

def custom_indicator_1m(close, rsi_window = 14,ma_window_slow = 20,  entry = 30, exit = 70):
    close_5m = close.resample("5T").last()
#     print(close_5m)
    rsi = vbt.RSI.run(close_5m,  window = rsi_window).rsi
    rsi, _= rsi.align(close, 
                      broadcast_axis = 0,
                     method = 'ffill',
                     join = 'right')
    rsi = rsi.to_numpy()
    close = close.to_numpy()
    ma_window_fast = ma_window_slow - window_delta
    ma_slow = vbt.MA.run(close, ma_window_slow).ma.to_numpy()
    ma_fast = vbt.MA.run(close, ma_window_fast).ma.to_numpy()
    trend = np.where(rsi > exit , -1, 0)
    trend = np.where( (rsi < entry) & (close < ma), 1, trend)
    return trend

def jinsong_indicator(rk50, rk100, rk250, rkprice,   entry, exit):
    trend = np.where((((rk50 + rk100)*100/150) >= exit) & (rk100 >= exit), -1, 0)
    trend = np.where( (((rk50 + rk100)*100/150) <= entry) & ((rk100 <= entry)| (rk250 == 1)), 1, trend)
    return trend

def jinsong_indicator_doublecheck(rk50, rk100, rk250, rkprice, deltaclose,  entry, exit):
    trend = np.where((((rk50 + rk100)/150*100)>= exit) & (rk100 >= exit) & (deltaclose<0), -1, 0)
    trend = np.where( (((rk50 + rk100)/150*100)<= entry) & (rk100 <= entry)& (deltaclose>0), 1, trend)
    return trend

# def cross3sma_indicator(Close, sma5, sma10, sma20, sma50,  entry, exit):
#     trend = np.where((((rk50 + rk100)/150*100)>= exit) & (rk100 >= exit) & (deltaclose<0), -1, 0)
#     trend = np.where( (((rk50 + rk100)/150*100)<= entry) & (rk100 <= entry)& (deltaclose>0), 1, trend)
#     return trend

def sma_indicator(Day, cross):
    trend = np.where((Day > 10) & (cross < 0) , -1, 0)
    trend = np.where((Day > 10) & (cross > 0), 1, trend)
    return trend


def sept_indicator(Month, Day):
    trend = np.where((Month == 9) & (Day < 10) , -1, 0)
    trend = np.where((Day > 10) & (Month == 9), 1, trend)
    return trend

# def trends(df: pd.DataFrame, mamode: str = "sma", fast: int = 50, slow: int = 200):
#     return ta.ma(mamode, df.close, length=fast) > ta.ma(mamode, df.close, length=slow) # SMA(fast) > SMA(slow) "Golden/Death Cross"

def ou_sma_indicator(Close, SMA_10, SMA_20, SMA_50, SMA_100, SMA_200, period, op):
    if (period == 20):
        price = SMA_20
    elif (period == 50):
        price = SMA_50
    elif (period == 100):
        price = SMA_100
    elif (period == 200):
        price = SMA_200
    else:
        price = SMA_10
        
    trend = np.where(price < Close, op, 0)
    trend = np.where(price > Close, -1 * op, trend)
    
    return trend

ou_ind = vbt.IndicatorFactory(
    class_name = "OuSMACross",
    short_name = "OuSMA",
    input_names = ["Close", "SMA_10", "SMA_20", "SMA_50", "SMA_100", "SMA_200"],
    param_names = ["period", 
                   "op"],
    output_names = ["value"]
    ).from_apply_func(
         ou_sma_indicator,
         period = 50,
         op = 1,
         keep_pd = True
        )

ind = vbt.IndicatorFactory(
    class_name = "Combination",
    short_name = "comb",
    input_names = ["close"],
    param_names = ["rsi_window", 
                   "ma_window_slow", 
#                    "ma_window_fast", 
                   "entry", 
                   "exit"],
    output_names = ["value"]
    ).from_apply_func(
         custom_indicator,
         rsi_window = 14,
         ma_window_slow = 50,
#          ma_window_fast = 20,
         entry = 30,
         exit = 70,         
         keep_pd = True
        )

sept_ind = vbt.IndicatorFactory(
    class_name = "BnH_Sept",
    short_name = "Sept",
    input_names = ['Month', 'Day'],
    output_names = ["value"]
    ).from_apply_func(
         sept_indicator,
         keep_pd = True
        )

sma_ind = vbt.IndicatorFactory(
    class_name = "Simple_SMA",
    short_name = "SSMA",
    input_names = ['Day', 'cross'],
    output_names = ["value"]
    ).from_apply_func(
         sma_indicator,
         keep_pd = True
        )

jc_ind = vbt.IndicatorFactory(
    class_name = "Jinsong",
    short_name = "js",
    input_names = ['rk50', 'rk100', 'rk250', 'rkprice'],
    param_names = ["entry", 
                   "exit"],
    output_names = ["value"]
    ).from_apply_func(
         jinsong_indicator,
         entry = 1,
         exit = 100,         
         keep_pd = True
        )

def trends(df: pd.DataFrame, mamode: str = "sma", fast: int = 50, slow: int = 200):
    return ta.ma(mamode, df.close, length=fast) > ta.ma(mamode, df.close, length=slow),ta.ma(mamode, df.close, length=fast) < ta.ma(mamode, df.close, length=slow) # SMA(fast) > SMA(slow) "Golden/Death Cross"
#     return ta.increasing(ta.ma(mamode, df.close, length=fast)) # Increasing MA(fast)
#     return ta.macd(df.close, fast, slow).iloc[:,1] > 0 # MACD Histogram is positive

def trends_ou_sma(df: pd.DataFrame, period: int = 50, op: int = 1):        
    res = ou_ind.run(df.Close, df.SMA_10, df.SMA_20, df.SMA_50, df.SMA_100, df.SMA_200, 
                     period = period, 
                     op = op)
    entries = res.value == 1.0
    exits = res.value == -1.0
    return entries,exits
        
def trends_rsi(df: pd.DataFrame, rsi_window: int = 14, ma_window: int = 20, entry: int = 30, exit: int = 70):
    res = ind.run(df.Close,
                 rsi_window = rsi_window, 
                 ma_window_slow = ma_window,
                 entry = entry,
                 exit = exit,
                 param_product = True)
    entries = res.value == 1.0
    exits = res.value == -1.0
#     trend = np.where(exits , -1, 0)
#     trend = np.where( entries, 1, trend)    
#     return  trend # rsi
    return entries,exits

def trends_sma(df: pd.DataFrame,  entry: int = 1, exit: int = 100):
    res = sma_ind.run(df.Day, df.cross,
                 param_product = True)
    entries = res.value == 1.0
    exits = res.value == -1.0
    return entries, exits

def trends_sept(df: pd.DataFrame,  entry: int = 1, exit: int = 100):
    res = sept_ind.run(df.Month, df.Day,
                 param_product = True)
    entries = res.value == 1.0
    exits = res.value == -1.0
    return entries, exits

def trends_jinsong(df: pd.DataFrame,  entry: int = 1, exit: int = 100):
    res = jc_ind.run(df.rk50, df.rk100, df.rk250, df.rkprice,
                 entry = entry,
                 exit = exit,
                 param_product = True)
    entries = res.value == 1.0
    exits = res.value == -1.0
    return entries, exits

def trends_jinsong_delta(df: pd.DataFrame,  entry: int = 1, exit: int = 98):
    res = jc_ind_delta.run(df.rk50, df.rk100, df.rk250, df.rkprice,df.deltaclose,
                 entry = entry,
                 exit = exit,
                 param_product = True)
    entries = res.value == 1.0
    exits = res.value == -1.0
    return entries, exits