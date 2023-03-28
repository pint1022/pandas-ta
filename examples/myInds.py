import numpy as np
import pandas as pd
import pandas_ta as ta
import vectorbt as vbt
import os
import yfinance as yf
import talib
from datetime import datetime, timedelta, timezone

def mark_ticker(ticker, holds, buys, sells, entries, exits, tf):
    buy = entries[entries]
    sell = exits[exits]
    delta = int(tf[:-1])
    interval = tf[-1]
    end_date = datetime.now(timezone.utc)
    if (interval == 'm'):
        start_date = end_date - timedelta(minutes=delta)
    elif (interval == 'h'):
        start_date = end_date - timedelta(hours=delta)
    else:
        start_date = end_date - timedelta(days=delta)

    if ((len(buy)> 0) and (buy.tail(1).index.item() > start_date) ):
        print("buy", buy.tail(1).index.item())
        buys.append(ticker)
    elif ((len(sell)> 0) and sell.tail(1).index.item() > start_date):
        print("sell", sell.tail(1).index.item())
        sells.append(ticker)
    else:
        holds.append(ticker)
        
def process(ticker, startdate, enddate, tf, flag='close', straSE='obv'):
    df = yf.download(ticker, 
                      start = startdate, 
                      end = enddate, 
                      interval=tf).fillna(0)
    # print(df.columns)
    df.columns = df.columns.str.lower()
    # print(df[['open','high','low','close']])
    # hammer pattern
    # res = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
    # res_ =talib.CDLHANGINGMAN(df['open'], df['high'], df['low'], df['close'])

    #
    #3candle
    #
    # res = talib.CDL3INSIDE(df['open'], df['high'], df['low'], df['close'])

    #
    # obv
    #
    if (straSE == 'CDL3O'):
        res = talib.CDL3OUTSIDE(df['open'], df['high'], df['low'], df['close'])
        entries =  res == 100
        exits = res == -100
    else: #(straSE == 'obv'
        obv = talib.OBV(df[flag], df['volume'])
        obv_ema = talib.EMA(obv, timeperiod=3)
        entries =  obv > obv_ema
        exits = obv < obv_ema
        
    pf = vbt.Portfolio.from_signals(df[flag], entries, exits)
    
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