import numpy as np
import pandas as pd
import pandas_ta as ta
import vectorbt as vbt
import os
import yfinance as yf
# import talib
from datetime import datetime, timedelta, timezone

#helper functions
def combine_stats(pf: vbt.portfolio.base.Portfolio, ticker: str, strategy: str, mode: int = 0):
    header = pd.Series({
        "Run Time": ta.get_time(full=False, to_string=True),
        "Mode": "LIVE" if mode else "TEST",
        "Strategy": strategy,
        "Direction": vbt.settings.portfolio["signal_direction"],
        "Symbol": ticker.upper(),
        "Fees [%]": 100 * vbt.settings.portfolio["fees"],
        "Slippage [%]": 100 * vbt.settings.portfolio["slippage"],
        "Accumulate": vbt.settings.portfolio["accumulate"],
    })
    rstats = pf.returns_stats().dropna(axis=0).T
    stats = pf.stats().dropna(axis=0).T
    joint = pd.concat([header, stats, rstats])
    return joint[~joint.index.duplicated(keep="first")]

def earliest_common_index(d: dict):
    """Returns index of the earliest common index of all DataFrames in the dict"""
    min_date = None
    for df in d.values():
        if min_date is None:
            min_date = df.index[0]
        elif min_date < df.index[0]:
            min_date = df.index[0]
    return min_date

def dl(tickers: list, same_start: bool = False, **kwargs):
    if isinstance(tickers, str):
        tickers = [tickers]
    
    if not isinstance(tickers, list) or len(tickers) == 0:
        print("Must be a non-empty list of tickers or symbols")
        return

    if "limit" in kwargs and kwargs["limit"] and len(tickers) > kwargs["limit"]:
        from itertools import islice            
        tickers = list(islice(tickers, kwargs["limit"]))
        print(f"[!] Too many assets to compare. Using the first {kwargs['limit']}: {', '.join(tickers)}")

    print(f"[i] Downloading: {', '.join(tickers)}")

    received = {}
    if len(tickers):
        _df = pd.DataFrame()
        for ticker in tickers:
            received[ticker] = _df.ta.ticker(ticker, **kwargs)
            print(f"[+] {ticker}{received[ticker].shape} {ta.get_time(full=False, to_string=True)}")
    
    if same_start and len(tickers) > 1:
        earliestci = earliest_common_index(received)
        print(f"[i] Earliest Common Date: {earliestci}")
        result = {ticker:df[df.index > earliestci].copy() for ticker,df in received.items()}
    else:
        result = received
    print(f"[*] Download Complete\n")
    return result

def dtmask(df: pd.DataFrame, start: datetime, end: datetime):
    df['Datetime'] = pd.to_datetime(df.index, utc=True)
    print(df['Datetime'])
    if not df.ta.datetime_ordered:
        df = df.set_index(pd.DatetimeIndex(df['Datetime']))    
    return df.loc[(df.index >= pd.to_datetime(start, utc=True)) & (df.index <= pd.to_datetime(end, utc=True)), :].copy()

def show_data(d: dict):
    [print(f"{t}[{df.index[0]} - {df.index[-1]}]: {df.shape} {df.ta.time_range:.2f} years") for t,df in d.items()]
    
def trade_table(pf: vbt.portfolio.base.Portfolio, k: int = 1, total_fees: bool = False):
    if not isinstance(pf, vbt.portfolio.base.Portfolio): return
    k = int(k) if isinstance(k, int) and k > 0 else 1

    df = pf.trades.records[["status", "direction", "size", "entry_price", "exit_price", "return", "pnl", "entry_fees", "exit_fees"]]
    if total_fees:
        df["total_fees"] = df["entry_fees"] + df["exit_fees"]
#     df.to_excel("trade_udow_2016to2023_exits.xlsx")
    print(f"\nLast {k} of {df.shape[0]} Trades\n{df.tail(k)}\n")
    
####################################################################################################
#
#  Indicators or strategies
#
###################################################################################################
def jinsong_indicator(rk50, rk100, rk250, rkprice,   entry, exit):
    trend = np.where((((rk50 + rk100)*100/150) >= exit) & (rk100 >= exit), -1, 0)
    trend = np.where( (((rk50 + rk100)*100/150) <= entry) & ((rk100 <= entry)| (rk250 == 1)), 1, trend)
    return trend

def jinsong_indicator_doublecheck(rk50, rk100, rk250, rkprice, deltaclose,  entry, exit):
    trend = np.where((((rk50 + rk100)/150*100)>= exit) & (rk100 >= exit) & (deltaclose<0), -1, 0)
    trend = np.where( (((rk50 + rk100)/150*100)<= entry) & (rk100 <= entry)& (deltaclose>0), 1, trend)
    
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

jc_ind_delta = vbt.IndicatorFactory(
    class_name = "Jinsong-delta",
    short_name = "js-delta",
    input_names = ['rk50', 'rk100', 'rk250', 'rkprice','deltaclose'],
    param_names = ["entry", 
                   "exit"],
    output_names = ["value"]
    ).from_apply_func(
         jinsong_indicator_doublecheck,
         entry = 1,
         exit = 100,         
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

# rwindow = 14
# mawindow = 50
# ent = 30
# ext = 70
rwindow = np.arange(20,40, step=5, dtype=int)
mawindow_slow = np.arange(30,200, step=10, dtype=int)
# mawindow_fast = np.arange(10,80, step=20, dtype=int)
# if (mawindow_fast > mawindow_slow):
#     mawinow_fast = mawindow_slow - 20
ent = np.arange(20,40, step=5, dtype=int)
ext = np.arange(60,80, step=10, dtype=int)

def trends(df: pd.DataFrame, mamode: str = "sma", fast: int = 50, slow: int = 200):
    return ta.ma(mamode, df.close, length=fast) > ta.ma(mamode, df.close, length=slow) # SMA(fast) > SMA(slow) "Golden/Death Cross"
#     return ta.increasing(ta.ma(mamode, df.close, length=fast)) # Increasing MA(fast)
#     return ta.macd(df.close, fast, slow).iloc[:,1] > 0 # MACD Histogram is positive

def trends_rsi(df: pd.DataFrame, rsi_window: int = 14, ma_window: int = 50, entry: int = 30, exit: int = 70):
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