import argparse

from pytz import timezone
import pytz

import asyncio
# import itertools
from datetime import datetime

# from IPython import display

import numpy as np
import pandas as pd
# import pandas_ta as ta
import vectorbt as vbt

import plotly.graph_objects as go
import sys
sys.path.append('examples')

from examples.utils import *
from examples.myInds import *

# def mark_ticker(ticker, holds, buys, sells, entries, exits, tf):
#     buy = entries[entries]
#     sell = exits[exits]
#     delta = int(tf[:-1])
#     interval = tf[-1]
#     end_date = datetime.now(timezone.utc)
#     if (interval == 'm'):
#         start_date = end_date - timedelta(minutes=delta)
#     elif (interval == 'h'):
#         start_date = end_date - timedelta(hours=delta)
#     else:
#         start_date = end_date - timedelta(days=delta)

#     if ((len(buy)> 0) and (buy.tail(1).index.item() > start_date) ):
#         print("buy", buy.tail(1).index.item())
#         buys.append(ticker)
#     elif ((len(sell)> 0) and sell.tail(1).index.item() > start_date):
#         print("sell", sell.tail(1).index.item())
#         sells.append(ticker)
#     else:
#         holds.append(ticker)
        
# def process(ticker, startdate, enddate, tf, flag='close', straSE='obv'):
#     df = yf.download(ticker, 
#                       start = startdate, 
#                       end = enddate, 
#                       interval=tf).fillna(0)
#     # print(df.columns)
#     df.columns = df.columns.str.lower()
#     # print(df[['open','high','low','close']])
#     # hammer pattern
#     # res = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
#     # res_ =talib.CDLHANGINGMAN(df['open'], df['high'], df['low'], df['close'])

#     #
#     #3candle
#     #
#     # res = talib.CDL3INSIDE(df['open'], df['high'], df['low'], df['close'])

#     #
#     # obv
#     #
#     if (straSE == 'CDL3O'):
#         res = talib.CDL3OUTSIDE(df['open'], df['high'], df['low'], df['close'])
#         entries =  res == 100
#         exits = res == -100
#     else: #(straSE == 'obv'
#         obv = talib.OBV(df[flag], df['volume'])
#         obv_ema = talib.EMA(obv, timeperiod=3)
#         entries =  obv > obv_ema
#         exits = obv < obv_ema
        
#     pf = vbt.Portfolio.from_signals(df[flag], entries, exits)
    
#     return pf, entries, exits

# # RSI = vbt.IndicatorFactory.from_talib('RSI')
# window_delta = 10
# def custom_indicator_obv(close, volumn, rsi_window, ma_window_slow,  entry, exit):
#     close_5m = close.resample("5T").last()
#     print("close_tm",close_5m)
# #     rsi = vbt.RSI.run(close,  window = rsi_window).rsi
#     obv = vbt.OBV.run(close,  window = rsi_window).rsi
# #     print("rsi", rsi)
#     ma_window_fast = ma_window_slow - window_delta
#     ma_slow = vbt.MA.run(close, ma_window_slow).ma.to_numpy()
#     ma_fast = vbt.MA.run(close, ma_window_fast).ma.to_numpy()
#     trend = np.where(rsi > exit , -1, 0)
#     trend = np.where( (rsi < entry) & (close < ma), 1, trend)
#     return trend

# def custom_indicator(close, rsi_window, ma_window_slow,  entry, exit):
# #     close_5m = close.resample("5T").last()
# #     print("close_tm",close_5m)
#     rsi = vbt.RSI.run(close,  window = rsi_window).rsi
# #     obv = vbt.OBV.run(close,  window = rsi_window).rsi
# #     print("rsi", rsi)
#     ma_window_fast = ma_window_slow - window_delta
#     ma_slow = vbt.MA.run(close, ma_window_slow).ma.to_numpy()
#     ma_fast = vbt.MA.run(close, ma_window_fast).ma.to_numpy()    
#     trend = np.where((rsi > exit) & (ma_fast < ma_slow), -1, 0)
#     trend = np.where( (rsi < entry) & ( ma_fast > ma_slow), 1, trend)
#     return trend

# def labeling_indicator(close, delta):
#     shift_delta = delta
#     shift_delta2 = delta *2
#     shift_delta3 = delta *3
#     value_delta = 1.5

#     exits = (((close > close.shift(-shift_delta)) & \
#             (close > close.shift(shift_delta))) & \
#             ((close > close.shift(-shift_delta2)) & \
#             (close > close.shift(shift_delta2)) & \
#             (close > close.shift(shift_delta3)) & \
#             (close > close.shift(-shift_delta3) )
#             ))

#     entries = (((close < close.shift(-shift_delta)) & \
#             (close < close.shift(shift_delta))) & \
#             ((close < close.shift(shift_delta2)) & \
#             (close < close.shift(-shift_delta2)) & \
#             (close < close.shift(shift_delta3)) & \
#             (close < close.shift(-shift_delta3))
#             ))     
#     trend = np.where(exits, -1, 0)
#     trend = np.where(entries, 1, trend)
#     return trend

# # RSI = vbt.IndicatorFactory.from_talib('RSI')
# window_delta = 10
# def custom_indicator_obv(close, volumn, rsi_window, ma_window_slow,  entry, exit):
#     close_5m = close.resample("5T").last()
#     print("close_tm",close_5m)
# #     rsi = vbt.RSI.run(close,  window = rsi_window).rsi
#     obv = vbt.OBV.run(close,  window = rsi_window).rsi
# #     print("rsi", rsi)
#     ma_window_fast = ma_window_slow - window_delta
#     ma_slow = vbt.MA.run(close, ma_window_slow).ma.to_numpy()
#     ma_fast = vbt.MA.run(close, ma_window_fast).ma.to_numpy()
#     trend = np.where(rsi > exit , -1, 0)
#     trend = np.where( (rsi < entry) & (close < ma), 1, trend)
#     return trend

# def custom_indicator(close, rsi_window, ma_window_slow,  entry, exit):
# #     close_5m = close.resample("5T").last()
# #     print("close_tm",close_5m)
#     rsi = vbt.RSI.run(close,  window = rsi_window).rsi
# #     obv = vbt.OBV.run(close,  window = rsi_window).rsi
# #     print("rsi", rsi)
#     ma_window_fast = ma_window_slow - window_delta
#     ma_slow = vbt.MA.run(close, ma_window_slow).ma.to_numpy()
#     ma_fast = vbt.MA.run(close, ma_window_fast).ma.to_numpy()    
#     trend = np.where((rsi > exit) & (ma_fast < ma_slow), -1, 0)
#     trend = np.where( (rsi < entry) & ( ma_fast > ma_slow), 1, trend)
#     return trend

# def custom_indicator_1m(close, rsi_window = 14,ma_window_slow = 20,  entry = 30, exit = 70):
#     close_5m = close.resample("5T").last()
# #     print(close_5m)
#     rsi = vbt.RSI.run(close_5m,  window = rsi_window).rsi
#     rsi, _= rsi.align(close, 
#                       broadcast_axis = 0,
#                      method = 'ffill',
#                      join = 'right')
#     rsi = rsi.to_numpy()
#     close = close.to_numpy()
#     ma_window_fast = ma_window_slow - window_delta
#     ma_slow = vbt.MA.run(close, ma_window_slow).ma.to_numpy()
#     ma_fast = vbt.MA.run(close, ma_window_fast).ma.to_numpy()
#     trend = np.where(rsi > exit , -1, 0)
#     trend = np.where( (rsi < entry) & (close < ma), 1, trend)
#     return trend

# def jinsong_indicator(rk50, rk100, rk250, rkprice,   entry, exit):
#     trend = np.where((((rk50 + rk100)*100/150) >= exit) & (rk100 >= exit), -1, 0)
#     trend = np.where( (((rk50 + rk100)*100/150) <= entry) & ((rk100 <= entry)| (rk250 == 1)), 1, trend)
#     return trend

# def jinsong_indicator_doublecheck(rk50, rk100, rk250, rkprice, deltaclose,  entry, exit):
#     trend = np.where((((rk50 + rk100)/150*100)>= exit) & (rk100 >= exit) & (deltaclose<0), -1, 0)
#     trend = np.where( (((rk50 + rk100)/150*100)<= entry) & (rk100 <= entry)& (deltaclose>0), 1, trend)
#     return trend

# # def cross3sma_indicator(Close, sma5, sma10, sma20, sma50,  entry, exit):
# #     trend = np.where((((rk50 + rk100)/150*100)>= exit) & (rk100 >= exit) & (deltaclose<0), -1, 0)
# #     trend = np.where( (((rk50 + rk100)/150*100)<= entry) & (rk100 <= entry)& (deltaclose>0), 1, trend)
# #     return trend

# def sma_indicator(Day, cross):
#     trend = np.where((Day > 10) & (cross < 0) , -1, 0)
#     trend = np.where((Day > 10) & (cross > 0), 1, trend)
#     return trend

# def sma_indicator(Day, cross):
#     trend = np.where((Day > 10) & (cross < 0) , -1, 0)
#     trend = np.where((Day > 10) & (cross > 0), 1, trend)
#     return trend

# def sept_indicator(Month, Day):
#     trend = np.where((Month == 9) & (Day < 10) , -1, 0)
#     trend = np.where((Day > 10) & (Month == 9), 1, trend)
#     return trend

# sept_ind = vbt.IndicatorFactory(
#     class_name = "BnH_Sept",
#     short_name = "Sept",
#     input_names = ['Month', 'Day'],
#     output_names = ["value"]
#     ).from_apply_func(
#          sept_indicator,
#          keep_pd = True
#         )

# sma_ind = vbt.IndicatorFactory(
#     class_name = "Simple_SMA",
#     short_name = "SSMA",
#     input_names = ['Day', 'cross'],
#     output_names = ["value"]
#     ).from_apply_func(
#          sma_indicator,
#          keep_pd = True
#         )

# jc_ind = vbt.IndicatorFactory(
#     class_name = "Jinsong",
#     short_name = "js",
#     input_names = ['rk50', 'rk100', 'rk250', 'rkprice'],
#     param_names = ["entry", 
#                    "exit"],
#     output_names = ["value"]
#     ).from_apply_func(
#          jinsong_indicator,
#          entry = 1,
#          exit = 100,         
#          keep_pd = True
#         )

# def trends(df: pd.DataFrame, mamode: str = "sma", fast: int = 50, slow: int = 200):
#     return ta.ma(mamode, df.close, length=fast) > ta.ma(mamode, df.close, length=slow) # SMA(fast) > SMA(slow) "Golden/Death Cross"
# #     return ta.increasing(ta.ma(mamode, df.close, length=fast)) # Increasing MA(fast)
# #     return ta.macd(df.close, fast, slow).iloc[:,1] > 0 # MACD Histogram is positive

# def trends_rsi(df: pd.DataFrame, rsi_window: int = 14, ma_window: int = 50, entry: int = 30, exit: int = 70):
#     res = ind.run(df.Close,
#                  rsi_window = rsi_window, 
#                  ma_window_slow = ma_window,
#                  entry = entry,
#                  exit = exit,
#                  param_product = True)
#     entries = res.value == 1.0
#     exits = res.value == -1.0
# #     trend = np.where(exits , -1, 0)
# #     trend = np.where( entries, 1, trend)    
# #     return  trend # rsi
#     return entries,exits

# def trends_sma(df: pd.DataFrame,  entry: int = 1, exit: int = 100):
#     res = sma_ind.run(df.Day, df.cross,
#                  param_product = True)
#     entries = res.value == 1.0
#     exits = res.value == -1.0
#     return entries, exits

# def trends_sept(df: pd.DataFrame,  entry: int = 1, exit: int = 100):
#     res = sept_ind.run(df.Month, df.Day,
#                  param_product = True)
#     entries = res.value == 1.0
#     exits = res.value == -1.0
#     return entries, exits

# def trends_jinsong(df: pd.DataFrame,  entry: int = 1, exit: int = 100):
#     res = jc_ind.run(df.rk50, df.rk100, df.rk250, df.rkprice,
#                  entry = entry,
#                  exit = exit,
#                  param_product = True)
#     entries = res.value == 1.0
#     exits = res.value == -1.0
#     return entries, exits

# def trends_jinsong_delta(df: pd.DataFrame,  entry: int = 1, exit: int = 98):
#     res = jc_ind_delta.run(df.rk50, df.rk100, df.rk250, df.rkprice,df.deltaclose,
#                  entry = entry,
#                  exit = exit,
#                  param_product = True)
#     entries = res.value == 1.0
#     exits = res.value == -1.0
#     return entries, exits

# def trade_table(pf: vbt.portfolio.base.Portfolio, k: int = 1, total_fees: bool = False):
#     if not isinstance(pf, vbt.portfolio.base.Portfolio): return
#     k = int(k) if isinstance(k, int) and k > 0 else 1

#     df = pf.trades.records[["status", "direction", "size", "entry_price", "exit_price", "return", "pnl"]]
#     if total_fees:
#         df["total_fees"] = df["entry_fees"] + df["exit_fees"]
# #     df.to_excel("trade_udow_2016to2023_exits.xlsx")
#     print(f"\nLast {k} of {df.shape[0]} Trades\n{df.tail(k)}\n")

       
# def combine_stats(pf: vbt.portfolio.base.Portfolio, ticker: str, strategy: str, mode: int = 0):
#     header = pd.Series({
#         "Run Time": ta.get_time(full=False, to_string=True),
#         "Mode": "LIVE" if mode else "TEST",
#         "Strategy": strategy,
#         "Direction": vbt.settings.portfolio["signal_direction"],
#         "Symbol": ticker.upper(),
#         "Fees [%]": 100 * vbt.settings.portfolio["fees"],
#         "Slippage [%]": 100 * vbt.settings.portfolio["slippage"],
#         "Accumulate": vbt.settings.portfolio["accumulate"],
#     })
#     rstats = pf.returns_stats().dropna(axis=0).T
#     stats = pf.stats().dropna(axis=0).T
#     joint = pd.concat([header, stats, rstats])
#     return joint[~joint.index.duplicated(keep="first")]

# %matplotlib inline
LIVE = 0
common_range = True

def show_trend(df, df_ori, ta_name, stratfile, bnhfile, resultfile, crs):
    if ta_name == "JC" :
        trend_jinsong={"entry": 12, "exit": 99}
        asset_trends, asset_exits = trends_jinsong(df, **trend_jinsong)
        strat_name = "Jinsong"
    elif ta_name == "SMA":
        asset_trends, asset_exits = trends_sma(df)
        strat_name = "SMA monthly trade"
    elif ta_name == "SEPT":
        asset_trends, asset_exits = trends_sept(df)
        strat_name = "Ou_September Sell"
    else:
        print("Not supported strategy")
        return 

    ##########################################################
    #
    #  Target Strategy
    #
    #########################################################
    
    asset_trends.copy().astype(int).plot(figsize=(16, 1), kind="area", color=["green"], alpha=0.45, title=f"{df.name} Trends", grid=True)    
    # Asset Portfolio from Trade Signals
    assetpf_signals = vbt.Portfolio.from_signals(
        df.Close,
        signal_args=(vbt.Rep("entries"), vbt.Rep("exits")),
        entries=asset_trends,
        exits=asset_exits,    
    #     entries=asset_signals.TS_Entries,
    #     exits=asset_signals.TS_Exits,
    )
    # print(assetpf_signals)
    trade_table(assetpf_signals, k=5)
    combine_stats(assetpf_signals, df.name, strat_name, LIVE)
    end_value: float = assetpf_signals.stats()['End Value']
    total_return: float = assetpf_signals.stats()['Total Return [%]']
    total_trades: int = assetpf_signals.stats()['Total Trades']
       
    assetpf_signals.plot (
        subplots = [
            'trades',
            'drawdowns',
            'cash',
            'value',
        ]
    ).write_image(stratfile)
    
    ##########################################################
    #
    #  Buy and Hold
    #
    #########################################################
    assetpf_bnh = vbt.Portfolio.from_holding(df_ori.Close)
    trade_table(assetpf_bnh, k=5)
    combine_stats(assetpf_bnh, df.name, strat_name, LIVE)
    
    bnh_value: float = assetpf_bnh.stats()['End Value']
    bnh_return: float = assetpf_bnh.stats()['Total Return [%]']
    assetpf_bnh.plot (
        subplots = [
            'trades',
            'drawdowns',
            'value',
        ]
    ).write_image(bnhfile)
    
    with open(resultfile, "w") as f:
        f.write( "=" * 50 + "\n")
        f.write(f"Analysis of: {df.name}{crs if common_range else ''}\n")
        f.write(f"             {strat_name}        Buy-n-Hold\n")
        f.write( "=" * 50 + "\n")    
        f.write(f"total_return: {round(total_return,2)},               {round(bnh_return,2)}\n")
        f.write(f"total_value:   {round(end_value, 2)}              {round(bnh_value,2)}\n")
        f.write( "=" * 50 + "\n")
    
def run(ticker, ta_name):
    cheight, cwidth = 500, 1000 # Adjust as needed for Chart Height and Width
    vbt.settings.set_theme("dark") # Options: "light" (Default), "dark" (my fav), "seaborn"

    # Must be set
    vbt.settings.portfolio["freq"] = "1d" # Daily

    # Predefine vectorbt Portfolio settings
    # vbt.settings.portfolio["init_cash"] = 100
    vbt.settings.portfolio["fees"] = 0.00 # 0.25%
    vbt.settings.portfolio["slippage"] = 0.00 # 0.25%
    # vbt.settings.portfolio["size"] = 100
    # vbt.settings.portfolio["accumulate"] = False
    vbt.settings.portfolio["allow_partial"] = False

    pf_settings = pd.DataFrame(vbt.settings.portfolio.items(), columns=["Option", "Value"])
    pf_settings.set_index("Option", inplace=True)

    print(f"Portfolio Settings [Initial]")
    pf_settings

    tf = '1d'
    assets = retrieve_data([ticker], tf=tf)

    start_date = datetime(2010, 1, 1) # Adjust as needed
    end_date = datetime(2023, 10, 4)   # Adjust as needed

    mbegin = pd.date_range(start_date, end_date, freq='BMS').strftime('%Y-%m-%d')
    mend = pd.date_range(start_date, end_date, freq='BM').strftime('%Y-%m-%d')

    datadir = '/home/steven/av_data/trades_analysis'
    startdate=datetime.now()
    # new_tz = pytz.timezone('US/Eastern')
    # startdate = startdate.astimezone(timezone('US/Pacific'))

    tadate = startdate.strftime('%Y-%m-%d')
    tadir =os.path.join(datadir, 'ta-'+ ta_name)
    if not os.path.exists(tadir):
        os.makedirs(tadir)    
    filename=os.path.join(tadir, 'chart-'+ticker+'.jpg')
    bnh_filename=os.path.join(tadir, 'chart-'+ticker+'-bnh.jpg')
    result_filename=os.path.join(tadir, ticker+'-perf.txt')
    common_range = True
    LIVE = 0

    df_ori = assets.data[ticker]
    if common_range:
        crs = f" from {start_date} to {end_date}"
        df_ori = dtmask(df_ori, start_date, end_date)

    if ta_name == "JC":
        assetdf = add_jc_data(df_ori)
    elif ta_name == "SMA":
        assetdf = dtmonth(df_ori, start= mbegin, end = mend)
        check_cross(assetdf, sma = 200)
    elif ta_name == "SEPT":
        assetdf = dtmonth(df_ori, start= mbegin, end = mend)
    else:
        assetdf = []

    if not assetdf.empty:
        assetdf.name = ticker
        print(f"Analysis of: {assetdf.name}{crs if common_range else ''}")
        show_trend(df=assetdf, df_ori=df_ori, ta_name = ta_name, stratfile=filename, bnhfile=bnh_filename, resultfile=result_filename, crs=crs)
    else:
        print("Dataframe is empty")

def parse():
    parser = argparse.ArgumentParser(description='stock data collection')
    parser.add_argument('--ticker', '-t', metavar='ticker', default='SPY',
                        help='stock ticker')     
#     parser.add_argument('--missing', default=False, action='store_true')    
#     parser.add_argument('--dir', '-d', metavar='DIR', default='dataset',
#                         help='path(s) to dataset (if one path is provided, it is assumed\n' +
#                        'to have subdirectories named "train" and "val"; alternatively,\n' +
#                        'train and val paths can be specified directly by providing both paths as arguments)')
#     parser.add_argument('--list', '-l', metavar='LIST', default='all',
#                         help='list of stock groups: sp500, dow or all') 
#     parser.add_argument('--date', '-dt', metavar='DATE', type=mkdate, default=datetime.now(),
#                         help='which date is to retrieve') 
# # 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo                      
    parser.add_argument('--strategy', '-s', metavar='strat', default='SMA',
                        help='trade strategy name')                       
#     parser.add_argument('--option-period', '-p', default='24', type=int,
#                         metavar='N', help='option period (default: 24)')
    args = parser.parse_args()
    return args

def main():
    args = parse()
    run(ticker = args.ticker.upper(), ta_name = args.strategy)


if __name__ == '__main__':
    main()