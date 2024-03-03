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
import talib

import plotly.graph_objects as go
import sys
sys.path.append('examples')

from examples.utils import *
from examples.myInds import *

LIVE = 0
common_range = True

def obv_trigger(df, ta_name, stratfile, resultfile, crs):
    obv = talib.OBV(df['close'], df['volume'])
    obv_ema = talib.EMA(obv, timeperiod=3)
    # print(obv_ema)
    entries =  obv > obv_ema
    exits = obv < obv_ema

    pf = vbt.Portfolio.from_signals(df['open'], entries, exits)
    print(pf.stats())
    fig = pf.plot()
    fig.layout.xaxis.type = 'category'
    fig.layout.xaxis2.type = 'category'
    fig.layout.xaxis3.type = 'category'
    # fig.show()
    fig.write_image(stratfile)
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
    print( "=" * 50)
    print(f"Analysis of: {df.name}{crs if common_range else ''}")
    print(f"             {strat_name}        Buy-n-Hold")
    print( "=" * 50 + "")    
    print(f"total_return: {round(total_return,2)},               {round(bnh_return,2)}")
    print(f"total_value:   {round(end_value, 2)}              {round(bnh_value,2)}")
    print( "=" * 50)

def get_trend(df, df_ori, ta_name, stratfile, interval='15m', duration=60, start_date = None, end_date = None, period=5):
    if ta_name == "PSMA":
        vbt.settings.portfolio["fees"] = 0.01 # 0.25%
        strat_name = "Ou SMA cross buy"        
        asset_trends, asset_exits = trends_ou_sma(df, period = period, op = 1)         
    elif ta_name == "RPSMA":
        vbt.settings.portfolio["fees"] = 0.01 # 0.25%
        strat_name = "Ou SMA cross buy reverse"        
        asset_trends, asset_exits = trends_ou_sma(df, period = period, op = -1)         
    elif ta_name == "RSI":
        vbt.settings.portfolio["fees"] = 0.01 # 0.25%
        strat_name = "RSI"  
        trend_rsi={"rsi_window": 14, "ma_window": 50, "entry": 30, "exit": 60}
        asset_trends, asset_exits = trends_rsi(df, **trend_rsi)        
    elif ta_name == "GOLDEN_DEATH":
        vbt.settings.portfolio["fees"] = 0.01 # 0.25%
        strat_name = "RSI"  
        trend_kwargs = {"mamode": "sma", "fast": 50, "slow": 200}
        asset_trends, asset_exits = trends(df, **trend_kwargs)        
    elif ta_name == "OBV":
        obv = talib.OBV(df['Close'], df['Volume'])
        obv_ema = talib.EMA(obv, timeperiod=period)
        # print(obv_ema)
        asset_trends =  obv > obv_ema
        asset_exits = obv < obv_ema
        strat_name = "OBV-Open"
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
    if ta_name == "OBV":    
        assetpf_signals = vbt.Portfolio.from_signals(
            df.Open,
            entries=asset_trends,
            exits=asset_exits,    
        #     entries=asset_signals.TS_Entries,
        #     exits=asset_signals.TS_Exits,
        )
    else:
        assetpf_signals = vbt.Portfolio.from_signals(
            df.Close,
            entries=asset_trends,
            exits=asset_exits,    
        )
    # print(assetpf_signals)
    trade_table(assetpf_signals, k=5)
    combine_stats(assetpf_signals, df.name, strat_name, LIVE)
    
    assetpf_signals.plot (
        subplots = [
            'trades',
            'drawdowns',
            'cash',
            'value',
        ]
    ).write_image(stratfile)
        
    end_value: float = assetpf_signals.stats()['End Value']
    total_return: float = assetpf_signals.stats()['Total Return [%]']
    total_trades: int = assetpf_signals.stats()['Total Trades']
    sharpe: float = assetpf_signals.stats()['Sharpe Ratio']
        
    return total_return, end_value, sharpe       
    
def show_trend(df, df_ori, ta_name, stratfile, bnhfile, resultfile, crs, interval='15m', duration=60, start_date = None, end_date = None, period=5):
    if ta_name == "JC" :
        trend_jinsong={"entry": 12, "exit": 99}
        asset_trends, asset_exits = trends_jinsong(df, **trend_jinsong)
        strat_name = "Jinsong"
    elif ta_name == "PSMA":
        vbt.settings.portfolio["fees"] = 0.01 # 0.25%
        strat_name = "Ou SMA cross buy"        
        asset_trends, asset_exits = trends_ou_sma(df, period = period, op = 1)         
    elif ta_name == "RPSMA":
        vbt.settings.portfolio["fees"] = 0.01 # 0.25%
        strat_name = "Ou SMA cross buy reverse"        
        asset_trends, asset_exits = trends_ou_sma(df, period = period, op = -1)         
    elif ta_name == "SMA":
        asset_trends, asset_exits = trends_sma(df)
        strat_name = "SMA monthly trade"
    elif ta_name == "SEPT":
        asset_trends, asset_exits = trends_sept(df)
        strat_name = "Ou_September Sell"
    elif ta_name == "OBV":
        obv = talib.OBV(df['Close'], df['Volume'])
        obv_ema = talib.EMA(obv, timeperiod=3)
        # print(obv_ema)
        asset_trends =  obv > obv_ema
        asset_exits = obv < obv_ema
        strat_name = "OBV-Open"
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
    if ta_name == "OBV":    
        assetpf_signals = vbt.Portfolio.from_signals(
            df.Open,
            entries=asset_trends,
            exits=asset_exits,    
        #     entries=asset_signals.TS_Entries,
        #     exits=asset_signals.TS_Exits,
        )
    else:
        assetpf_signals = vbt.Portfolio.from_signals(
            df.Close,
            entries=asset_trends,
            exits=asset_exits,    
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
    print( "=" * 50)
    print(f"Analysis of: {df.name}{crs if common_range else ''}")
    print(f"             {strat_name}        Buy-n-Hold")
    print( "=" * 50 + "")    
    print(f"total_return: {round(total_return,2)},               {round(bnh_return,2)}")
    print(f"total_value:   {round(end_value, 2)}              {round(bnh_value,2)}")
    print( "=" * 50)

    with open(resultfile, "w") as f:
        f.write( "=" * 50 + "\n")
        f.write(f"Analysis of: {df.name}{crs if common_range else ''}\n")
        f.write(f"             {strat_name}        Buy-n-Hold\n")
        f.write( "=" * 50 + "\n")    
        f.write(f"total_return: {round(total_return,2)},               {round(bnh_return,2)}\n")
        f.write(f"total_value:   {round(end_value, 2)}              {round(bnh_value,2)}\n")
        f.write( "=" * 50 + "\n")
    
def run(datadir, ticker, ta_name, start_date, end_date, interval='1d', period=5, fee=0.01, duration=60):
    cheight, cwidth = 500, 1000 # Adjust as needed for Chart Height and Width
    vbt.settings.set_theme("dark") # Options: "light" (Default), "dark" (my fav), "seaborn"

    # Must be set
    vbt.settings.portfolio["freq"] = interval # Daily

    # Predefine vectorbt Portfolio settings
    # vbt.settings.portfolio["init_cash"] = 100
    vbt.settings.portfolio["fees"] = fee # 0.25%
    vbt.settings.portfolio["slippage"] = 0.00 # 0.25%
    # vbt.settings.portfolio["size"] = 100
    # vbt.settings.portfolio["accumulate"] = False
    vbt.settings.portfolio["allow_partial"] = False

    pf_settings = pd.DataFrame(vbt.settings.portfolio.items(), columns=["Option", "Value"])
    pf_settings.set_index("Option", inplace=True)

    print(f"Portfolio Settings [Initial]")
    print(pf_settings)

    tf = interval
    assets = retrieve_data([ticker], tf=tf, stratName = None)

    # start_date = datetime(2010, 1, 1) # Adjust as needed
    # end_date = datetime(2023, 10, 4)   # Adjust as needed

    mbegin = pd.date_range(start_date, end_date, freq='BMS').strftime('%Y-%m-%d')
    mend = pd.date_range(start_date, end_date, freq='BM').strftime('%Y-%m-%d')

    # datadir = '/home/steven/av_data/trades_analysis'
    # startdate=datetime.now()
    # new_tz = pytz.timezone('US/Eastern')
    # startdate = startdate.astimezone(timezone('US/Pacific'))

    # tadate = startdate.strftime('%Y-%m-%d')
    tadir =os.path.join(datadir, 'ta-'+ ta_name)
    if not os.path.exists(tadir):
        os.makedirs(tadir)    
    filename=os.path.join(tadir, 'chart-'+ticker+'.jpg')
    bnh_filename=os.path.join(tadir, 'chart-'+ticker+'-bnh.jpg')
    result_filename=os.path.join(tadir, ticker+'-perf.txt')
    if (interval[-1] == 'm') or (interval[-1] == 'h'):
        common_range = False
    else:
        common_range = True
    LIVE = 0

    df_ori = assets.data[ticker]
#     print(df_ori.columns)
    if common_range:
        crs = f" from {start_date} to {end_date}"
        df_ori = dtmask(df_ori, start_date, end_date)
    else:
        if (interval[-1] == 'm'):
            days = min(60, duration)
            end_date = datetime.now()- timedelta(days=1)            
            msg = 'Minute analysis'
        elif (interval[-1] == 'h'):
            days = min(720, duration)
            msg = 'Hourly analysis'
            end_date = datetime.now()- timedelta(days=1)            
        crs = f" {msg} from {start_date} to {end_date}"
        
    if ta_name == "JC":
        assetdf = add_jc_data(df_ori)
    elif ta_name == "SMA":
        assetdf = dtmonth(df_ori, start= mbegin, end = mend)
        check_cross(assetdf, sma = 200)
    elif ta_name == "SEPT":
        assetdf = dtmonth(df_ori, start= mbegin, end = mend)
    elif (ta_name == "PSMA") or (ta_name == "RPSMA"):
        assetdf = df_ori
    elif ta_name == "OBV":
        assetdf = df_ori
    else:
        assetdf = pd.DataFrame() 

    if not assetdf.empty:
        assetdf.name = ticker
        print(f"Analysis of: {assetdf.name}{crs if common_range else ''}")
        show_trend(df=assetdf, df_ori=df_ori, ta_name = ta_name, stratfile=filename, bnhfile=bnh_filename, resultfile=result_filename, crs=crs, interval=interval, duration=duration, start_date = None, end_date = None, period=period)
    else:
        print("Dataframe is empty")

def grun(datadir, ticker, ta_name, start_date, end_date, interval='1d', periods=[5], fee=0.01, duration=60):
    cheight, cwidth = 500, 1000 # Adjust as needed for Chart Height and Width
    vbt.settings.set_theme("dark") # Options: "light" (Default), "dark" (my fav), "seaborn"

    # Must be set
    vbt.settings.portfolio["freq"] = interval # Daily

    # Predefine vectorbt Portfolio settings
    # vbt.settings.portfolio["init_cash"] = 100
    vbt.settings.portfolio["fees"] = fee # 0.25%
    vbt.settings.portfolio["slippage"] = 0.00 # 0.25%
    # vbt.settings.portfolio["size"] = 100
    # vbt.settings.portfolio["accumulate"] = False
    vbt.settings.portfolio["allow_partial"] = False

    pf_settings = pd.DataFrame(vbt.settings.portfolio.items(), columns=["Option", "Value"])
    pf_settings.set_index("Option", inplace=True)

    print(f"Portfolio Settings [Initial]")
    print(pf_settings)

    tf = interval
    assets = retrieve_data([ticker], tf=tf, stratName = None)

    tadir =os.path.join(datadir, 'ta-'+ ta_name)
    if not os.path.exists(tadir):
        os.makedirs(tadir)    
    bnh_filename=os.path.join(tadir, 'chart-'+ticker+'-bnh.jpg')
    result_filename=os.path.join(tadir, ticker+'-perf.txt')
    if (interval[-1] == 'm') or (interval[-1] == 'h'):
        common_range = False
    else:
        common_range = True
    LIVE = 0

    df_ori = assets.data[ticker]
#     print(df_ori)
    if common_range:
        crs = f" from {start_date} to {end_date}"
        df_ori = dtmask(df_ori, start_date, end_date)
    else:
        if (interval[-1] == 'm'):
            days = min(60, duration)
            end_date = datetime.now()- timedelta(days=1)            
            msg = 'Minute analysis'
        elif (interval[-1] == 'h'):
            days = min(720, duration)
            msg = 'Hourly analysis'
            end_date = datetime.now()- timedelta(days=1)            
        crs = f" {msg} from {start_date} to {end_date}"
        
    if (ta_name == "PSMA"):
        straName = "SMA Cross"
        assetdf = df_ori
    elif (ta_name == "RPSMA"):
        straName = "Reverse SMA Cross"
        assetdf = df_ori
    elif ta_name == "OBV":
        assetdf = df_ori
    else:
        assetdf = pd.DataFrame() 

    if not assetdf.empty:
        assetdf.name = ticker
        list_returns = []
        print(f"Analysis of: {assetdf.name}{crs if common_range else ''}")
        for i in periods:
            filename=os.path.join(tadir, 'chart-SMA-'+str(i)+'-'+ticker+'.jpg')
            
            #
            # simulate the trades
            #
            treturn, evalue, sharpe = get_trend(df=assetdf, df_ori=df_ori, ta_name = ta_name, stratfile=filename, interval=interval, duration=duration, start_date = None, end_date = None, period=i)
            list_returns.append(['SMA'+str(i), treturn, evalue, sharpe])
            
        assetpf_bnh = vbt.Portfolio.from_holding(assetdf.Close)
        assetpf_bnh.plot (
            subplots = [
                'trades',
                'drawdowns',
                'value',
            ]
        ).write_image(bnh_filename)
        trade_table(assetpf_bnh, k=5)
        
        bnh_value: float = assetpf_bnh.stats()['End Value']
        bnh_return: float = assetpf_bnh.stats()['Total Return [%]']
        bnh_sharpe: float = assetpf_bnh.stats()['Sharpe Ratio']
        list_returns.append(['BuyNHold', bnh_return, bnh_value, bnh_sharpe])
        num_eq = 60
        
        print( "=" * num_eq)
        print(f"{straName} analysis of: {assetdf.name}{crs if common_range else ''}")
        print(f"Strategy          Total Return(%)       End Value        Sharpe Ratio(%)")
        print( "=" * num_eq + "")    
        for items in list_returns:
            print('%10s  %14s          %12s     %12s'%(items[0],round(items[1],2),round(items[2],2),round(items[3],2)))

        print( "=" * num_eq)
        
        with open(result_filename, "w") as f:
            f.write( "=" * num_eq)
            f.write(f"{straName} analysis of: {ta_name} of {assetdf.name}{crs if common_range else ''}")
            f.write(f"Period        Total Return(%)       End Value          Sharpe Ratio(%)")
            f.write( "=" * num_eq + "\n")    
            for items in list_returns:
                f.write('%10s  %14s          %12s     %12s'%(items[0],round(items[1],2),round(items[2],2),round(items[3],2)))
           
            f.write( "=" * num_eq)
            
#         print(list_returns)
    else:
        print("Dataframe is empty")        
        
def sgrun(datadir, ticker,  start_date, end_date, ta_name=['OBV'], interval='1d', period=5, fee=0.01, duration=60):
    cheight, cwidth = 500, 1000 # Adjust as needed for Chart Height and Width
    vbt.settings.set_theme("dark") # Options: "light" (Default), "dark" (my fav), "seaborn"

    # Must be set
    vbt.settings.portfolio["freq"] = interval # Daily

    # Predefine vectorbt Portfolio settings
    # vbt.settings.portfolio["init_cash"] = 100
    vbt.settings.portfolio["fees"] = fee # 0.25%
    vbt.settings.portfolio["slippage"] = 0.00 # 0.25%
    # vbt.settings.portfolio["size"] = 100
    # vbt.settings.portfolio["accumulate"] = False
    vbt.settings.portfolio["allow_partial"] = False

    pf_settings = pd.DataFrame(vbt.settings.portfolio.items(), columns=["Option", "Value"])
    pf_settings.set_index("Option", inplace=True)

    print(f"Portfolio Settings [Initial]")
    print(pf_settings)

    tf = interval
    assets = retrieve_data([ticker], tf=tf, stratName = None)

    tadir =os.path.join(datadir, 'ta-groups')
    if not os.path.exists(tadir):
        os.makedirs(tadir)    
    bnh_filename=os.path.join(tadir, 'chart-'+ticker+'-bnh.jpg')
    result_filename=os.path.join(tadir, ticker+'-perf-strategies.txt')
    if (interval[-1] == 'm') or (interval[-1] == 'h'):
        common_range = False
    else:
        common_range = True
    LIVE = 0

    df_ori = assets.data[ticker]
#     print(df_ori)
    if common_range:
        crs = f" from {start_date} to {end_date}"
        df_ori = dtmask(df_ori, start_date, end_date)
    else:
        if (interval[-1] == 'm'):
            days = min(60, duration)
            end_date = datetime.now()- timedelta(days=1)            
            msg = 'Minute analysis'
        elif (interval[-1] == 'h'):
            days = min(720, duration)
            msg = 'Hourly analysis'
            end_date = datetime.now()- timedelta(days=1)            
        crs = f" {msg} from {start_date} to {end_date}"
        
    assetdf = df_ori

    if not assetdf.empty:
        assetdf.name = ticker
        list_returns = []
        print(f"Analysis of: {assetdf.name}{crs if common_range else ''}")
        
        for ta in ta_name:
            filename=os.path.join(tadir, 'chart-'+ta+'-'+ticker+'.jpg')
            treturn, evalue, sharpe = get_trend(df=assetdf, df_ori=df_ori, ta_name = ta, stratfile=filename, interval=interval, duration=duration, start_date = None, end_date = None, period=period)
            list_returns.append([ta, treturn, evalue, sharpe])
            
        assetpf_bnh = vbt.Portfolio.from_holding(assetdf.Close)
        assetpf_bnh.plot (
            subplots = [
                'trades',
                'drawdowns',
                'value',
            ]
        ).write_image(bnh_filename)
        trade_table(assetpf_bnh, k=5)
        
        bnh_value: float = assetpf_bnh.stats()['End Value']
        bnh_return: float = assetpf_bnh.stats()['Total Return [%]']
        bnh_sharpe: float = assetpf_bnh.stats()['Sharpe Ratio']
            
        list_returns.append(['BuyNHold', bnh_return, bnh_value, bnh_sharpe])
        num_eq = 60
        
        print( "=" * num_eq)
        print(f"The analysis of: {assetdf.name}{crs if common_range else ''}")
        print(f"Strategy          Total Return(%)       End Value        Sharpe Ratio(%)")
        print( "=" * num_eq + "")    
        for items in list_returns:
            print('%10s  %14s          %12s     %12s'%(items[0],round(items[1],2),round(items[2],2),round(items[3],2)))
            
#             print(f" {items[0]}               {round(items[1],2)}               {round(items[2],2)}            {round(items[3],2)}")

        print( "=" * num_eq)
        
        with open(result_filename, "w") as f:
            f.write( "=" * num_eq)
            f.write(f"The analysis of: {assetdf.name}{crs if common_range else ''}")
            f.write(f"Period        Total Return(%)       End Value          Sharpe Ratio(%)")
            f.write( "=" * num_eq + "\n")    
            for items in list_returns:
                f.write('%10s  %14s          %12s     %12s'%(items[0],round(items[1],2),round(items[2],2),round(items[3],2)))
                
            f.write( "=" * num_eq)
            
#         print(list_returns)
    else:
        print("Dataframe is empty")
        
def mkdate(datestr):
    return datetime.strptime(datestr, '%Y-%m-%d')

def parse():
    parser = argparse.ArgumentParser(description='stock data collection')
    parser.add_argument('--ticker', '-t', metavar='ticker', default='SPY',
                        help='stock ticker')     
#     parser.add_argument('--missing', default=False, action='store_true')    
    parser.add_argument('--dir', '-d', metavar='DIR', default='/home/steven/av_data/trades_analysis',
                        help='path(s) to dataset (if one path is provided, it is assumed\n' +
                       'to have subdirectories named "train" and "val"; alternatively,\n' +
                       'train and val paths can be specified directly by providing both paths as arguments)')
#     parser.add_argument('--list', '-l', metavar='LIST', default='all',
#                         help='list of stock groups: sp500, dow or all') 
    parser.add_argument('--sdate', '-sdt', metavar='START DATE', type=mkdate, default=datetime.now()- timedelta(days=1*365),
                        help='what is the start date of the period') 
    parser.add_argument('--edate', '-edt', metavar='END DATE', type=mkdate, default=datetime.now(),
                        help='what is the end date of the period') 
    parser.add_argument('--duration', '-u', metavar='the duration of trades', type=mkdate, default=datetime.now(),
                        help='how many days of trades are backtested from today')     
    parser.add_argument('--interval', '-i', metavar='interval of data point',  default = '1d',
                        help='data point interval') # 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo             
    parser.add_argument('--periods', '-pds', metavar='SMA periods', default=None,
                        help='a list of SMA period') # 
    parser.add_argument('--period', '-p', metavar='SMA period', type=int, default=5, help='SMA period') # 
    parser.add_argument('--strategy', '-s', metavar='strat', default='SMA',
                        help='trade strategy name')                       
    parser.add_argument('--ta', '-ss', metavar='stratlist', default='OBV', help='list of trade strategy name')                       
    parser.add_argument('--fee', '-e', metavar='fee of trade', default=0.01,
                        help='trade fee')                       
#     parser.add_argument('--option-period', '-p', default='24', type=int,
#                         metavar='N', help='option period (default: 24)')
    args = parser.parse_args()
    return args

def main():
    args = parse()
    if args.periods is not None:
        periods = [int(item) for item in args.periods.split(',')]
        grun(datadir=args.dir, ticker = args.ticker.upper(), ta_name = args.strategy, start_date=args.sdate, end_date=args.edate, interval=args.interval, periods=periods, fee=args.fee, duration=args.duration)
    elif args.ta is not None:
        strategies = [item for item in args.ta.split(',')]    
        sgrun(datadir=args.dir, ticker = args.ticker.upper(), ta_name = strategies, start_date=args.sdate, end_date=args.edate, interval=args.interval, period=args.period, fee=args.fee, duration=args.duration)
    else:
        strategies = None  
        run(datadir=args.dir, ticker = args.ticker.upper(), ta_name = args.strategy, start_date=args.sdate, end_date=args.edate, interval=args.interval, period=period, fee=args.fee, duration=args.duration)
      

if __name__ == '__main__':
    main()