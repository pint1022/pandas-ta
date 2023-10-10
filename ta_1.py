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
    
def run(datadir, ticker, ta_name, start_date, end_date):
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
    elif ta_name == "OBV":
        assetdf = df_ori
    else:
        assetdf = pd.DataFrame() 

    if not assetdf.empty:
        assetdf.name = ticker
        print(f"Analysis of: {assetdf.name}{crs if common_range else ''}")
        show_trend(df=assetdf, df_ori=df_ori, ta_name = ta_name, stratfile=filename, bnhfile=bnh_filename, resultfile=result_filename, crs=crs)
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
# # 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo                      
    parser.add_argument('--strategy', '-s', metavar='strat', default='SMA',
                        help='trade strategy name')                       
#     parser.add_argument('--option-period', '-p', default='24', type=int,
#                         metavar='N', help='option period (default: 24)')
    args = parser.parse_args()
    return args

def main():
    args = parse()
    run(datadir=args.dir, ticker = args.ticker.upper(), ta_name = args.strategy, start_date=args.sdate, end_date=args.edate)


if __name__ == '__main__':
    main()