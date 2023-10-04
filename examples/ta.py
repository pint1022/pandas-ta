from pytz import timezone
import pytz

from plotly.subplots import make_subplots
import asyncio
import itertools
from datetime import datetime

# from IPython import display

import numpy as np
import pandas as pd
import pandas_ta as ta
import vectorbt as vbt

import plotly.graph_objects as go

from utils import *
from myInds import *

%matplotlib inline

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


ticker = 'SPY'
assets = retrieve_data([ticker], tf=tf)

start_date = datetime(2016, 12, 5) # Adjust as needed
end_date = datetime(2023, 9, 29)   # Adjust as needed
datadir = '/Users/steve/av_data/trades_analysis'
startdate=datetime.now()
startdate = startdate.astimezone(timezone('US/Pacific'))
tadate = startdate.strftime('%Y-%m-%d')
tadir =os.path.join(datadir, 'ta-'+tadate)
if not os.path.exists(tadir):
    os.makedirs(tadir)    
filename=os.path.join(tadir, 'chart-'+ticker+'.jpg')
bnh_filename=os.path.join(tadir, 'chart-'+ticker+'-bnh.jpg')
    
assetdf = assets.data[ticker]
assetdf = add_jc_data(assetdf)
common_range = True
LIVE = 0
trend_jinsong={"entry": 12, "exit": 99}

if common_range:
    crs = f" from {start_date} to {end_date}"
    assetdf = dtmask(assetdf, start_date, end_date)

# Update DataFrame names
assetdf.name = ticker
print(f"Analysis of: {assetdf.name}{crs if common_range else ''}")


def show_trend(df):
    asset_trends, asset_exits = trends_jinsong(df, **trend_jinsong)
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
    combine_stats(assetpf_signals, df.name, "Long Strategy", LIVE)
       
    assetpf_signals.plot (
        subplots = [
            'trades',
            'drawdowns',
            'cash',
            'value',
        ]
    ).write_image(filename)
    assetpf_bnh = vbt.Portfolio.from_holding(df.Close)
    assetpf_bnh.plot (
        subplots = [
            'trades',
            'drawdowns',
            'value',
        ]
    ).write_image(bnh_filename)
#     ta_fig = make_subplots(rows=2, cols=1,
#         shared_xaxes=True,
#         vertical_spacing=0.05,
#         subplot_titles=(ticker,),
#         row_heights = [500,200])
#     ta_fig.layout.xaxis.type = 'category'
#     ta_fig.add_trace(assetpf_signals.trades.plot(title=f"{df.name} | Trades", height=cheight, width=cwidth),
#                      row=1, col=1)
#     ta_fig.add_trace(assetpf_signals.cash().vbt.plot(title=f"{benchmarkdf.name} | Cash", trace_kwargs=dict(name=u"\u00A4"), height=cheight // 2, width=cwidth),
#                      row=2, col=1)
#     assetpf_signals.trades.plot(title=f"{df.name} | Trades", height=cheight, width=cwidth).write_image(filename)
#     ta_fig.write_image(filename)
show_trend(df=assetdf)
