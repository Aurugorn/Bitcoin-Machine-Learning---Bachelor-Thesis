'''
Created on 4 mrt. 2020

@author: stan
'''
# load package
from cryptory import Cryptory
import datetime
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import re
import plotly.graph_objects as go


# initialise object 
# pull data from start of 2017 to present day
my_cryptory = Cryptory(from_date = "2015-12-31", to_date = "2020-03-21")

dates = list(reversed(my_cryptory.extract_bitinfocharts("btc")["date"]))
#print(dates)
priceBTC = list(reversed(my_cryptory.extract_bitinfocharts("btc")["btc_price"]))
#print(my_cryptory.get_google_trends(kw_list=['bitcoin']))
GoogleTrends = list(reversed(my_cryptory.get_google_trends(kw_list=['bitcoin'])["bitcoin"]))

#Subreddit
RedditData = my_cryptory.extract_reddit_metrics("bitcoin", "subscriber-growth-perc")["subscriber_growth_perc"]


#print("Length dates: {} ".format(len(dates)))
#print("Length priceBTC: {} ".format(len(priceBTC)))
#print("Length GoogleTrends: {} ".format(len(GoogleTrends)))
#Create dataframe
df = pd.DataFrame({'priceBTC':priceBTC})
df['GoogleTrends'] = GoogleTrends
df['RedditData'] = RedditData

df[['priceBTC','GoogleTrends','RedditData']] = (
        df[['priceBTC','GoogleTrends','RedditData']]-df[['priceBTC','GoogleTrends','RedditData']].min())/(
        df[['priceBTC','GoogleTrends','RedditData']].max()-df[['priceBTC','GoogleTrends','RedditData']].min())
        

fig = go.Figure()

# Line for portfolio
fig.add_trace(
    go.Scatter(x=dates, y=df['priceBTC'], name="BTC Price")
)

# Line for breakeven
fig.add_trace(
    go.Scatter(x=dates, y=df['GoogleTrends'], name="GoogleTrends")
)

# Reddit data
fig.add_trace(
    go.Scatter(x=dates, y=df['RedditData'], name="RedditData")
)

fig.update_layout(
    title = 'Correlation Google searches for BTC and BTC Price',
    xaxis_tickformat = '%d %B (%a)<br>%Y'
)


fig.show()
