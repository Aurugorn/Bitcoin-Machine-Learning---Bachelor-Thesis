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
import time



# initialise object 
# pull data from start of 2017 to present day
from_date = "2013-04-28" 
to_date = "2020-03-21"
to_date_bitcoin = "2020-03-20"
from_date_cmp = time.strptime(from_date, "%Y-%m-%d")
to_date_cmp = time.strptime(to_date, "%Y-%m-%d")

lowest_date = ''
highest_date = ''

my_cryptory = Cryptory(from_date = from_date, to_date = to_date, timeout=100)

coin_col = False
coin = "bitcoin"

def extract_coinmarketcap(coin, coin_col=False):
    try:
        output = pd.read_html("https://coinmarketcap.com/currencies/{}/historical-data/?start={}&end={}".format(
            coin, from_date.replace("-", ""), to_date_bitcoin.replace("-", "")))[2]
      
    except:
        # future versions may split out the different exceptions (e.g. timeout)
        raise
    output = output.assign(Date=pd.to_datetime(output['Date']))
    outputHeader = output
    for col in outputHeader.columns:
        if outputHeader[col].dtype == np.dtype('O'):
            outputHeader.loc[outputHeader[col]=="-",col]=0
            outputHeader[col] = outputHeader[col].astype('int64')
    outputHeader.columns = [re.sub(r"[^a-z]", "", col.lower()) for col in outputHeader.columns]
    if coin_col:
        outputHeader['coin'] = coin
    
    return output

#info bitcoin
df = extract_coinmarketcap("bitcoin", coin_col=False)

# Set lowest date from bitcoin info
earliestDate = str(df['date'].tail(1).values[0]).split('T')[0]
earliestDate = time.strptime(earliestDate, "%Y-%m-%d")

if earliestDate > from_date_cmp:
    from_date_cmp = earliestDate

#Subreddit
RedditData_Req = my_cryptory.extract_reddit_metrics("bitcoin", "subscriber-growth-perc")
RedditData_Date = list(reversed(RedditData_Req["date"]))[0]
RedditData_Date = time.strptime(str(RedditData_Date).split(' ')[0], "%Y-%m-%d")
if RedditData_Date > from_date_cmp:
    from_date_cmp = RedditData_Date
    

RedditData = RedditData_Req["subscriber_growth_perc"]
'''
#Google trends
GoogleTrends_Req = my_cryptory.get_google_trends(kw_list=['bitcoin'])
GoogleTrends_Date = list(reversed(GoogleTrends_Req["date"]))[0]
GoogleTrends_Date = time.strptime(str(GoogleTrends_Date).split(' ')[0], "%Y-%m-%d %H:%M:%s")
if GoogleTrends_Date > from_date_cmp:
    from_date_cmp = GoogleTrends_Date

print(GoogleTrends_Date)
exit()
'''
print('From date: {} '.format(from_date_cmp))
print(len(df))

print(len(RedditData_Req))

df['RedditData'] = RedditData
print(df)
print(RedditData_Req)
dates = list(reversed(my_cryptory.extract_bitinfocharts("btc")["date"]))
priceBTC = list(reversed(my_cryptory.extract_bitinfocharts("btc")["btc_price"]))
