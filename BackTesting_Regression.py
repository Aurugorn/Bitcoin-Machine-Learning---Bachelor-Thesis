'''
Created on 1 mrt. 2020

@author: stan
'''

# Load the Pandas libraries with alias 'pd' 
import pandas as pd 
import numpy as np
from datetime import datetime, timedelta
from openpyxl import workbook #For export to Excel
import time
import ta
import matplotlib.pyplot as plt
import plotly.graph_objects as go
#new imports
from cryptory import Cryptory

import locale

locale.setlocale( locale.LC_ALL, '' )
# Read data from file 'filename.csv' 
# (in the same directory that your python process is based)
# Control delimiters, rows, column names with read_csv (see later) 
# Portfolio axes

y_back_0 = []
strat0_name = 'Portfolio: Regression, Str=1, Strictness=1'
y_back_1 = []
strat1_name = 'Portfolio: Regression, Str=1, Strictness=10'
y_back_2 = []
strat2_name = 'Portfolio: Regression, Str=2, Strictness=10'

for backTestingCycle in range(0, 3):
    if backTestingCycle == 0:
        df = pd.read_csv("modelOutputsRegression/PredictedModel_RF_ALLTI_1MIN_DIFF_n10.csv") 
    #df = df[0:10000]
    # 1min data currently available until: presentDate = "2020-02-05"
    # 1hour data currently available until: presentDate = "2020-04-08"
    # 1day data currently available until: presentDate = today - 2 days
    presentDate = "2020-02-05"
    
    MinMaxScaling = False
    takerFee = 0.0175 #Buyers fee
    #takerFee = 0
    makerFee = 0 #Seller fee
    startingPortfolio = 10000
    EUROS = startingPortfolio
    BITCOIN = 0
    lastBoughtPrice = 0
    # Iterate on every row
    '''
    if Predicted = True: Go All In OR HOLD
    if Predicted = False: Sell all OR HOLD
    '''
    amountOfRows = len(df)
    percCompleted = 0    
    showMessage = True    
    amountTimesBought = 0
    amountTimesSold = 0
    previousDate = ''
    highestBITCOINValue = 0
    highestBV_Date = ''
    lowestBITCOINValue = startingPortfolio
    lowestBV_Date = ''
    ACTION = 'HOLD'
    x_axis = []
    y_axis = []
    # Set following parameters for strategy:
    TYPE_MACHINE_LEARNING = "Regression" # Regression or Classification
    STRATEGY = 1 # 1 = Buy and Sell if model says so
                    # 2 = Buy if model says so. And sell only when bitcoin value + transaction costs > last bought price
                    # 3 = ORACLE strategy --> 100% Accuracy prediction
    ADDITIONAL_STRAT = "BuyThreshold" # BuyThreshold: Buy only if regression model predicts that bitcoin will grow more than transaction fee of buying                
                                      # NoStrat : Nothing
    # ONLY FOR REGRESSION AND ADDITIONAL_STRAT == "BuyThreshold" AND STRATEGY = 1 OR 2
    MODEL_STRICTNESS = 10 # 1 = 100% Follow predicted value --> Very Strict
                          # 10 = 10% Follow predicted value --> Low Strict
                          # 100 = 1% Follow predicted value --> We don't trust the predicted value at all.
    #backTestingCycle=0: Regression, Strategy = 1, ADDITIONAL_STRAT = "BuyThreshold", Model_strictness = 1
    #backTestingCycle=1: Regression, Strategy = 1, ADDITIONAL_STRAT = "BuyThreshold", Model_strictness = 10
    #backTestingCycle=2: Regression, Strategy = 2, ADDITIONAL_STRAT = "BuyThreshold", Model_strictness = 10 
    if backTestingCycle == 0:
        # Set following parameters for strategy:
        TYPE_MACHINE_LEARNING = "Regression" 
        STRATEGY = 1 
        ADDITIONAL_STRAT = "BuyThreshold" 
        MODEL_STRICTNESS = 1
    elif backTestingCycle == 1:
        # Set following parameters for strategy:
        TYPE_MACHINE_LEARNING = "Regression" 
        STRATEGY = 1 
        ADDITIONAL_STRAT = "BuyThreshold" 
        MODEL_STRICTNESS = 10 
    elif backTestingCycle == 2:
        # Set following parameters for strategy:
        TYPE_MACHINE_LEARNING = "Regression" 
        STRATEGY = 2 
        ADDITIONAL_STRAT = "BuyThreshold" 
        MODEL_STRICTNESS = 10 
                          
    '''
    --------------------- Combinations of strategies ---------------------
    1. Classification
        a. STRATEGY = 1 (ALWAYS Buy & Sell if model predicts respectively True / False)
        b. STRATEGY = 2 (ALWAYS Buy if model predicts True. ONLY Sell if you made profit on your last bought value. (Current BITCOIN value + Buyer Transaction Fee must be > LastBoughtPrice))
        c. STRATEGY = 3 (Oracle Strategy: Buy & Sell according to Target value (100% accuracy))
        
    2. Regression
        a. STRATEGY = 1, ADDITIONAL_STRAT = "NoStrat" (ALWAYS Buy & Sell if model predicts respectively > 0 / < 0)
        b. STRATEGY = 1, ADDITIONAL_STRAT = "BuyThreshold" (Same as (a) **AND** BUY Only if model predicts price change > buying transaction fee )
            1b. MODEL_STRICTNESS = 1 (Listen to model predicted value without; ex. = +0.120 %. (This is > Buyer Transaction Fee (0.0175, so would buy)))
            2b. MODEL_STRICTNESS = 10 (ex. Predicted = +0.120 %. (This is > Buyer Transaction Fee (0.0175, so would buy)), BUT With applying strictness; Predicted = +0.0120. (This is < Buyer Transaction Fee, so do not BUY)
        c. STRATEGY = 2, ADDITIONAL_STRAT = "NoStrat" (ALWAYS Buy if model predicts > 0. ONLY Sell if you made profit on your last bought value. (Current BITCOIN value + Buyer Transaction Fee must be > LastBoughtPrice))
        d. STRATEGY = 2, ADDITIONAL_STRAT = "BuyThreshold" (Same as (c) **AND** BUY Only if model predicts price change > buying transaction fee
            1d. MODEL_STRICTNESS = 1
            2d. MODEL_STRICTNESS = 10
        e. STRATEGY = 3, ADDITIONAL_STRAT = "BuyThreshold" (Oracle Strategy: Buy & Sell according to Target value (100% accuracy))
    '''
    
    print('We are going to backtest with the following strategy:')
    print('Backtest machine learning type: {}'.format(TYPE_MACHINE_LEARNING))
    if STRATEGY == 1:
        STRATEGY_PRINT = 'ALWAYS Buy and Sell if models says so using Predicted value of model.'
    elif STRATEGY == 2:
        STRATEGY_PRINT = 'Buy if model predicts it is going to raise. Sell ONLY when you made profit, that is when your current BITCOIN value + Buying Transaction Fee > LastBoughtPrice.'
    elif STRATEGY == 3:
        STRATEGY_PRINT = 'Oracle strategy. Buy and Sell using Target value of model (100% Accuracy)'
    print('Strategy is to {}'.format(STRATEGY_PRINT))
    if ADDITIONAL_STRAT != "NoStrat":
        if ADDITIONAL_STRAT == "BuyThreshold":
            print('Extra: Buy only if regression model predicts that BITCOIN in the next interval will grow more than the buying transaction fee ({}%)'.format(takerFee))
    '''
    Back-Testing explanation:
    _____________________________
    Iterate on every row
        If you have some value in BITCOIN, calculate the profit/loss based on the df['Change'] column (contains percentage change of this interval vs previous interval)
        If Model Predicts TRUE --> Model thinks price will increase in next interval:
            Remember the price you have bought the BITCOIN for BEFORE transaction fees (lastBoughtPrice (ex. 10.000))
            BUY only if you don't own any BITCOIN.
            CONVERT all your EUROS to BITCOIN value including transaction fee (takerFee = buyer Fee) (ex. 10.000 * 0,99 = 99.900)
            Now you own BITCOIN worth of 99.900 EUR
        If Model Predicts FALSE --> Model thinks price will decrease in next interval:
            SELL only if you made some profit. Meaning your BITCOIN value + transaction fees is higher than the last bought price 
            If your BITCOIN value (ex. 10.050) + transaction fees (10.050 * 0,99 = 99.949) > lastBoughtPrice (ex. 10.000)
                YES: SELL all your BITCOIN to EURO (You've made some profit)
                NO: Just HOLD your BITCOIN until it ever reaches this value
    
    This way, we only BUY BITCOIN if our model says so and we only SELL if our model says so AND we are sure that we've made some profit based on the transaction fee 
    if we every decide to buy again.
    If we do not take the transaction fee into account whenever our model wants to sell, then it is possible we sell our BITCOIN when it has a value lower than we bought
    the BITCOIN for. And whenever we ever want to BUY the BITCOIN again, we make loss since we always pay the transaction fee.
    '''
    for i, j in df.iterrows(): 
        # If STRATEGY = 3 : Oracle strategy, always predict what it was in reality
        if STRATEGY == 3:
            modelPredicts = j.Target
            MODEL_STRICTNESS = 1
        else:
            modelPredicts = j.Predicted
       
        if BITCOIN > 0:
            BITCOIN = BITCOIN * (1 + j.Change) #if j.Change = -0.3% == -0.003 == 0.997 * BITCOIN (1 - 0.003)
             #Highest and lowest value of bitcoin
            if BITCOIN > highestBITCOINValue:
                highestBITCOINValue = BITCOIN
                highestBV_Date = datetime.fromtimestamp(j.Timestamp).strftime("%Y-%m-%d %H:%M")
            if BITCOIN < lowestBITCOINValue:
                lowestBITCOINValue = BITCOIN
                lowestBV_Date = datetime.fromtimestamp(j.Timestamp).strftime("%Y-%m-%d %H:%M")
        #If we are backtesting regression & we predict price goes up and it will raise higher than the transaction fee, buy it
        #If backtesting classification & predicted Yes it will raise, and we haven't invested all into BITCOIN YET, DO IT. Else HOLD
        if TYPE_MACHINE_LEARNING == "Classification" or ADDITIONAL_STRAT == "NoStrat":
            buyThreshold = 0
            modelPredictsBUY = modelPredicts #No percentage, just normal difference
        elif ADDITIONAL_STRAT == "BuyThreshold":
            buyThreshold = takerFee #Fee in percentages
            modelPredictsBUY =  (((j.Close + modelPredicts) * 100) / j.Close) - 100 # Model predict change in percentage
        # BUYING BITCOIN ACCORDING TO CONDITIONS    
        if ((TYPE_MACHINE_LEARNING == "Regression" and modelPredictsBUY/MODEL_STRICTNESS > buyThreshold) or (TYPE_MACHINE_LEARNING == "Classification" and modelPredicts == True)) and BITCOIN == 0: 
            #print('Modelpredicts {} > BuyThreshold {}'.format(modelPredictsPerc, buyThreshold))
            lastBoughtPrice = EUROS # Price you bought BITCOIN at EXCLUDED transaction fees
            
            BITCOIN = EUROS * (1-(takerFee/100))
            EUROS = 0
            amountTimesBought += 1
            ACTION = 'BUY'
          
        # SELLING BITCOIN ACCORDING TO CONDITIONS
        elif ((TYPE_MACHINE_LEARNING == "Regression" and modelPredicts < 0) or (TYPE_MACHINE_LEARNING == "Classification" and modelPredicts == False)) and EUROS == 0: #Sell all for euros
            if (STRATEGY == 2 and (BITCOIN * (1-(takerFee/100)) > lastBoughtPrice)) or (STRATEGY == 1) or (STRATEGY == 3):
                EUROS = BITCOIN * (1-(makerFee/100))
                BITCOIN = 0
                amountTimesSold += 1
                ACTION = 'SELL'
              
        else:
            ACTION = 'HOLD'
            
       
        #To View progress of loops
        percCompleted = round((100*i)/amountOfRows, 0)
        if percCompleted % 10 == 0 and percCompleted > 0 and showMessage == True:   
            print('Completed {}% of the rows'.format(percCompleted))
            showMessage = False
        if percCompleted % 11 == 0 and showMessage == False:   
            showMessage = True  
        
        dateTime_string = datetime.fromtimestamp(j.Timestamp).strftime("%Y-%m-%d")
        if dateTime_string != previousDate:
            x_axis.append(dateTime_string)
            if BITCOIN == 0:
                y_axis.append(lastBoughtPrice)
                print("{} (EUR): {}".format(dateTime_string, locale.currency(lastBoughtPrice, grouping=True)))  
            else:
                y_axis.append(BITCOIN)
                print("{} (BTC): {}".format(dateTime_string, locale.currency(BITCOIN, grouping=True)))  
            
            
        previousDate = dateTime_string
    #Now sell all to EUROS if BITCOIN > 0
    if BITCOIN > 0:
        EUROS = BITCOIN
        BITCOIN = 0
    
    #Copy portfolio axes to correct backtesting cycle axis
    if backTestingCycle == 0:
        y_back_0 = y_axis
    elif backTestingCycle == 1:
        y_back_1 = y_axis
    elif backTestingCycle == 2:
        y_back_2 = y_axis
    
    print('----------- TOTAL PROFIT/LOSS -----------')
    print('Started portfolio in EUR: {}'.format(locale.currency(startingPortfolio, grouping=True)))
    print('Bitcoin in EUR: €0')
    print('Over {} iterations, our model bought {} times and sold {} times'.format(amountOfRows, amountTimesBought, amountTimesSold))
    print('Lowest value of Bitcoin was on {} value: {}'.format(lowestBV_Date, locale.currency(lowestBITCOINValue, grouping=True)))
    print('Highest value of Bitcoin was on {} value: {}'.format(highestBV_Date, locale.currency(highestBITCOINValue, grouping=True)))
    print('Ended portfolio in EUR: {}'.format(locale.currency(EUROS, grouping=True)))
    print('% Profit/Loss: {}%'.format(round(((EUROS - startingPortfolio) / startingPortfolio) * 100), 2))


#Plotting results 
y_axis_breakeven = []
for i in range(0, len(y_axis)):
    y_axis_breakeven.append(startingPortfolio)
    
#date_1 = datetime.strptime(x_axis[0], "%Y-%m-%d")
#modifiedDate = date_1 + timedelta(days=1)
modifiedDate = x_axis[0]

my_cryptory = Cryptory(from_date = modifiedDate, to_date = presentDate) # First element of x_axis = start date of data
priceBTC = list(reversed(my_cryptory.extract_bitinfocharts("btc")["btc_price"]))
print(my_cryptory.extract_bitinfocharts("btc"))
print("Length Portfolio: {} ".format(len(y_axis)))
print("Length BTC Pric: {} ".format(len(priceBTC)))
print("Length Break-even: {} ".format(len(y_axis_breakeven)))
print("Start date: {} ".format(x_axis[0]))
print("Start date modified +1: {} ".format(modifiedDate))

#Create dataframe
df = pd.DataFrame({'Portfolio':y_axis})
df['BTC Price'] = priceBTC
df['Break-even'] = y_axis_breakeven

if MinMaxScaling:
    df[['Portfolio','BTC Price', 'Break-even']] = (
            df[['Portfolio','BTC Price', 'Break-even']]-df[['Portfolio','BTC Price', 'Break-even']].min())/(
            df[['Portfolio','BTC Price', 'Break-even']].max()-df[['Portfolio','BTC Price', 'Break-even']].min())
     

fig = go.Figure()

# Line for portfolio
fig.add_trace(
    go.Scatter(x=x_axis, y=y_back_0, name=strat0_name)
)

# Line for portfolio
fig.add_trace(
    go.Scatter(x=x_axis, y=y_back_1, name=strat1_name)
)


# Line for portfolio
fig.add_trace(
    go.Scatter(x=x_axis, y=y_back_2, name=strat2_name)
)


# Line for breakeven
fig.add_trace(
    go.Scatter(x=x_axis, y=df['Break-even'], name="Break-even")
)
# Line for BTC price
fig.add_trace(
    go.Scatter(x=x_axis, y=df['BTC Price'], name="BTC Price")
)



fig.update_layout(
    title = 'BTC/EUR Portfolio value over time',
    xaxis_tickformat = '%d %B (%a)<br>%Y'
)


fig.show()
exit()

# Plot the graph
plt.plot(x_axis, y_axis)
plt.yticks(np.arange(min(y), max(y)+1, 1.0))
plt.xlabel('Date')
plt.ylabel('Portfolio in €')
plt.title('BTC/EUR Portfolio value over time')
plt.show()
    