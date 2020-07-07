'''
Created on 13 feb. 2020

@author: stan
'''
# Load the Pandas libraries with alias 'pd' 
import pandas as pd 
import numpy as np
import ta
import talib
from ta.momentum import StochasticOscillator
from datetime import datetime
from openpyxl import workbook #For export to Excel
import time
from sklearn.linear_model import LogisticRegression

# Configuration
LOAD_CSV = "inputs/bitfinex_tBTCUSD_1m.csv"
CREATE_TIMESTAMP = False
CREATE_INDICATORS = True
TYPE_MACHINE_LEARNING = "Classification" #Classification or Regression
TYPE_REGRESSION = "Difference" #Difference or Percentage or Target (value of tomorrow) Only applies to TYPE_MACHINE_LEARNING = 'Regression'
INTERVAL_PERIOD = 1 #How much intervals do we need to look in the future? 1 = next
EXPORT = "CSV" #Options: EXCEL , CSV , NONE
EXPORT_NAME_EXCEL = "output_TestData.xlsx"
EXPORT_NAME_CSV = "outputs/23062020_Classification_1MIN_ALLTI.csv"

# Read data from file 'filename.csv' 
# (in the same directory that your python process is based)
# Control delimiters, rows, column names with read_csv (see later) 
print('Loading CSV...')
start_time = time.time()
df = pd.read_csv(LOAD_CSV, encoding = "ISO-8859-1") 
# Clean NaN values
df = ta.utils.dropna(df)
df.index.names = ['index']
end_time = time.time()
print('Time to load and prepare CSV: ' + str(end_time - start_time) + ' seconds')


# Calculate timestamp based on format Date + Time = month/day/year hour:min:sec
# Overwrite 'Time' column with the timestamp
# Delete 'Date' column
# Rename 'Time' column as 'Timestamp'
if CREATE_TIMESTAMP:
    print('Generating Timestamp Column...')
    start_time = time.time()
    df['Time'] = df.apply (lambda row: datetime.timestamp(datetime.strptime(str(row.Date + ' ' + row.Time), '%m/%d/%Y %H:%M:%S')), axis=1) #axis 1 = apply function to each row
    del df['Date']
    #Rename Time column to Timestamp column
    df = df.rename({'Time': 'Timestamp'}, axis=1)
    end_time = time.time()
    print('Time to create Timestamp: ' + str(end_time - start_time) + ' seconds')


# Start initializing technical indicators
if CREATE_INDICATORS:
    print('Calculating Technical Indicators...')
    start_time_allTI = time.time()
    # Simple Moving Average (SMA) 
    '''
    Description:
    iloc is primarily integer position based (from 0 to length-1 of the axis), but may also be used with a boolean array.
    '''
    period = 30
    df['SMA_Close'] = talib.SMA(df["Close"], timeperiod=period)
    print('Time #1 batch TI: ' + str(time.time() - start_time) + ' seconds')
    start_time = time.time()
    # Stochastic Oscillator (SO)
    period = 14
    sma_period = 3
    StochasticOscillator = ta.momentum.StochasticOscillator(high=df["High"], low=df["Low"], close=df["Close"], n=period, d_n=sma_period, fillna=False)
    df['SO'] = StochasticOscillator.stoch()
    
    # Momentum (M)
    period = 3
    df['Momentum'] = talib.MOM(df["Close"], timeperiod=period)


    # Price Rate Of Change (ROC)
    '''
    Description:
    is a pure momentum oscillator that measures the percent change in price from one period to the next
    The ROC calculation compares the current price with the price “n” periods ago
    '''
    period = 12
    RateOfChange = ta.momentum.ROCIndicator(close=df["Close"], n=period, fillna=False)
    df['ROC'] = RateOfChange.roc()
    
    # Williams %R
    '''
    Description:
    Williams %R reflects the level of the close relative to the highest high for the look-back period
    Williams %R oscillates from 0 to -100.
    Readings from 0 to -20 are considered overbought. Readings from -80 to -100 are considered oversold.
    '''
    lookback_period = 14
    WilliamsR = ta.momentum.WilliamsRIndicator(high=df["High"], low=df["Low"], close=df["Close"], lbp=lookback_period, fillna=False)
    df['WR'] = WilliamsR.wr()
    
    # Weighted Closing Price (WCP)
    df['WCP'] = talib.WCLPRICE(df["High"], df["Low"], df["Close"])
    
    # Williams Accumulation Distribution Line
    # AKA Accumulation/Distribution Index (ADI)????
    '''
    Description:
    a volume-based indicator designed to measure the cumulative flow of money into and out of a security
    The Accumulation Distribution Line rises when the multiplier is positive and falls when the multiplier is negative.
    '''
    ADI = ta.volume.AccDistIndexIndicator(high=df["High"], low=df["Low"], close=df["Close"], volume=df["Volume"], fillna=False)
    df['ADI'] = ADI.acc_dist_index()


    # Moving Average Convergence Divergence (MACD)
    '''
    Description:
    Is a trend-following momentum indicator that shows the relationship between two moving averages of prices.
    '''
    period_longterm = 26
    period_shortterm = 12
    period_to_signal = 9
    MACD = ta.trend.MACD(close=df["Close"], n_slow=period_longterm, n_fast=period_shortterm, n_sign=period_to_signal, fillna=False)
    df['MACD'] = MACD.macd()

    print('Time #2 batch TI: ' + str(time.time() - start_time) + ' seconds')
    start_time = time.time()
    # Commodity Channel Index (CCI)
    '''
    Description:
    CCI measures the difference between a security’s price change and its average price change. 
    High positive readings indicate that prices are well above their average, which is a show of strength. 
    Low negative readings indicate that prices are well below their average, which is a show of weakness.
    '''
    periods = 20
    constant = 0.015
    #CCI = ta.trend.cci(high=df["High"], low=df["Low"], close=df["Close"], n=periods, c=constant, fillna=False)
    #df['CCI'] = CCI

    # Bollinger Bands (BB)
    '''
    Description:
    CCI measures the difference between a security’s price change and its average price change. 
    High positive readings indicate that prices are well above their average, which is a show of strength. 
    Low negative readings indicate that prices are well below their average, which is a show of weakness.
    '''
    periods = 20
    n_factor_standard_dev = 2
    indicator_bb = ta.volatility.BollingerBands(close=df["Close"], n=periods, ndev=n_factor_standard_dev, fillna=False)
    # Add Bollinger Bands features
    #df['bb_bbm'] = indicator_bb.bollinger_mavg()
    df['BB_H'] = indicator_bb.bollinger_hband()
    df['BB_L'] = indicator_bb.bollinger_lband()
    
    # Add Bollinger Band high indicator
    df['bb_bbhi'] = indicator_bb.bollinger_hband_indicator()
    
    # Add Bollinger Band low indicator
    df['bb_bbli'] = indicator_bb.bollinger_lband_indicator()
    
    # Add width size Bollinger Bands
    df['bb_bbw'] = indicator_bb.bollinger_wband()
    print('Time #3 batch TI: ' + str(time.time() - start_time) + ' seconds')
    start_time = time.time()
    # Mean Open & Close (M_O, M_C)
    period=3
    df['MEAN_O_C'] = (talib.SMA(df["Open"], timeperiod=period) / 2) + (talib.SMA(df["Close"], timeperiod=period) / 2)
    
    
    # Variance Open & Close
    df["VAR_Close"] = talib.VAR(df["Close"], timeperiod=5, nbdev=1)
    df["VAR_Open"] = talib.VAR(df["Open"], timeperiod=5, nbdev=1)
    
    
    # High Price Average
    '''
    Description:
    Simple moving average over the high
    '''
    period=3
    df['SMA_High'] = talib.SMA(df["High"], timeperiod=period)
    # Low Price Average
    '''
    Description:
    Simple moving average over the low
    '''
    period=3
    df['SMA_Low'] = talib.SMA(df["Low"], timeperiod=period)
    print('Time #4 batch TI: ' + str(time.time() - start_time) + ' seconds')
    start_time = time.time()
    # High, Low Average
    '''
    Description:
    Simple moving average over the sum of high and low
    '''
    period=3
    df['SMA_H+L'] = talib.SMA(df["High"] + df["Low"], timeperiod=period)
    
    
    # Trading Day Price Average
    '''
    Description:
    Simple moving average over the sum of the open, high, low and close
    '''
    period=3
    df['SMA_H+L+C+O'] = talib.SMA(df["High"] + df["Low"] + df["Open"] + df["Close"], timeperiod=period)
    
    # From here on adding random indicators according to the ta-lib library
    # ######################## OVERLAP STUDIES ############################
    # Double Exponential Moving Average
    period=30
    df['DEMA'] = talib.DEMA(df["Close"], timeperiod=period)
    # Exponential Moving Average
    period=30
    df['EMA'] = talib.EMA(df["Close"], timeperiod=period)
    # HT_TRENDLINE - Hilbert Transform - Instantaneous Trendline
    df['HT_TRENDLINE'] = talib.HT_TRENDLINE(df["Close"])
    # KAMA - Kaufman Adaptive Moving Average
    period=30
    df['KAMA'] = talib.KAMA(df["Close"], timeperiod=period)
    # MA - Moving average
    period=30
    print('Time #5 batch TI: ' + str(time.time() - start_time) + ' seconds')
    start_time = time.time()
    df['MA'] = talib.MA(df["Close"], timeperiod=period, matype=0)
    # MIDPOINT - MidPoint over period
    period=14
    df['MIDPOINT'] = talib.MIDPOINT(df["Close"], timeperiod=period)
    # MIDPRICE - Midpoint Price over period
    period=14
    df['MIDPOINT'] = talib.MIDPRICE(df["High"], df["Low"], timeperiod=period)
    # SAR - Parabolic SAR
    df['SAR'] = talib.SAR(df["High"], df["Low"], acceleration=0, maximum=0)
    # SAREXT - Parabolic SAR - Extended
    df['SAREXT'] = talib.SAREXT(df["High"], df["Low"], startvalue=0, offsetonreverse=0, accelerationinitlong=0, accelerationlong=0, accelerationmaxlong=0, accelerationinitshort=0, accelerationshort=0, accelerationmaxshort=0)
    # T3 - Triple Exponential Moving Average (T3)
    period=5
    df['T3'] = talib.T3(df["Close"], timeperiod=period, vfactor=0)
    # TEMA - Triple Exponential Moving Average
    period=30
    df['TEMA'] = talib.TEMA(df["Close"], timeperiod=period)
    # TRIMA - Triangular Moving Average
    period=30
    df['TRIMA'] = talib.TRIMA(df["Close"], timeperiod=period)
    # WMA - Weighted Moving Average
    period=30
    df['WMA'] = talib.WMA(df["Close"], timeperiod=period)
    
    # ######################## Momentum Indicators ############################
    # ADX - Average Directional Movement Index
    period=14
    df['ADX'] = talib.ADX(df["High"], df["Low"], df["Close"], timeperiod=period)
    # ADXR - Average Directional Movement Index Rating
    period=14
    df['ADXR'] = talib.ADXR(df["High"], df["Low"], df["Close"], timeperiod=period)
    print('Time #6 batch TI: ' + str(time.time() - start_time) + ' seconds')
    start_time = time.time()
    # APO - Absolute Price Oscillator
    df['APO'] = talib.APO(df["Close"], fastperiod=12, slowperiod=26, matype=0)
    # AROON - Aroon
    df['aroondown'], df['aroonup'] = talib.AROON(df["High"], df["Low"], timeperiod=14)
    # AROONOSC - Aroon Oscillator
    period=14
    df['AROONOSC'] = talib.AROONOSC(df["High"], df["Low"], timeperiod=14)
    # BOP - Balance Of Power
    period=14
    df['BOP'] = talib.BOP(df["Open"], df["High"], df["Low"], df["Close"])
    # CMO - Chande Momentum Oscillator
    df['CMO'] = talib.CMO(df["Close"], timeperiod=14)
    # DX - Directional Movement Index
    df['DX'] = talib.DX(df["High"], df["Low"], df["Close"], timeperiod=14)
    # MFI - Money Flow Index
    df['MFI'] = talib.MFI(df["High"], df["Low"], df["Close"], df["Volume"], timeperiod=14)
    # MINUS_DI - Minus Directional Indicator
    df['MINUS_DI'] = talib.MINUS_DI(df["High"], df["Low"], df["Close"], timeperiod=14)
    # MINUS_DM - Minus Directional Movement
    df['MINUS_DM'] = talib.MINUS_DM(df["High"], df["Low"], timeperiod=14)
    # PLUS_DI - Plus Directional Indicator
    df['PLUS_DI'] = talib.PLUS_DI(df["High"], df["Low"], df["Close"], timeperiod=14)
    # PLUS_DM - Plus Directional Movement
    df['PLUS_DM'] = talib.PLUS_DM(df["High"], df["Low"], timeperiod=14)
    print('Time #7 batch TI: ' + str(time.time() - start_time) + ' seconds')
    start_time = time.time()
    # PPO - Percentage Price Oscillator
    df['PPO'] = talib.PPO(df["Close"], fastperiod=12, slowperiod=26, matype=0)
    # ROCP - Rate of change Percentage: (price-prevPrice)/prevPrice
    df['ROCP'] = talib.ROCP(df["Close"], timeperiod=10)
    # ROCR - Rate of change ratio: (price/prevPrice)
    df['ROCR'] = talib.ROCR(df["Close"], timeperiod=10)
    # ROCR100 - Rate of change ratio 100 scale: (price/prevPrice)*100
    df['ROCR100'] = talib.ROCR100(df["Close"], timeperiod=10)
    # RSI - Relative Strength Index
    df['RSI'] = talib.RSI(df["Close"], timeperiod=14)
    # TRIX - 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA
    df['TRIX'] = talib.TRIX(df["Close"], timeperiod=30)
    # ULTOSC - Ultimate Oscillator
    df['ULTOSC'] = talib.ULTOSC(df["High"], df["Low"], df["Close"], timeperiod1=7, timeperiod2=14, timeperiod3=28)
    
    # ######################## Volume Indicators ############################
    # AD - Chaikin A/D Line
    df['AD'] = talib.AD(df["High"], df["Low"], df["Close"], df["Volume"])
    # ADOSC - Chaikin A/D Oscillator
    df['ADOSC'] = talib.ADOSC(df["High"], df["Low"], df["Close"], df["Volume"], fastperiod=3, slowperiod=10)
    # OBV - On Balance Volume
    df['OBV'] = talib.OBV(df["Close"], df["Volume"])
    print('Time #8 batch TI: ' + str(time.time() - start_time) + ' seconds')
    start_time = time.time()
    # ######################## Cycle Indicators ############################
    # HT_DCPERIOD - Hilbert Transform - Dominant Cycle Period
    df['HT_DCPERIOD'] = talib.HT_DCPERIOD(df["Close"])
    # HT_DCPHASE - Hilbert Transform - Dominant Cycle Phase

    #df['HT_DCPHASE'] = talib.HT_DCPHASE(df["Close"])

    # HT_TRENDMODE - Hilbert Transform - Trend vs Cycle Mode
    #df['HT_TRENDMODE'] = talib.HT_TRENDMODE(df["Close"])

    # ######################## Price transform functions ############################
    # AVGPRICE - Average Price
    df['AVGPRICE'] = talib.AVGPRICE(df["Open"], df["High"], df["Low"], df["Close"])
    # MEDPRICE - Median Price
    df['MEDPRICE'] = talib.MEDPRICE(df["High"], df["Low"])
    # TYPPRICE - Typical Price
    df['TYPPRICE'] = talib.TYPPRICE(df["High"], df["Low"], df["Close"])
    # WCLPRICE - Weighted Close Price
    df['WCLPRICE'] = talib.WCLPRICE(df["High"], df["Low"], df["Close"])

    print('Time #9 batch TI: ' + str(time.time() - start_time) + ' seconds')
    # ################################ END OF TECHINCAL INDICATORS #########################
    end_time = time.time()
    print('Time to create Technical Inidicators: ' + str(end_time - start_time_allTI) + ' seconds')

print('Creating Target...')
start_time = time.time()
if TYPE_MACHINE_LEARNING == "Classification":
    # Add new column 'Target' . True if Close price from this timestmap is higher than Close price of previous timestamp
    df['Target'] = np.where(df.Close.shift(-1) > df.Close, True, False)
elif TYPE_MACHINE_LEARNING == "Regression":
    if TYPE_REGRESSION == "Difference":
        df['Target'] = df.Close.shift(-INTERVAL_PERIOD) - df.Close
    elif TYPE_REGRESSION == "Percentage":
        dfClose = df['Close']
        df['Target'] = -dfClose.pct_change(periods=-INTERVAL_PERIOD)
    elif TYPE_REGRESSION == "Target":
        df['Target'] = df.Close.shift(-INTERVAL_PERIOD)
    
end_time = time.time()
print('Time to create Target: ' + str(end_time - start_time) + ' seconds')


# Print to Excel dataframe
print(df)
#exit()
if EXPORT == "EXCEL" or EXPORT == "CSV":
    print('Exporting Excel...')
    start_time = time.time()
    if EXPORT == "EXCEL":
        df.to_excel(EXPORT_NAME_EXCEL)
    if EXPORT == "CSV": 
        df.to_csv(EXPORT_NAME_CSV)
    end_time = time.time()
    print('Time to export excel/CSV: ' + str(end_time - start_time) + ' seconds')

