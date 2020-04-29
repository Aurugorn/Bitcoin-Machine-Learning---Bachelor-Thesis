# Authors: Stan van der Avoird <stan@restica.nl>
#

import numbers
import warnings

import pandas as pd
import ta
import talib
import numpy as np
import numpy.ma as ma
from scipy import sparse
from scipy import stats
import time
from datetime import datetime

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.sparsefuncs import _get_median
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import FLOAT_DTYPES
from sklearn.utils._mask import _get_mask
from sklearn.utils import is_scalar_nan
from sklearn.utils import check_array
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

#------------------------------------- BitcoinTransformer component -------------------------------------#
class BitcoinTransformer(BaseEstimator, TransformerMixin):
    """Bitcoin Transformer to prepare dataset for machine learning
        Requirements
        __________
        pip install pandas
        pip install numpy
        pip install TA-Lib (*)
        pip install --upgrade ta
        ##pip install openpyxl
        pip install -U scikit-learn
        sudo apt-get install python3-matplotlib
        pip install beautifulsoup4
        pip install pytrends
        pip install cryptory
        pip install auto-sklearn

        (*) TA-LIB dependency:
        Download ta-lib-0.4.0-src.tar.gz http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz and:

        $ tar -xzf ta-lib-0.4.0-src.tar.gz
        $ cd ta-lib/
        $ ./configure --prefix=/usr
        $ make
        $ sudo make install
        Parameters
        ----------
        inputX : array with columns [timestamp, open, low, high, close, (volume)] the 'volume' column is optional but will increase performance of the model.
            An array that contains Bitcoin's open, low, high and close values per timestamp
            `inputX` will be used for Technical Analyses.

    """

    # Constructor
    def __init__(self, 
                 sma_close_timeperiod=3,
                 so_n=14,
                 so_d_n=3):
        self.useVolume = True
        # All parameters for all technical indicators
        self.sma_close_timeperiod = sma_close_timeperiod
        self.so_n = so_n
        self.so_d_n = so_d_n

    def _validate_input(self, X):
        # Check whether we have a timestamp, open, low, high, close, volume columns
        if 'Open' not in X:
            raise ValueError("Missing column 'Open'. Make sure to rename your dataframe colums when needed.")
        if 'Low' not in X:
            raise ValueError("Missing column 'Low'. Make sure to rename your dataframe colums when needed.")
        if 'High' not in X:
            raise ValueError("Missing column 'High'. Make sure to rename your dataframe colums when needed.")
        if 'Close' not in X:
            raise ValueError("Missing column 'Close'. Make sure to rename your dataframe colums when needed.")
        if 'Volume' not in X:
            print(
                "Warning: 'Volume' column not found. Column is optional but might have a positive effect on the performance of the models.")
            self.useVolume = False
        return X

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = self._validate_input(X)
        X = X.copy()
        # Simple Moving Average (SMA)
        '''
        Description:
        iloc is primarily integer position based (from 0 to length-1 of the axis), but may also be used with a boolean array.
        '''

        X['SMA_Close'] = talib.SMA(X["Close"], timeperiod=self.sma_close_timeperiod)

        # Stochastic Oscillator (SO)
        StochasticOscillator = ta.momentum.StochasticOscillator(high=X["High"], low=X["Low"],
                                                                close=X["Close"],
                                                                n=self.so_n, d_n=self.so_d_n, fillna=False)
        X['SO'] = StochasticOscillator.stoch()

        # Momentum (M)
        period = 3
        X['Momentum'] = talib.MOM(X["Close"], timeperiod=period)

        # Price Rate Of Change (ROC)
        '''
        Description:
        is a pure momentum oscillator that measures the percent change in price from one period to the next
        The ROC calculation compares the current price with the price “n” periods ago
        '''
        period = 12
        RateOfChange = ta.momentum.ROCIndicator(close=X["Close"], n=period, fillna=False)
        X['ROC'] = RateOfChange.roc()

        # Williams %R
        '''
        Description:
        Williams %R reflects the level of the close relative to the highest high for the look-back period
        Williams %R oscillates from 0 to -100.
        Readings from 0 to -20 are considered overbought. Readings from -80 to -100 are considered oversold.
        '''
        lookback_period = 14
        WilliamsR = ta.momentum.WilliamsRIndicator(high=X["High"], low=X["Low"], close=X["Close"],
                                                   lbp=lookback_period, fillna=False)
        X['WR'] = WilliamsR.wr()

        # Weighted Closing Price (WCP)
        X['WCP'] = talib.WCLPRICE(X["High"], X["Low"], X["Close"])

        # Williams Accumulation Distribution Line
        # AKA Accumulation/Distribution Index (ADI)????
        '''
        Description:
        a volume-based indicator designed to measure the cumulative flow of money into and out of a security
        The Accumulation Distribution Line rises when the multiplier is positive and falls when the multiplier is negative.
        '''
        ADI = ta.volume.AccDistIndexIndicator(high=X["High"], low=X["Low"], close=X["Close"],
                                              volume=X["Volume"],
                                              fillna=False)
        X['ADI'] = ADI.acc_dist_index()

        # Moving Average Convergence Divergence (MACD)
        '''
        Description:
        Is a trend-following momentum indicator that shows the relationship between two moving averages of prices.
        '''
        period_longterm = 26
        period_shortterm = 12
        period_to_signal = 9
        MACD = ta.trend.MACD(close=X["Close"], n_slow=period_longterm, n_fast=period_shortterm,
                             n_sign=period_to_signal, fillna=False)
        X['MACD'] = MACD.macd()

        # Commodity Channel Index (CCI)
        '''
        Description:
        CCI measures the difference between a security’s price change and its average price change. 
        High positive readings indicate that prices are well above their average, which is a show of strength. 
        Low negative readings indicate that prices are well below their average, which is a show of weakness.
        '''
        periods = 20
        constant = 0.015
        CCI = ta.trend.cci(high=X["High"], low=X["Low"], close=X["Close"], n=periods, c=constant,
                           fillna=False)
        X['CCI'] = CCI

        # Bollinger Bands (BB)
        '''
        Description:
        CCI measures the difference between a security’s price change and its average price change. 
        High positive readings indicate that prices are well above their average, which is a show of strength. 
        Low negative readings indicate that prices are well below their average, which is a show of weakness.
        '''
        periods = 20
        n_factor_standard_dev = 2
        indicator_bb = ta.volatility.BollingerBands(close=X["Close"], n=periods, ndev=n_factor_standard_dev,
                                                    fillna=False)
        # Add Bollinger Bands features
        # X['bb_bbm'] = indicator_bb.bollinger_mavg()
        X['BB_H'] = indicator_bb.bollinger_hband()
        X['BB_L'] = indicator_bb.bollinger_lband()

        # Add Bollinger Band high indicator
        X['bb_bbhi'] = indicator_bb.bollinger_hband_indicator()

        # Add Bollinger Band low indicator
        X['bb_bbli'] = indicator_bb.bollinger_lband_indicator()

        # Add width size Bollinger Bands
        X['bb_bbw'] = indicator_bb.bollinger_wband()

        # Mean Open & Close (M_O, M_C)
        period = 3
        X['MEAN_O_C'] = (talib.SMA(X["Open"], timeperiod=period) / 2) + (
                talib.SMA(X["Close"], timeperiod=period) / 2)

        # Variance Open & Close
        X["VAR_Close"] = talib.VAR(X["Close"], timeperiod=5, nbdev=1)
        X["VAR_Open"] = talib.VAR(X["Open"], timeperiod=5, nbdev=1)

        # High Price Average
        '''
        Description:
        Simple moving average over the high
        '''
        period = 3
        X['SMA_High'] = talib.SMA(X["High"], timeperiod=period)
        # Low Price Average
        '''
        Description:
        Simple moving average over the low
        '''
        period = 3
        X['SMA_Low'] = talib.SMA(X["Low"], timeperiod=period)

        # High, Low Average
        '''
        Description:
        Simple moving average over the sum of high and low
        '''
        period = 3
        X['SMA_H+L'] = talib.SMA(X["High"] + X["Low"], timeperiod=period)

        # Trading Day Price Average
        '''
        Description:
        Simple moving average over the sum of the open, high, low and close
        '''
        period = 3
        X['SMA_H+L+C+O'] = talib.SMA(X["High"] + X["Low"] + X["Open"] + X["Close"],
                                           timeperiod=period)

        # From here on adding random indicators according to the ta-lib library
        # ######################## OVERLAP STUDIES ############################
        # Double Exponential Moving Average
        period = 30
        X['DEMA'] = talib.DEMA(X["Close"], timeperiod=period)
        # Exponential Moving Average
        period = 30
        X['EMA'] = talib.EMA(X["Close"], timeperiod=period)
        # HT_TRENDLINE - Hilbert Transform - Instantaneous Trendline
        X['HT_TRENDLINE'] = talib.HT_TRENDLINE(X["Close"])
        # KAMA - Kaufman Adaptive Moving Average
        period = 30
        X['KAMA'] = talib.KAMA(X["Close"], timeperiod=period)
        # MA - Moving average
        period = 30
        X['MA'] = talib.MA(X["Close"], timeperiod=period, matype=0)
        # MIDPOINT - MidPoint over period
        period = 14
        X['MIDPOINT'] = talib.MIDPOINT(X["Close"], timeperiod=period)
        # MIDPRICE - Midpoint Price over period
        period = 14
        X['MIDPOINT'] = talib.MIDPRICE(X["High"], X["Low"], timeperiod=period)
        # SAR - Parabolic SAR
        X['SAR'] = talib.SAR(X["High"], X["Low"], acceleration=0, maximum=0)
        # SAREXT - Parabolic SAR - Extended
        X['SAREXT'] = talib.SAREXT(X["High"], X["Low"], startvalue=0, offsetonreverse=0,
                                         accelerationinitlong=0,
                                         accelerationlong=0, accelerationmaxlong=0, accelerationinitshort=0,
                                         accelerationshort=0, accelerationmaxshort=0)
        # T3 - Triple Exponential Moving Average (T3)
        period = 5
        X['T3'] = talib.T3(X["Close"], timeperiod=period, vfactor=0)
        # TEMA - Triple Exponential Moving Average
        period = 30
        X['TEMA'] = talib.TEMA(X["Close"], timeperiod=period)
        # TRIMA - Triangular Moving Average
        period = 30
        X['TRIMA'] = talib.TRIMA(X["Close"], timeperiod=period)
        # WMA - Weighted Moving Average
        period = 30
        X['WMA'] = talib.WMA(X["Close"], timeperiod=period)

        # ######################## Momentum Indicators ############################
        # ADX - Average Directional Movement Index
        period = 14
        X['ADX'] = talib.ADX(X["High"], X["Low"], X["Close"], timeperiod=period)
        # ADXR - Average Directional Movement Index Rating
        period = 14
        X['ADXR'] = talib.ADXR(X["High"], X["Low"], X["Close"], timeperiod=period)
        # APO - Absolute Price Oscillator
        X['APO'] = talib.APO(X["Close"], fastperiod=12, slowperiod=26, matype=0)
        # AROON - Aroon
        X['aroondown'], X['aroonup'] = talib.AROON(X["High"], X["Low"], timeperiod=14)
        # AROONOSC - Aroon Oscillator
        period = 14
        X['AROONOSC'] = talib.AROONOSC(X["High"], X["Low"], timeperiod=14)
        # BOP - Balance Of Power
        period = 14
        X['BOP'] = talib.BOP(X["Open"], X["High"], X["Low"], X["Close"])
        # CMO - Chande Momentum Oscillator
        X['CMO'] = talib.CMO(X["Close"], timeperiod=14)
        # DX - Directional Movement Index
        X['DX'] = talib.DX(X["High"], X["Low"], X["Close"], timeperiod=14)
        # MFI - Money Flow Index
        X['MFI'] = talib.MFI(X["High"], X["Low"], X["Close"], X["Volume"], timeperiod=14)
        # MINUS_DI - Minus Directional Indicator
        X['MINUS_DI'] = talib.MINUS_DI(X["High"], X["Low"], X["Close"], timeperiod=14)
        # MINUS_DM - Minus Directional Movement
        X['MINUS_DM'] = talib.MINUS_DM(X["High"], X["Low"], timeperiod=14)
        # PLUS_DI - Plus Directional Indicator
        X['PLUS_DI'] = talib.PLUS_DI(X["High"], X["Low"], X["Close"], timeperiod=14)
        # PLUS_DM - Plus Directional Movement
        X['PLUS_DM'] = talib.PLUS_DM(X["High"], X["Low"], timeperiod=14)
        # PPO - Percentage Price Oscillator
        X['PPO'] = talib.PPO(X["Close"], fastperiod=12, slowperiod=26, matype=0)
        # ROCP - Rate of change Percentage: (price-prevPrice)/prevPrice
        X['ROCP'] = talib.ROCP(X["Close"], timeperiod=10)
        # ROCR - Rate of change ratio: (price/prevPrice)
        X['ROCR'] = talib.ROCR(X["Close"], timeperiod=10)
        # ROCR100 - Rate of change ratio 100 scale: (price/prevPrice)*100
        X['ROCR100'] = talib.ROCR100(X["Close"], timeperiod=10)
        # RSI - Relative Strength Index
        X['RSI'] = talib.RSI(X["Close"], timeperiod=14)
        # TRIX - 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA
        X['TRIX'] = talib.TRIX(X["Close"], timeperiod=30)
        # ULTOSC - Ultimate Oscillator
        X['ULTOSC'] = talib.ULTOSC(X["High"], X["Low"], X["Close"], timeperiod1=7,
                                         timeperiod2=14, timeperiod3=28)

        # ######################## Volume Indicators ############################
        # AD - Chaikin A/D Line
        X['AD'] = talib.AD(X["High"], X["Low"], X["Close"], X["Volume"])
        # ADOSC - Chaikin A/D Oscillator
        X['ADOSC'] = talib.ADOSC(X["High"], X["Low"], X["Close"], X["Volume"],
                                       fastperiod=3, slowperiod=10)
        # OBV - On Balance Volume
        X['OBV'] = talib.OBV(X["Close"], X["Volume"])

        # ######################## Cycle Indicators ############################
        # HT_DCPERIOD - Hilbert Transform - Dominant Cycle Period
        X['HT_DCPERIOD'] = talib.HT_DCPERIOD(X["Close"])
        # HT_DCPHASE - Hilbert Transform - Dominant Cycle Phase
        X['HT_DCPHASE'] = talib.HT_DCPHASE(X["Close"])
        # HT_TRENDMODE - Hilbert Transform - Trend vs Cycle Mode
        X['HT_TRENDMODE'] = talib.HT_TRENDMODE(X["Close"])
        # ######################## Price transform functions ############################
        # AVGPRICE - Average Price
        X['AVGPRICE'] = talib.AVGPRICE(X["Open"], X["High"], X["Low"], X["Close"])
        # MEDPRICE - Median Price
        X['MEDPRICE'] = talib.MEDPRICE(X["High"], X["Low"])
        # TYPPRICE - Typical Price
        X['TYPPRICE'] = talib.TYPPRICE(X["High"], X["Low"], X["Close"])
        # WCLPRICE - Weighted Close Price
        X['WCLPRICE'] = talib.WCLPRICE(X["High"], X["Low"], X["Close"])
        # ################################ END OF TECHINCAL INDICATORS #########################
        # Delete columns that contains only NaN values
        ''' 
        cols = X.columns[X.isna().all()]
        X = X.drop(cols, axis=1)
        # Delete rows that contain at least one NaN
        X = X.dropna()
        '''
        return X

#------------------------------------- Custom help functions -------------------------------------#
def csvToDF(input="", dropna=True, timeOutput=False):
    """Reads a CSV as input. Converts the CSV to a pandas dataframe
    """
    # Load the input
    if timeOutput:
        print('Reading input...')
        start_time = time.time()
    df = pd.read_csv(input, encoding="ISO-8859-1")
    if dropna:
        # Clean NaN values
        # df = ta.utils.dropna(df)
        df = df.dropna(how='any')
    df.index.names = ['index']
    if timeOutput:
        end_time = time.time()
        print('Input read, took ' + str(end_time - start_time) + ' seconds')
    return df

def datetimeToTimestamp(input="", Date="01/01/1970", Time="00:00:00", Format="%m/%d/%Y %H:%M:%S", timeOutput=False):
    """Reads a pandas dataframe as input. Takes 3 parameters to format datetime to timestamp
        """
    if timeOutput:
        print('Generating Timestamp Column...')
        start_time = time.time()
    input['Time'] = input.apply(
        lambda row: datetime.timestamp(datetime.strptime(str(row.Date + ' ' + row.Time), Format)),
        axis=1)  # axis 1 = apply function to each row
    del input['Date']
    # Rename Time column to Timestamp column
    input = input.rename({'Time': 'Timestamp'}, axis=1)
    # Move timestamp column to first column after index
    timestamp = input['Timestamp']
    input.drop(labels=['Timestamp'], axis=1, inplace=True)
    input.insert(0, 'Timestamp', timestamp)
    if timeOutput:
        end_time = time.time()
        print('Time to create Timestamp: ' + str(end_time - start_time) + ' seconds')
    return input

def generateTarget(df="", method="Classification", typeRegression="Difference", intervalPeriods=1):
    """
    Parameters
    _____________
    df : pandas dataframe

    method : 'Classification' or 'Regression' . Type of machine learning to generate the 'Target' column
            if method equals 'Regression' additional parameters can be set for delta price calculations (typeRegression, 'intervalPeriods')

    typeRegression : 'Difference', 'Percentage' or 'ExactPrice'
        'Difference' : Target column = the difference between the Close price current interval and 'intervalPeriods' further as a float
        'Percentage' : Target column = the difference between the Close price current interval and 'intervalPeriods' further in percentages
        'ExactPrice' : Target column = the Close price of the current interval

    intervalPeriods : integer > 0
        How much intervals do we need to look in the future to calculate the difference
    """
    # Validate parameters
    if not isinstance(df, pd.DataFrame):
        raise ValueError('The df input is not a pandas dataframe')

    allowed_methods = ["Classification", "Regression"]
    if method not in allowed_methods:
        raise ValueError("Can only use these methods: {0} "
                         " got method={1}".format(allowed_methods,
                                                  method))
    if intervalPeriods < 1:
        raise ValueError("intervalPeriods parameter must be > 0 "
                         " got intervalPeriods={0}".format(intervalPeriods))

    allowed_methods = ["Difference", "Percentage", "ExactPrice"]
    if typeRegression not in allowed_methods:
        raise ValueError("Can only use these types of regression: {0} "
                         " got typeRegression={1}".format(allowed_methods,
                                                          typeRegression))
    # Start with function
    if method == "Classification":
        # Add new column 'Target' . True if Close price from this timestmap is higher than Close price of previous timestamp
        df['Target'] = np.where(df.Close.shift(-1) > df.Close, True, False)
    elif method == "Regression":
        if typeRegression == "Difference":
            df['Target'] = df.Close.shift(-intervalPeriods) - df.Close
        elif typeRegression == "Percentage":
            dfclose = df['Close']
            df['Target'] = -dfclose.pct_change(periods=-intervalPeriods)
        elif typeRegression == "Target":
            df['Target'] = df.Close.shift(-intervalPeriods)

    return df['Target']

#------------------------------------- Workflow, preparing dataset ~ results of models -------------------------------------#
# 1. Load CSV
# 2. X = Open, Low, High, Close and Volume columns with data
# 3. target = target column, calculate with generateTarget()
# 4. Train test split
# 5. Pipeline
# 6. Fit & Predict
# 7. Results

# 1. Load CSV
# __________________________
df = csvToDF("bitfinex_tBTCUSD_1m.csv", dropna=False)
    # (optional) Convert datetime to timestamp
    #inputX = datetimeToTimestamp(input=inputX, Date=inputX['Date'], Time=inputX['Time'], Format='%Y-%m-%d %H:%M:%S')
# 2. Prepare X
# __________________________
X = df[["Open", "Low", "High", "Close", "Volume"]]
# 3. Prepare target
# __________________________
target = generateTarget(df, method="Classification", typeRegression="Difference", intervalPeriods=1)
# 4. Train test split
# __________________________
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2, train_size=0.8, shuffle=False, random_state = 42)

# 5. Start the pipeline
# __________________________
pipeline = Pipeline(steps=[
  ('BitcoinTransformer', BitcoinTransformer(sma_close_timeperiod=3,
                                         so_n=14,
                                         so_d_n=3)
   ),
  ('imputing',SimpleImputer(missing_values=np.nan, strategy='mean')),
  ('classify', RandomForestClassifier(n_estimators=10, random_state=42))
])

# 6. Fit & Predict the pipeline
# __________________________
pipeline.fit(X_train, y_train)
y_predict = pipeline.predict(X_test)

# 7. Scores
# __________________________
print('-------------  Real Random Forest Model ------------- ')
print('Accuracy score:')
print(round(accuracy_score(y_test, y_predict)  * 100, 2))
print('Conf matrix:')
print(confusion_matrix(y_test, y_predict))
print('Classification report:')
print(classification_report(y_test, y_predict))
