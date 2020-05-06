# Authors: Stan van der Avoird <stan@restica.nl>
#

import ta
import talib
from sklearn.base import BaseEstimator, TransformerMixin

# ------------------------------------- BitcoinTransformer component -------------------------------------#
class BitcoinTransformer(BaseEstimator, TransformerMixin):
    """
    Bitcoin Transformer to prepare dataset for machine learning
    Accepts dataset with 4 case sensitive columns; "Open", "Low", "High", "Close"
    Accepts optional extra case sensitive column to calculate volume related technical indicators; "Volume"
    And calculates Technical Indicators.
    """
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
        """
        Function to train the dataset. After fitting the model can be used to make predictions, usually with a .predict() method call.
        @param X: Training set
        @type X: numpy array of shape [n_samples, n_features]
        @param y: Target values.
        @type y: numpy array of shape [n_samples]
        @return: Trained set
        @rtype: self
        """
        return self

    def transform(self, X):
        """
        @param X: Training set
        @type X: numpy array of shape [n_samples, n_features]
        @return: Transformed array.
        @rtype: X_newnumpy array of shape [n_samples, n_features_new]
        """
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
        if self.useVolume:
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
        if self.useVolume:
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

        if self.useVolume:
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

