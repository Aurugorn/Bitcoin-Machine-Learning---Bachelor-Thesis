# Authors: Stan van der Avoird <stan@restica.nl>
#
import datetime

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
                 so_d_n=3,
                 momentum_period=3,
                 roc_period=12,
                 wr_lookback_period=14,
                 macd_period_longterm=26,
                 macd_period_shortterm=12,
                 macd_period_to_signal=9,
                 bb_periods=20,
                 mean_o_c_period=3,
                 var_close_period=5,
                 var_open_period=5,
                 sma_high_period=3,
                 sma_low_period=3,
                 sma_handl_period=3,
                 sma_h_l_c_o_period=3,
                 dema_period=30,
                 ema_period=30,
                 kama_period=30,
                 ma_period=30,
                 midpoint_period=14,
                 midprice_period=14,
                 sar_acceleration=0,
                 sar_maximum=0,
                 sarext_startvalue=0,
                 sarext_offsetonreverse=0,
                 sarext_accelerationinitlong=0,
                 sarext_accelerationlong=0,
                 sarext_accelerationmaxlong=0,
                 sarext_accelerationinitshort=0,
                 sarext_accelerationshort=0,
                 sarext_accelerationmaxshort=0,
                 t3_period=5,
                 tema_period=30,
                 trima_period=30,
                 wma_period=30,
                 adx_period=14,
                 adxr_period=14,
                 apo_fastperiod=12,
                 apo_slowperiod=26,
                 aroon_period=14,
                 aroonosc_period=14,
                 cmo_period=14,
                 dx_period=14,
                 mfi_period=14,
                 minus_di_period=14,
                 minus_dm_period=14,
                 plus_di_period=14,
                 plus_dm_period=14,
                 ppo_fastperiod=12,
                 ppo_slowperiod=26,
                 rocp_period=10,
                 rocr_period=10,
                 rocr100_period=10,
                 rsi_period=14,
                 trix_period=30,
                 ultosc_period1=7,
                 ultosc_period2=14,
                 ultosc_period3=28,
                 adosc_fastperiod=3,
                 adosc_slowperiod=10
                 ):
        '''
        @param sma_close_timeperiod:
        @type sma_close_timeperiod:
        @param so_n:
        @type so_n:
        @param so_d_n:
        @type so_d_n:
        @param momentum_period:
        @type momentum_period:
        @param roc_period:
        @type roc_period:
        @param wr_lookback_period:
        @type wr_lookback_period:
        @param macd_period_longterm:
        @type macd_period_longterm:
        @param macd_period_shortterm:
        @type macd_period_shortterm:
        @param macd_period_to_signal:
        @type macd_period_to_signal:
        @param bb_periods:
        @type bb_periods:
        @param bb_n_factor_standard_dev:
        @type bb_n_factor_standard_dev:
        @param mean_o_c_period:
        @type mean_o_c_period:
        @param var_close_period:
        @type var_close_period:
        @param var_close_nbdev:
        @type var_close_nbdev:
        @param var_open_period:
        @type var_open_period:
        @param var_open_nbdev:
        @type var_open_nbdev:
        @param sma_high_period:
        @type sma_high_period:
        @param sma_low_period:
        @type sma_low_period:
        @param sma_handl_period:
        @type sma_handl_period:
        @param sma_h_l_c_o_period:
        @type sma_h_l_c_o_period:
        @param dema_period:
        @type dema_period:
        @param ema_period:
        @type ema_period:
        @param kama_period:
        @type kama_period:
        @param ma_period:
        @type ma_period:
        @param midpoint_period:
        @type midpoint_period:
        @param midprice_period:
        @type midprice_period:
        @param sar_acceleration:
        @type sar_acceleration:
        @param sar_maximum:
        @type sar_maximum:
        @param sarext_startvalue:
        @type sarext_startvalue:
        @param sarext_offsetonreverse:
        @type sarext_offsetonreverse:
        @param sarext_accelerationinitlong:
        @type sarext_accelerationinitlong:
        @param sarext_accelerationlong:
        @type sarext_accelerationlong:
        @param sarext_accelerationmaxlong:
        @type sarext_accelerationmaxlong:
        @param sarext_accelerationinitshort:
        @type sarext_accelerationinitshort:
        @param sarext_accelerationshort:
        @type sarext_accelerationshort:
        @param sarext_accelerationmaxshort:
        @type sarext_accelerationmaxshort:
        @param t3_period:
        @type t3_period:
        @param t3_vfactor:
        @type t3_vfactor:
        @param tema_period:
        @type tema_period:
        @param trima_period:
        @type trima_period:
        @param wma_period:
        @type wma_period:
        @param adx_period:
        @type adx_period:
        @param adxr_period:
        @type adxr_period:
        @param apo_fastperiod:
        @type apo_fastperiod:
        @param apo_slowperiod:
        @type apo_slowperiod:
        @param apo_matype:
        @type apo_matype:
        @param aroon_period:
        @type aroon_period:
        @param aroonosc_period:
        @type aroonosc_period:
        @param cmo_period:
        @type cmo_period:
        @param dx_period:
        @type dx_period:
        @param mfi_period:
        @type mfi_period:
        @param minus_di_period:
        @type minus_di_period:
        @param minus_dm_period:
        @type minus_dm_period:
        @param plus_di_period:
        @type plus_di_period:
        @param plus_dm_period:
        @type plus_dm_period:
        @param ppo_fastperiod:
        @type ppo_fastperiod:
        @param ppo_slowperiod:
        @type ppo_slowperiod:
        @param ppo_matype:
        @type ppo_matype:
        @param rocp_period:
        @type rocp_period:
        @param rocr_period:
        @type rocr_period:
        @param rocr100_period:
        @type rocr100_period:
        @param rsi_period:
        @type rsi_period:
        @param trix_period:
        @type trix_period:
        @param ultosc_period1:
        @type ultosc_period1:
        @param ultosc_period2:
        @type ultosc_period2:
        @param ultosc_period3:
        @type ultosc_period3:
        @param adosc_fastperiod:
        @type adosc_fastperiod:
        @param adosc_slowperiod:
        @type adosc_slowperiod:
        '''
        self.useVolume = True
        # All parameters for all technical indicators
        self.sma_close_timeperiod = sma_close_timeperiod
        self.so_n = so_n
        self.so_d_n = so_d_n
        self.momentum_period = momentum_period
        self.roc_period = roc_period
        self.wr_lookback_period = wr_lookback_period
        self.macd_period_longterm = macd_period_longterm
        self.macd_period_shortterm = macd_period_shortterm
        self.macd_period_to_signal = macd_period_to_signal
        self.bb_periods = bb_periods
        self.mean_o_c_period = mean_o_c_period
        self.var_close_period = var_close_period
        self.var_open_period = var_open_period
        self.sma_high_period = sma_high_period
        self.sma_low_period = sma_low_period
        self.sma_handl_period = sma_handl_period
        self.sma_h_l_c_o_period = sma_h_l_c_o_period
        self.dema_period = dema_period
        self.ema_period = ema_period
        self.kama_period = kama_period
        self.ma_period = ma_period
        self.midpoint_period = midpoint_period
        self.midprice_period = midprice_period
        self.sar_acceleration = sar_acceleration
        self.sar_maximum = sar_maximum
        self.sarext_startvalue = sarext_startvalue
        self.sarext_offsetonreverse = sarext_offsetonreverse
        self.sarext_accelerationinitlong = sarext_accelerationinitlong
        self.sarext_accelerationlong = sarext_accelerationlong
        self.sarext_accelerationmaxlong = sarext_accelerationmaxlong
        self.sarext_accelerationinitshort = sarext_accelerationinitshort
        self.sarext_accelerationshort = sarext_accelerationshort
        self.sarext_accelerationmaxshort = sarext_accelerationmaxshort
        self.t3_period = t3_period
        self.tema_period = tema_period
        self.trima_period = trima_period
        self.wma_period = wma_period
        self.adx_period = adx_period
        self.adxr_period = adxr_period
        self.apo_fastperiod = apo_fastperiod
        self.apo_slowperiod = apo_slowperiod
        self.aroon_period = aroon_period
        self.aroonosc_period = aroonosc_period
        self.cmo_period = cmo_period
        self.dx_period = dx_period
        self.mfi_period = mfi_period
        self.minus_di_period = minus_di_period
        self.minus_dm_period = minus_dm_period
        self.plus_di_period = plus_di_period
        self.plus_dm_period = plus_dm_period
        self.ppo_fastperiod = ppo_fastperiod
        self.ppo_slowperiod = ppo_slowperiod
        self.rocp_period = rocp_period
        self.rocr_period = rocr_period
        self.rocr100_period = rocr100_period
        self.rsi_period = rsi_period
        self.trix_period = trix_period
        self.ultosc_period1 = ultosc_period1
        self.ultosc_period2 = ultosc_period2
        self.ultosc_period3 = ultosc_period3
        self.adosc_fastperiod = adosc_fastperiod
        self.adosc_slowperiod = adosc_slowperiod

        #Set couple of paramaters to default, dont change
        self.bb_n_factor_standard_dev = 2
        self.var_close_nbdev = 1
        self.var_open_nbdev = 1
        self.t3_vfactor = 0
        self.apo_matype = 0
        self.ppo_matype = 0

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
        self.sma_close_timeperiod = int(self.sma_close_timeperiod)
        self.so_n = int(self.so_n)
        self.so_d_n = int(self.so_d_n)
        self.momentum_period = int(self.momentum_period)
        self.roc_period = int(self.roc_period)
        self.wr_lookback_period = int(self.wr_lookback_period)
        self.macd_period_longterm = int(self.macd_period_longterm)
        self.macd_period_shortterm = int(self.macd_period_shortterm)
        self.macd_period_to_signal = int(self.macd_period_to_signal)
        self.bb_periods = int(self.bb_periods)
        self.mean_o_c_period = int(self.mean_o_c_period)
        self.var_close_period = int(self.var_close_period)
        self.var_open_period = int(self.var_open_period)
        self.sma_high_period = int(self.sma_high_period)
        self.sma_low_period = int(self.sma_low_period)
        self.sma_handl_period = int(self.sma_handl_period)
        self.sma_h_l_c_o_period = int(self.sma_h_l_c_o_period)
        self.dema_period = int(self.dema_period)
        self.ema_period = int(self.ema_period)
        self.kama_period = int(self.kama_period)
        self.ma_period = int(self.ma_period)
        self.midpoint_period = int(self.midpoint_period)
        self.midprice_period = int(self.midprice_period)
        self.sar_acceleration = int(self.sar_acceleration)
        self.sar_maximum = int(self.sar_maximum)
        self.sarext_startvalue = int(self.sarext_startvalue)
        self.sarext_offsetonreverse = int(self.sarext_offsetonreverse)
        self.sarext_accelerationinitlong = int(self.sarext_accelerationinitlong)
        self.sarext_accelerationlong = int(self.sarext_accelerationlong)
        self.sarext_accelerationmaxlong = int(self.sarext_accelerationmaxlong)
        self.sarext_accelerationinitshort = int(self.sarext_accelerationinitshort)
        self.sarext_accelerationshort = int(self.sarext_accelerationshort)
        self.sarext_accelerationmaxshort = int(self.sarext_accelerationmaxshort)
        self.t3_period = int(self.t3_period)
        self.tema_period = int(self.tema_period)
        self.trima_period = int(self.trima_period)
        self.wma_period = int(self.wma_period)
        self.adx_period = int(self.adx_period)
        self.adxr_period = int(self.adxr_period)
        self.apo_fastperiod = int(self.apo_fastperiod)
        self.apo_slowperiod = int(self.apo_slowperiod)
        self.aroon_period = int(self.aroon_period)
        self.aroonosc_period = int(self.aroonosc_period)
        self.cmo_period = int(self.cmo_period)
        self.dx_period = int(self.dx_period)
        self.mfi_period = int(self.mfi_period)
        self.minus_di_period = int(self.minus_di_period)
        self.minus_dm_period = int(self.minus_dm_period)
        self.plus_di_period = int(self.plus_di_period)
        self.plus_dm_period = int(self.plus_dm_period)
        self.ppo_fastperiod = int(self.ppo_fastperiod)
        self.ppo_slowperiod = int(self.ppo_slowperiod)
        self.rocp_period = int(self.rocp_period)
        self.rocr_period = int(self.rocr_period)
        self.rocr100_period = int(self.rocr100_period)
        self.rsi_period = int(self.rsi_period)
        self.trix_period = int(self.trix_period)
        self.ultosc_period1 = int(self.ultosc_period1)
        self.ultosc_period2 = int(self.ultosc_period2)
        self.ultosc_period3 = int(self.ultosc_period3)
        self.adosc_fastperiod = int(self.adosc_fastperiod)
        self.adosc_slowperiod = int(self.adosc_slowperiod)
        return self

    def transform(self, X):
        now = datetime.datetime.now()
        # print('[{}] : Starting new Transformer optimizing 67 Technical Indicator parameters'.format(now.strftime("%Y-%m-%d %H:%M:%S")))

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
        X['Momentum'] = talib.MOM(X["Close"], timeperiod=self.momentum_period)

        # Price Rate Of Change (ROC)
        '''
        Description:
        is a pure momentum oscillator that measures the percent change in price from one period to the next
        The ROC calculation compares the current price with the price “n” periods ago
        '''
        RateOfChange = ta.momentum.ROCIndicator(close=X["Close"], n=self.roc_period, fillna=False)
        X['ROC'] = RateOfChange.roc()

        # Williams %R
        '''
        Description:
        Williams %R reflects the level of the close relative to the highest high for the look-back period
        Williams %R oscillates from 0 to -100.
        Readings from 0 to -20 are considered overbought. Readings from -80 to -100 are considered oversold.
        '''
        WilliamsR = ta.momentum.WilliamsRIndicator(high=X["High"], low=X["Low"], close=X["Close"],
                                                   lbp=self.wr_lookback_period, fillna=False)
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
        MACD = ta.trend.MACD(close=X["Close"], n_slow=self.macd_period_longterm, n_fast=self.macd_period_shortterm,
                             n_sign=self.macd_period_to_signal, fillna=False)
        X['MACD'] = MACD.macd()

        # Bollinger Bands (BB)
        '''
        Description:
        CCI measures the difference between a security’s price change and its average price change. 
        High positive readings indicate that prices are well above their average, which is a show of strength. 
        Low negative readings indicate that prices are well below their average, which is a show of weakness.
        '''
        indicator_bb = ta.volatility.BollingerBands(close=X["Close"], n=self.bb_periods, ndev=self.bb_n_factor_standard_dev,
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
        X['MEAN_O_C'] = (talib.SMA(X["Open"], timeperiod=self.mean_o_c_period) / 2) + (
                talib.SMA(X["Close"], timeperiod=self.mean_o_c_period) / 2)

        # Variance Open & Close
        X["VAR_Close"] = talib.VAR(X["Close"], timeperiod=self.var_close_period, nbdev=self.var_close_nbdev)
        X["VAR_Open"] = talib.VAR(X["Open"], timeperiod=self.var_open_period, nbdev=self.var_open_nbdev)

        # High Price Average
        '''
        Description:
        Simple moving average over the high
        '''
        X['SMA_High'] = talib.SMA(X["High"], timeperiod=self.sma_high_period)
        # Low Price Average
        '''
        Description:
        Simple moving average over the low
        '''
        X['SMA_Low'] = talib.SMA(X["Low"], timeperiod=self.sma_low_period)

        # High, Low Average
        '''
        Description:
        Simple moving average over the sum of high and low
        '''
        X['SMA_H+L'] = talib.SMA(X["High"] + X["Low"], timeperiod=self.sma_handl_period)

        # Trading Day Price Average
        '''
        Description:
        Simple moving average over the sum of the open, high, low and close
        '''
        X['SMA_H+L+C+O'] = talib.SMA(X["High"] + X["Low"] + X["Open"] + X["Close"],
                                     timeperiod=self.sma_h_l_c_o_period)

        # From here on adding random indicators according to the ta-lib library
        # ######################## OVERLAP STUDIES ############################
        # Double Exponential Moving Average
        X['DEMA'] = talib.DEMA(X["Close"], timeperiod=self.dema_period)
        # Exponential Moving Average
        X['EMA'] = talib.EMA(X["Close"], timeperiod=self.ema_period)
        # HT_TRENDLINE - Hilbert Transform - Instantaneous Trendline
        X['HT_TRENDLINE'] = talib.HT_TRENDLINE(X["Close"])
        # KAMA - Kaufman Adaptive Moving Average
        X['KAMA'] = talib.KAMA(X["Close"], timeperiod=self.kama_period)
        # MA - Moving average
        X['MA'] = talib.MA(X["Close"], timeperiod=self.ma_period, matype=0)
        # MIDPOINT - MidPoint over period
        X['MIDPOINT'] = talib.MIDPOINT(X["Close"], timeperiod=self.midpoint_period)
        # MIDPRICE - Midpoint Price over period
        X['MIDPRICE'] = talib.MIDPRICE(X["High"], X["Low"], timeperiod=self.midprice_period)
        # SAR - Parabolic SAR
        X['SAR'] = talib.SAR(X["High"], X["Low"], acceleration=self.sar_acceleration, maximum=self.sar_maximum)
        # SAREXT - Parabolic SAR - Extended
        X['SAREXT'] = talib.SAREXT(X["High"], X["Low"], startvalue=self.sarext_startvalue, offsetonreverse=self.sarext_offsetonreverse,
                                   accelerationinitlong=self.sarext_accelerationinitlong,
                                   accelerationlong=self.sarext_accelerationlong, accelerationmaxlong=self.sarext_accelerationmaxlong, accelerationinitshort=self.sarext_accelerationinitshort,
                                   accelerationshort=self.sarext_accelerationshort, accelerationmaxshort=self.sarext_accelerationmaxshort)
        # T3 - Triple Exponential Moving Average (T3)
        X['T3'] = talib.T3(X["Close"], timeperiod=self.t3_period, vfactor=self.t3_vfactor)
        # TEMA - Triple Exponential Moving Average
        X['TEMA'] = talib.TEMA(X["Close"], timeperiod=self.tema_period)
        # TRIMA - Triangular Moving Average
        X['TRIMA'] = talib.TRIMA(X["Close"], timeperiod=self.trima_period)
        # WMA - Weighted Moving Average
        X['WMA'] = talib.WMA(X["Close"], timeperiod=self.wma_period)

        # ######################## Momentum Indicators ############################
        # ADX - Average Directional Movement Index
        X['ADX'] = talib.ADX(X["High"], X["Low"], X["Close"], timeperiod=self.adx_period)
        # ADXR - Average Directional Movement Index Rating
        X['ADXR'] = talib.ADXR(X["High"], X["Low"], X["Close"], timeperiod=self.adxr_period)
        # APO - Absolute Price Oscillator

        X['APO'] = talib.APO(X["Close"], fastperiod=self.apo_fastperiod, slowperiod=self.apo_slowperiod, matype=self.apo_matype)
        # AROON - Aroon
        X['aroondown'], X['aroonup'] = talib.AROON(X["High"], X["Low"], timeperiod=self.aroon_period)
        # AROONOSC - Aroon Oscillator
        X['AROONOSC'] = talib.AROONOSC(X["High"], X["Low"], timeperiod=self.aroonosc_period)
        # BOP - Balance Of Power
        X['BOP'] = talib.BOP(X["Open"], X["High"], X["Low"], X["Close"])
        # CMO - Chande Momentum Oscillator
        X['CMO'] = talib.CMO(X["Close"], timeperiod=self.cmo_period)
        # DX - Directional Movement Index
        X['DX'] = talib.DX(X["High"], X["Low"], X["Close"], timeperiod=self.dx_period)
        if self.useVolume:
            # MFI - Money Flow Index
            X['MFI'] = talib.MFI(X["High"], X["Low"], X["Close"], X["Volume"], timeperiod=self.mfi_period)
        # MINUS_DI - Minus Directional Indicator
        X['MINUS_DI'] = talib.MINUS_DI(X["High"], X["Low"], X["Close"], timeperiod=self.minus_di_period)
        # MINUS_DM - Minus Directional Movement
        X['MINUS_DM'] = talib.MINUS_DM(X["High"], X["Low"], timeperiod=self.minus_dm_period)
        # PLUS_DI - Plus Directional Indicator
        X['PLUS_DI'] = talib.PLUS_DI(X["High"], X["Low"], X["Close"], timeperiod=self.plus_di_period)
        # PLUS_DM - Plus Directional Movement
        X['PLUS_DM'] = talib.PLUS_DM(X["High"], X["Low"], timeperiod=self.plus_dm_period)
        # PPO - Percentage Price Oscillator
        X['PPO'] = talib.PPO(X["Close"], fastperiod=self.ppo_fastperiod, slowperiod=self.ppo_slowperiod, matype=self.ppo_matype)
        # ROCP - Rate of change Percentage: (price-prevPrice)/prevPrice
        X['ROCP'] = talib.ROCP(X["Close"], timeperiod=self.rocp_period)
        # ROCR - Rate of change ratio: (price/prevPrice)
        X['ROCR'] = talib.ROCR(X["Close"], timeperiod=self.rocr_period)
        # ROCR100 - Rate of change ratio 100 scale: (price/prevPrice)*100
        X['ROCR100'] = talib.ROCR100(X["Close"], timeperiod=self.rocr100_period)
        # RSI - Relative Strength Index
        X['RSI'] = talib.RSI(X["Close"], timeperiod=self.rsi_period)
        # TRIX - 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA
        X['TRIX'] = talib.TRIX(X["Close"], timeperiod=self.trix_period)
        # ULTOSC - Ultimate Oscillator
        X['ULTOSC'] = talib.ULTOSC(X["High"], X["Low"], X["Close"], timeperiod1=self.ultosc_period1,
                                   timeperiod2=self.ultosc_period1, timeperiod3=self.ultosc_period3)

        if self.useVolume:
            # ######################## Volume Indicators ############################
            # AD - Chaikin A/D Line
            X['AD'] = talib.AD(X["High"], X["Low"], X["Close"], X["Volume"])
            # ADOSC - Chaikin A/D Oscillator
            X['ADOSC'] = talib.ADOSC(X["High"], X["Low"], X["Close"], X["Volume"],
                                     fastperiod=self.adosc_fastperiod, slowperiod=self.adosc_slowperiod)
            # OBV - On Balance Volume
            X['OBV'] = talib.OBV(X["Close"], X["Volume"])

        # ######################## Cycle Indicators ############################
        # HT_DCPERIOD - Hilbert Transform - Dominant Cycle Period
        X['HT_DCPERIOD'] = talib.HT_DCPERIOD(X["Close"])
        # HT_DCPHASE - Hilbert Transform - Dominant Cycle Phase
        #X['HT_DCPHASE'] = talib.HT_DCPHASE(X["Close"])
        # HT_TRENDMODE - Hilbert Transform - Trend vs Cycle Mode
        #X['HT_TRENDMODE'] = talib.HT_TRENDMODE(X["Close"])
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

# Class to combine several parameters from the BitcoinTransformer to reduce the amount of total combination of parameters
class ParameterRelationsBTCTrans(BitcoinTransformer):
    """
    Child of BitcoinTransformer to combine several technical indicator hyper parameters to reduce the
    total amount of hyperparameters exposed to (auto) sklearn.
    This child combines matching default values of the parameters.
    """
    def __init__(self,
                 fastperiod = 3,
                 longterm = 26,
                 midterm = 14,
                 shortterm = 12,
                 bb_cci = 20,
                 var_t3 = 5,
                 dema_trema = 30,
                 zero = 0,
                 rocperiod = 10):
        self.fastperiod = fastperiod
        self.longterm = longterm
        self.midterm = midterm
        self.shortterm = shortterm
        self.bb_cci = bb_cci
        self.var_t3 = var_t3
        self.dema_trema = dema_trema
        self.zero = zero
        self.rocperiod = rocperiod


        BitcoinTransformer.__init__(self, sma_close_timeperiod=self.fastperiod, so_d_n=self.fastperiod, momentum_period=self.fastperiod, mean_o_c_period=self.fastperiod, sma_high_period=self.fastperiod, sma_low_period=self.fastperiod , sma_handl_period=self.fastperiod, sma_h_l_c_o_period=self.fastperiod, adosc_fastperiod=self.fastperiod,
                                    macd_period_longterm=self.longterm, apo_slowperiod=self.longterm, ppo_slowperiod=self.longterm,
                                    so_n=self.midterm,
                                    wr_lookback_period=self.midterm,
                                    midpoint_period=self.midterm,
                                    midprice_period=self.midterm,
                                    adx_period=self.midterm,
                                    adxr_period=self.midterm,
                                    aroon_period=self.midterm,
                                    aroonosc_period=self.midterm,
                                    cmo_period=self.midterm,
                                    dx_period=self.midterm,
                                    mfi_period=self.midterm,
                                    minus_di_period=self.midterm,
                                    minus_dm_period=self.midterm,
                                    plus_di_period=self.midterm,
                                    plus_dm_period=self.midterm,
                                    rsi_period=self.midterm,
                                    ultosc_period2=self.midterm,

                                    roc_period=self.shortterm,
                                    macd_period_shortterm=self.shortterm,
                                    apo_fastperiod=self.shortterm,
                                    ppo_fastperiod=self.shortterm,

                                    bb_periods=self.bb_cci,
                                    var_close_period=self.var_t3,
                                    var_open_period=self.var_t3,
                                    t3_period=self.var_t3,

                                    dema_period=self.dema_trema,
                                    ema_period=self.dema_trema,
                                    kama_period=self.dema_trema,
                                    ma_period=self.dema_trema,
                                    tema_period=self.dema_trema,
                                    trima_period=self.dema_trema,
                                    wma_period=self.dema_trema,
                                    trix_period=self.dema_trema,

                                    sar_acceleration=self.zero,
                                    sar_maximum=self.zero,
                                    sarext_startvalue=self.zero,
                                    sarext_offsetonreverse=self.zero,
                                    sarext_accelerationinitlong=self.zero,
                                    sarext_accelerationlong=self.zero,
                                    sarext_accelerationmaxlong=self.zero,
                                    sarext_accelerationinitshort=self.zero,
                                    sarext_accelerationshort=self.zero,
                                    sarext_accelerationmaxshort=self.zero,

                                    rocp_period=self.rocperiod,
                                    rocr_period=self.rocperiod,
                                    rocr100_period=self.rocperiod,
                                    adosc_slowperiod=self.rocperiod
                                    )
        '''
        sma_close_timeperiod=3,
        so_d_n=3,          
        momentum_period=3,
        mean_o_c_period=3,
        sma_high_period=3,
        sma_low_period=3,
        sma_handl_period=3,
        sma_h_l_c_o_period=3,
        adosc_fastperiod=3,
        
        so_n=14,
        wr_lookback_period=14,
        midpoint_period=14,
        midprice_period=14,
        adx_period=14,
        adxr_period=14,
        aroon_period=14,
        aroonosc_period=14,
        cmo_period=14,
        dx_period=14,
        mfi_period=14,
        minus_di_period=14,
        minus_dm_period=14,
        plus_di_period=14,
        plus_dm_period=14,
        rsi_period=14,
        ultosc_period2=14,
        
        roc_period=12,
        macd_period_shortterm=12,
        apo_fastperiod=12,
        ppo_fastperiod=12,
        
        macd_period_longterm=26,
        apo_slowperiod=26,
        ppo_slowperiod=26,
        
        macd_period_to_signal=9,
        
        bb_periods=20,
      
        
        bb_n_factor_standard_dev=2,
        
        var_close_period=5,
        var_open_period=5,
        t3_period=5,
        
        var_close_nbdev=1,
        var_open_nbdev=1,
        
        dema_period=30,
        ema_period=30,
        kama_period=30,
        ma_period=30,
        tema_period=30,
        trima_period=30,
        wma_period=30,
        trix_period=30,
        
        sar_acceleration=0,
        sar_maximum=0,
        sarext_startvalue=0,
        sarext_offsetonreverse=0,
        sarext_accelerationinitlong=0,
        sarext_accelerationlong=0,
        sarext_accelerationmaxlong=0,
        sarext_accelerationinitshort=0,
        sarext_accelerationshort=0,
        sarext_accelerationmaxshort=0,
        t3_vfactor=0,
        apo_matype=0,
        ppo_matype=0,
        
        rocp_period=10,
        rocr_period=10,
        rocr100_period=10,
        adosc_slowperiod=10
        
        ultosc_period1=7,
        
        ultosc_period3=28,
        

        '''