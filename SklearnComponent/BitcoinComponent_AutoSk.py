from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter
from ConfigSpace.conditions import InCondition

import sklearn.metrics
import autosklearn.classification
import autosklearn.pipeline.components.feature_preprocessing
from autosklearn.pipeline.components.base \
    import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import DENSE, SIGNED_DATA, \
    UNSIGNED_DATA
from autosklearn.util.common import check_none
import ta
import talib

# Create BitcoinTransformer_AutoSk component for auto-sklearn.
class BitcoinTransformer_AutoSk(AutoSklearnPreprocessingAlgorithm):
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
                 cci_periods=20,
                 cci_constant=0.015,
                 bb_periods=20,
                 bb_n_factor_standard_dev=2,
                 mean_o_c_period=3,
                 var_close_period=5,
                 var_close_nbdev=1,
                 var_open_period=5,
                 var_open_nbdev=1,
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
                 t3_vfactor=0,
                 tema_period=30,
                 trima_period=30,
                 wma_period=30,
                 adx_period=14,
                 adxr_period=14,
                 apo_fastperiod=12,
                 apo_slowperiod=26,
                 apo_matype=0,
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
                 ppo_matype=0,
                 rocp_period=10,
                 rocr_period=10,
                 rocr100_period=10,
                 rsi_period=14,
                 trix_period=30,
                 ultosc_period1=7,
                 ultosc_period2=14,
                 ultosc_period3=28,
                 adosc_fastperiod=3,
                 adosc_slowperiod=10,
                 random_state=None):


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
        @param cci_periods:
        @type cci_periods:
        @param cci_constant:
        @type cci_constant:
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
        self.cci_periods = cci_periods
        self.cci_constant = cci_constant
        self.bb_periods = bb_periods
        self.bb_n_factor_standard_dev = bb_n_factor_standard_dev
        self.mean_o_c_period = mean_o_c_period
        self.var_close_period = var_close_period
        self.var_close_nbdev = var_close_nbdev
        self.var_open_period = var_open_period
        self.var_open_nbdev = var_open_nbdev
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
        self.t3_vfactor = t3_vfactor
        self.tema_period = tema_period
        self.trima_period = trima_period
        self.wma_period = wma_period
        self.adx_period = adx_period
        self.adxr_period = adxr_period
        self.apo_fastperiod = apo_fastperiod
        self.apo_slowperiod = apo_slowperiod
        self.apo_matype = apo_matype
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
        self.ppo_matype = ppo_matype
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

        self.random_state = random_state
        self.preprocessor = None

    def fit(self, X, y=None):

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

        # Commodity Channel Index (CCI)
        '''
        Description:
        CCI measures the difference between a security’s price change and its average price change. 
        High positive readings indicate that prices are well above their average, which is a show of strength. 
        Low negative readings indicate that prices are well below their average, which is a show of weakness.
        '''
        CCI = ta.trend.cci(high=X["High"], low=X["Low"], close=X["Close"], n=self.cci_periods, c=self.cci_constant,
                           fillna=False)
        X['CCI'] = CCI

        # Bollinger Bands (BB)
        '''
        Description:
        CCI measures the difference between a security’s price change and its average price change. 
        High positive readings indicate that prices are well above their average, which is a show of strength. 
        Low negative readings indicate that prices are well below their average, which is a show of weakness.
        '''
        indicator_bb = ta.volatility.BollingerBands(close=X["Close"], n=self.bb_periods,
                                                    ndev=self.bb_n_factor_standard_dev,
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
        X['SAREXT'] = talib.SAREXT(X["High"], X["Low"], startvalue=self.sarext_startvalue,
                                   offsetonreverse=self.sarext_offsetonreverse,
                                   accelerationinitlong=self.sarext_accelerationinitlong,
                                   accelerationlong=self.sarext_accelerationlong,
                                   accelerationmaxlong=self.sarext_accelerationmaxlong,
                                   accelerationinitshort=self.sarext_accelerationinitshort,
                                   accelerationshort=self.sarext_accelerationshort,
                                   accelerationmaxshort=self.sarext_accelerationmaxshort)
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
        X['APO'] = talib.APO(X["Close"], fastperiod=self.apo_fastperiod, slowperiod=self.apo_slowperiod,
                             matype=self.apo_matype)
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
        X['PPO'] = talib.PPO(X["Close"], fastperiod=self.ppo_fastperiod, slowperiod=self.ppo_slowperiod,
                             matype=self.ppo_matype)
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

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'BitcoinTransformer_AutoSk',
                'name': 'BitcoinTransformer for Auto Sk Learn',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': False,
                'handles_multilabel': False,
                'is_deterministic': True,
                'input': (DENSE, UNSIGNED_DATA, SIGNED_DATA),
                'output': (DENSE, UNSIGNED_DATA, SIGNED_DATA)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()

        # base_estimator = Constant(name="base_estimator", value="None")
        sma_close_timeperiod = cs.add_hyperparameter(UniformIntegerHyperparameter(
            name="sma_close_timeperiod",lower=1, upper=100,))
        so_n = cs.add_hyperparameter(UniformFloatHyperparameter(
            name="so_n", lower=1, upper=100))


        '''
         self.sma_close_timeperiod = sma_close_timeperiod
        self.so_n = so_n
        self.so_d_n = so_d_n
        self.momentum_period = momentum_period
        '''
        return cs
