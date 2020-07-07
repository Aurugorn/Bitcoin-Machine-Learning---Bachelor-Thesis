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
    UNSIGNED_DATA, SPARSE, PREDICTIONS, INPUT
from autosklearn.util.common import check_none
import ta
import talib


# Create BitcoinTransformer_AutoSk component for auto-sklearn.
class BitcoinTransformer_AutoSk(AutoSklearnPreprocessingAlgorithm):
    def __init__(self,
                 random_state=None,
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
                 adosc_slowperiod=10):
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
        self.random_state = random_state
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

        self.preprocessor = None

    def fit(self, X, y=None):
        from BitcoinComponent import BitcoinTransformer
        self.preprocessor = \
            BitcoinTransformer(
                sma_close_timeperiod=self.sma_close_timeperiod,
                so_n=self.so_n,
                so_d_n=self.so_d_n,
                momentum_period=self.momentum_period,
                roc_period=self.roc_period,
                wr_lookback_period=self.wr_lookback_period,
                macd_period_longterm=self.macd_period_longterm,
                macd_period_shortterm=self.macd_period_shortterm,
                macd_period_to_signal=self.macd_period_to_signal,
                cci_periods=self.cci_periods,
                cci_constant=self.cci_constant,
                bb_periods=self.bb_periods,
                bb_n_factor_standard_dev=self.bb_n_factor_standard_dev,
                mean_o_c_period=self.mean_o_c_period,
                var_close_period=self.var_close_period,
                var_close_nbdev=self.var_close_nbdev,
                var_open_period=self.var_open_period,
                var_open_nbdev=self.var_open_nbdev,
                sma_high_period=self.sma_high_period,
                sma_low_period=self.sma_low_period,
                sma_handl_period=self.sma_handl_period,
                sma_h_l_c_o_period=self.sma_h_l_c_o_period,
                dema_period=self.dema_period,
                ema_period=self.ema_period,
                kama_period=self.kama_period,
                ma_period=self.ma_period,
                midpoint_period=self.midpoint_period,
                midprice_period=self.midprice_period,
                sar_acceleration=self.sar_acceleration,
                sar_maximum=self.sar_maximum,
                sarext_startvalue=self.sarext_startvalue,
                sarext_offsetonreverse=self.sarext_offsetonreverse,
                sarext_accelerationinitlong=self.sarext_accelerationinitlong,
                sarext_accelerationlong=self.sarext_accelerationlong,
                sarext_accelerationmaxlong=self.sarext_accelerationmaxlong,
                sarext_accelerationinitshort=self.sarext_accelerationinitshort,
                sarext_accelerationshort=self.sarext_accelerationshort,
                sarext_accelerationmaxshort=self.sarext_accelerationmaxshort,
                t3_period=self.t3_period,
                t3_vfactor=self.t3_vfactor,
                tema_period=self.tema_period,
                trima_period=self.trima_period,
                wma_period=self.wma_period,
                adx_period=self.adx_period,
                adxr_period=self.adxr_period,
                apo_fastperiod=self.apo_fastperiod,
                apo_slowperiod=self.apo_slowperiod,
                apo_matype=self.apo_matype,
                aroon_period=self.aroon_period,
                aroonosc_period=self.aroonosc_period,
                cmo_period=self.cmo_period,
                dx_period=self.dx_period,
                mfi_period=self.mfi_period,
                minus_di_period=self.minus_di_period,
                minus_dm_period=self.minus_dm_period,
                plus_di_period=self.plus_di_period,
                plus_dm_period=self.plus_dm_period,
                ppo_fastperiod=self.ppo_fastperiod,
                ppo_slowperiod=self.ppo_slowperiod,
                ppo_matype=self.ppo_matype,
                rocp_period=self.rocp_period,
                rocr_period=self.rocr_period,
                rocr100_period=self.rocr100_period,
                rsi_period=self.rsi_period,
                trix_period=self.trix_period,
                ultosc_period1=self.ultosc_period1,
                ultosc_period2=self.ultosc_period2,
                ultosc_period3=self.ultosc_period3,
                adosc_fastperiod=self.adosc_fastperiod,
                adosc_slowperiod=self.adosc_slowperiod
            )
        self.preprocessor.fit(X, y)
        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'BitcoinTransformer_AutoSk',
                'name': 'BitcoinTransformer for Auto Sk Learn',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': False,
                'handles_multilabel': False,
                'is_deterministic': True,
                'input': (SPARSE, DENSE, UNSIGNED_DATA),
                'output': (INPUT,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        sma_close_timeperiod = UniformIntegerHyperparameter("sma_close_timeperiod", lower=1, upper=1000, log=True)
        so_n = UniformIntegerHyperparameter("so_n", lower=1, upper=1000, log=True)
        so_d_n = UniformIntegerHyperparameter("so_d_n", lower=1, upper=1000, log=True)
        momentum_period = UniformIntegerHyperparameter("momentum_period", lower=1, upper=1000, log=True)
        roc_period = UniformIntegerHyperparameter("roc_period", lower=1, upper=1000, log=True)
        wr_lookback_period = UniformIntegerHyperparameter("wr_lookback_period", lower=1, upper=1000, log=True)
        macd_period_longterm = UniformIntegerHyperparameter("macd_period_longterm", lower=1, upper=1000, log=True)
        macd_period_shortterm = UniformIntegerHyperparameter("macd_period_shortterm", lower=1, upper=1000, log=True)
        macd_period_to_signal = UniformIntegerHyperparameter("macd_period_to_signal", lower=1, upper=1000, log=True)
        cci_periods = UniformIntegerHyperparameter("cci_periods", lower=1, upper=1000, log=True)
        cci_constant = UniformIntegerHyperparameter("cci_constant", lower=1, upper=1000, log=True)
        bb_periods = UniformIntegerHyperparameter("bb_periods", lower=1, upper=1000, log=True)
        bb_n_factor_standard_dev = UniformIntegerHyperparameter("bb_n_factor_standard_dev", lower=1, upper=1000,
                                                                log=True)
        mean_o_c_period = UniformIntegerHyperparameter("mean_o_c_period", lower=1, upper=1000, log=True)
        var_close_period = UniformIntegerHyperparameter("var_close_period", lower=1, upper=1000, log=True)
        var_close_nbdev = UniformIntegerHyperparameter("var_close_nbdev", lower=1, upper=1000, log=True)
        var_open_period = UniformIntegerHyperparameter("var_open_period", lower=1, upper=1000, log=True)
        var_open_nbdev = UniformIntegerHyperparameter("var_open_nbdev", lower=1, upper=1000, log=True)
        sma_high_period = UniformIntegerHyperparameter("sma_high_period", lower=1, upper=1000, log=True)
        sma_low_period = UniformIntegerHyperparameter("sma_low_period", lower=1, upper=1000, log=True)
        sma_handl_period = UniformIntegerHyperparameter("sma_handl_period", lower=1, upper=1000, log=True)
        sma_h_l_c_o_period = UniformIntegerHyperparameter("sma_h_l_c_o_period", lower=1, upper=1000, log=True)
        dema_period = UniformIntegerHyperparameter("dema_period", lower=1, upper=1000, log=True)
        ema_period = UniformIntegerHyperparameter("ema_period", lower=1, upper=1000, log=True)
        kama_period = UniformIntegerHyperparameter("kama_period", lower=1, upper=1000, log=True)
        ma_period = UniformIntegerHyperparameter("ma_period", lower=1, upper=1000, log=True)
        midpoint_period = UniformIntegerHyperparameter("midpoint_period", lower=1, upper=1000, log=True)
        midprice_period = UniformIntegerHyperparameter("midprice_period", lower=1, upper=1000, log=True)
        sar_acceleration = UniformIntegerHyperparameter("sar_acceleration", lower=1, upper=1000, log=True)
        sar_maximum = UniformIntegerHyperparameter("sar_maximum", lower=1, upper=1000, log=True)
        sarext_startvalue = UniformIntegerHyperparameter("sarext_startvalue", lower=1, upper=1000, log=True)
        sarext_offsetonreverse = UniformIntegerHyperparameter("sarext_offsetonreverse", lower=1, upper=1000, log=True)
        sarext_accelerationinitlong = UniformIntegerHyperparameter("sarext_accelerationinitlong", lower=1, upper=1000,
                                                                   log=True)
        sarext_accelerationlong = UniformIntegerHyperparameter("sarext_accelerationlong", lower=1, upper=1000, log=True)
        sarext_accelerationmaxlong = UniformIntegerHyperparameter("sarext_accelerationmaxlong", lower=1, upper=1000,
                                                                  log=True)
        sarext_accelerationinitshort = UniformIntegerHyperparameter("sarext_accelerationinitshort", lower=1, upper=1000,
                                                                    log=True)
        sarext_accelerationshort = UniformIntegerHyperparameter("sarext_accelerationshort", lower=1, upper=1000,
                                                                log=True)
        sarext_accelerationmaxshort = UniformIntegerHyperparameter("sarext_accelerationmaxshort", lower=1, upper=1000,
                                                                   log=True)
        t3_period = UniformIntegerHyperparameter("t3_period", lower=1, upper=1000, log=True)
        t3_vfactor = UniformIntegerHyperparameter("t3_vfactor", lower=1, upper=1000, log=True)
        tema_period = UniformIntegerHyperparameter("tema_period", lower=1, upper=1000, log=True)
        trima_period = UniformIntegerHyperparameter("trima_period", lower=1, upper=1000, log=True)
        wma_period = UniformIntegerHyperparameter("wma_period", lower=1, upper=1000, log=True)
        adx_period = UniformIntegerHyperparameter("adx_period", lower=1, upper=1000, log=True)
        adxr_period = UniformIntegerHyperparameter("adxr_period", lower=1, upper=1000, log=True)
        apo_fastperiod = UniformIntegerHyperparameter("apo_fastperiod", lower=1, upper=1000, log=True)
        apo_slowperiod = UniformIntegerHyperparameter("apo_slowperiod", lower=1, upper=1000, log=True)
        apo_matype = UniformIntegerHyperparameter("apo_matype", lower=1, upper=1000, log=True)
        aroon_period = UniformIntegerHyperparameter("aroon_period", lower=1, upper=1000, log=True)
        aroonosc_period = UniformIntegerHyperparameter("aroonosc_period", lower=1, upper=1000, log=True)
        cmo_period = UniformIntegerHyperparameter("cmo_period", lower=1, upper=1000, log=True)
        dx_period = UniformIntegerHyperparameter("dx_period", lower=1, upper=1000, log=True)
        mfi_period = UniformIntegerHyperparameter("mfi_period", lower=1, upper=1000, log=True)
        minus_di_period = UniformIntegerHyperparameter("minus_di_period", lower=1, upper=1000, log=True)
        minus_dm_period = UniformIntegerHyperparameter("minus_dm_period", lower=1, upper=1000, log=True)
        plus_di_period = UniformIntegerHyperparameter("plus_di_period", lower=1, upper=1000, log=True)
        plus_dm_period = UniformIntegerHyperparameter("plus_dm_period", lower=1, upper=1000, log=True)
        ppo_fastperiod = UniformIntegerHyperparameter("ppo_fastperiod", lower=1, upper=1000, log=True)
        ppo_slowperiod = UniformIntegerHyperparameter("ppo_slowperiod", lower=1, upper=1000, log=True)
        ppo_matype = UniformIntegerHyperparameter("ppo_matype", lower=1, upper=1000, log=True)
        rocp_period = UniformIntegerHyperparameter("rocp_period", lower=1, upper=1000, log=True)
        rocr_period = UniformIntegerHyperparameter("rocr_period", lower=1, upper=1000, log=True)
        rocr100_period = UniformIntegerHyperparameter("rocr100_period", lower=1, upper=1000, log=True)
        rsi_period = UniformIntegerHyperparameter("rsi_period", lower=1, upper=1000, log=True)
        trix_period = UniformIntegerHyperparameter("trix_period", lower=1, upper=1000, log=True)
        ultosc_period1 = UniformIntegerHyperparameter("ultosc_period1", lower=1, upper=1000, log=True)
        ultosc_period2 = UniformIntegerHyperparameter("ultosc_period2", lower=1, upper=1000, log=True)
        ultosc_period3 = UniformIntegerHyperparameter("ultosc_period3", lower=1, upper=1000, log=True)
        adosc_fastperiod = UniformIntegerHyperparameter("adosc_fastperiod", lower=1, upper=1000, log=True)
        adosc_slowperiod = UniformIntegerHyperparameter("adosc_slowperiod", lower=1, upper=1000, log=True)
        cs = ConfigurationSpace()
        cs.add_hyperparameters([sma_close_timeperiod,
                                so_n,
                                so_d_n,
                                momentum_period,
                                roc_period,
                                wr_lookback_period,
                                macd_period_longterm,
                                macd_period_shortterm,
                                macd_period_to_signal,
                                cci_periods,
                                cci_constant,
                                bb_periods,
                                bb_n_factor_standard_dev,
                                mean_o_c_period,
                                var_close_period,
                                var_close_nbdev,
                                var_open_period,
                                var_open_nbdev,
                                sma_high_period,
                                sma_low_period,
                                sma_handl_period,
                                sma_h_l_c_o_period,
                                dema_period,
                                ema_period,
                                kama_period,
                                ma_period,
                                midpoint_period,
                                midprice_period,
                                sar_acceleration,
                                sar_maximum,
                                sarext_startvalue,
                                sarext_offsetonreverse,
                                sarext_accelerationinitlong,
                                sarext_accelerationlong,
                                sarext_accelerationmaxlong,
                                sarext_accelerationinitshort,
                                sarext_accelerationshort,
                                sarext_accelerationmaxshort,
                                t3_period,
                                t3_vfactor,
                                tema_period,
                                trima_period,
                                wma_period,
                                adx_period,
                                adxr_period,
                                apo_fastperiod,
                                apo_slowperiod,
                                apo_matype,
                                aroon_period,
                                aroonosc_period,
                                cmo_period,
                                dx_period,
                                mfi_period,
                                minus_di_period,
                                minus_dm_period,
                                plus_di_period,
                                plus_dm_period,
                                ppo_fastperiod,
                                ppo_slowperiod,
                                ppo_matype,
                                rocp_period,
                                rocr_period,
                                rocr100_period,
                                rsi_period,
                                trix_period,
                                ultosc_period1,
                                ultosc_period2,
                                ultosc_period3,
                                adosc_fastperiod,
                                adosc_slowperiod])

        return cs
