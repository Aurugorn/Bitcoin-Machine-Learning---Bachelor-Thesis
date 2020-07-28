import json
import sys, getopt
import numpy
from BitcoinComponent import BitcoinTransformer
from BitcoinComponent import ParameterRelationsBTCTrans
import pandas as pd
import numpy as np
import ta
import time

from datetime import datetime
from datetime import date
import ntpath

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.fixes import loguniform
from sklearn.impute import SimpleImputer
from sklearn import metrics
from sklearn.dummy import DummyRegressor
from sklearn.dummy import DummyClassifier
import matplotlib
import matplotlib.pyplot as plt
import ta
from collections import Counter

def csvToDF(input="", dropna=True, timeOutput=False):
    """
    @param input: dataset
    @type input: csv
    @param dropna: drop na rows
    @type dropna: boolean
    @param timeOutput: print time to perform operation
    @type timeOutput: boolean
    @return: converted csv to dataframe
    @rtype: pandas dataframe
    """
    # Load the input
    if timeOutput:
        logInfo = 'Reading input...'
        print(logInfo)
        start_time = time.time()
    df = pd.read_csv(input, encoding="ISO-8859-1")
    if dropna:
        # Clean NaN values
        df = ta.utils.dropna(df)

    df.index.names = ['index']
    if timeOutput:
        end_time = time.time()
        logInfo = 'Input read, took ' + str(end_time - start_time) + ' seconds'
        print(logInfo)
    df = df.reset_index()
    return df


def datetimeToTimestamp(input="", Date="01/01/1970", Time="00:00:00", Format="%m/%d/%Y %H:%M:%S", timeOutput=False):
    """
    @param input: dataset to be modified
    @type input: pandas dataframe
    @param Date: "Date" column in dataframe
    @type Date: column of pandas dataframe
    @param Time: "Time" column in dataframe
    @type Time: column of pandas dataframe
    @param Format: @param Date + ' ' @param Time datetime input format
    @type Format: string
    @param timeOutput: print time to perform operation
    @type timeOutput: boolean
    @return: transformed dataframe that now has a timestamp column
    @rtype: pandas dataframe
    """
    """Reads a pandas dataframe as input. Takes 3 parameters to format datetime to timestamp
        """
    if timeOutput:
        logInfo = 'Generating Timestamp Column...'
        print(logInfo)
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
        logInfo = 'Time to create Timestamp: ' + str(end_time - start_time) + ' seconds'
        print(logInfo)
    return input


def generateTarget(df="", method="Classification", typeRegression="Difference", intervalPeriods=1):
    """
    @param df: dataset
    @type df: pandas dataframe
    @param method: Type of machine learning to generate the 'Target' column
                    if method equals 'Regression' additional parameters can be set for delta price calculations (typeRegression, 'intervalPeriods')
                    Options available: "Classification", "Regression"
    @type method: string
    @param typeRegression: handles the "Target" column value
                    'Difference' : Target column = the difference between the Close price current interval and 'intervalPeriods' further as a float
                    'Percentage' : Target column = the difference between the Close price current interval and 'intervalPeriods' further in percentages
                    'ExactPrice' : Target column = the Close price of the current interval
                    Options available: "Difference", "Percentage", "ExactPrice"
    @type typeRegression: string
    @param intervalPeriods: How much intervals do we need to look in the future to calculate the difference for the "Target" column.
                    Condition: > 0
    @type intervalPeriods: integer
    @return: Target column, length of dataset
    @rtype: pandas dataframe consisting of 1 column
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
        # Add new column 'Target' . True if Close price from this timestamp is higher than Close price of previous timestamp
        #df['Target'] = np.where(df.Close.shift(-1) > df.Close, True, False)
        df['Target'] = np.where(df.Close.shift(-1)* (1-(0.10/100)) > df.Close, True, False)
    elif method == "Regression":
        if typeRegression == "Difference":
            df['Target'] = df.Close.shift(-intervalPeriods) - df.Close
        elif typeRegression == "Percentage":
            dfclose = df['Close']
            df['Target'] = -dfclose.pct_change(periods=-intervalPeriods)
        elif typeRegression == "ExactPrice":
            df['Target'] = df.Close.shift(-intervalPeriods)

    return df['Target']


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            logInfo = "Model with rank: {0}".format(i)
            print(logInfo)
            logInfo = "Mean validation score: {0:.3f} (std: {1:.3f})".format(results['mean_test_score'][candidate],
                                                                             results['std_test_score'][candidate])
            print(logInfo)
            logInfo = "Parameters: {0}".format(results['params'][candidate])
            print(logInfo)
            logInfo = ""
            print(logInfo)


# Evaluate base model compared to randomizedSearchCV model
def evaluate(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = round(accuracy_score(y_test, predictions) * 100, 2)
    print('Accuracy = {:0.2f}%.'.format(accuracy))

    return accuracy


def getHyperParamBTC(typeofrun=1):
    if typeofrun == 1:
        # BitcoinTransformer
        param_dist = {'classify__max_depth': [80, 90, 100, 110],
                      'classify__max_features': [2, 3],
                      'classify__min_samples_leaf': [3, 4, 5],
                      'classify__min_samples_split': [8, 10, 12],
                      'classify__n_estimators': [100, 200, 300, 1000],
                      'BitcoinTransformer__adosc_fastperiod': loguniform(2, 100),
                      'BitcoinTransformer__adosc_slowperiod': loguniform(2, 100),
                      'BitcoinTransformer__adx_period': loguniform(2, 100),
                      'BitcoinTransformer__adxr_period': loguniform(2, 100),
                      'BitcoinTransformer__apo_fastperiod': loguniform(2, 100),
                      'BitcoinTransformer__apo_slowperiod': loguniform(2, 100),
                      'BitcoinTransformer__aroon_period': loguniform(2, 100),
                      'BitcoinTransformer__aroonosc_period': loguniform(2, 100),
                      'BitcoinTransformer__bb_periods': loguniform(2, 100),
                      'BitcoinTransformer__cci_periods': loguniform(2, 100),
                      'BitcoinTransformer__cmo_period': loguniform(2, 100),
                      'BitcoinTransformer__dema_period': loguniform(2, 100),
                      'BitcoinTransformer__dx_period': loguniform(2, 100),
                      'BitcoinTransformer__ema_period': loguniform(2, 100),
                      'BitcoinTransformer__kama_period': loguniform(2, 100),
                      'BitcoinTransformer__ma_period': loguniform(2, 100),
                      'BitcoinTransformer__macd_period_longterm': loguniform(2, 100),
                      'BitcoinTransformer__macd_period_shortterm': loguniform(2, 100),
                      'BitcoinTransformer__macd_period_to_signal': loguniform(2, 100),
                      'BitcoinTransformer__mean_o_c_period': loguniform(2, 100),
                      'BitcoinTransformer__mfi_period': loguniform(2, 100),
                      'BitcoinTransformer__midpoint_period': loguniform(2, 100),
                      'BitcoinTransformer__midprice_period': loguniform(2, 100),
                      'BitcoinTransformer__minus_di_period': loguniform(2, 100),
                      'BitcoinTransformer__minus_dm_period': loguniform(2, 100),
                      'BitcoinTransformer__momentum_period': loguniform(2, 100),
                      'BitcoinTransformer__plus_di_period': loguniform(2, 100),
                      'BitcoinTransformer__plus_dm_period': loguniform(2, 100),
                      'BitcoinTransformer__ppo_fastperiod': loguniform(2, 100),
                      'BitcoinTransformer__ppo_slowperiod': loguniform(2, 100),
                      'BitcoinTransformer__roc_period': loguniform(2, 100),
                      'BitcoinTransformer__rocp_period': loguniform(2, 100),
                      'BitcoinTransformer__rocr100_period': loguniform(2, 100),
                      'BitcoinTransformer__rocr_period': loguniform(2, 100),
                      'BitcoinTransformer__rsi_period': loguniform(2, 100),
                      'BitcoinTransformer__sar_acceleration': loguniform(2, 100),
                      'BitcoinTransformer__sar_maximum': loguniform(2, 100),
                      'BitcoinTransformer__sarext_accelerationinitlong': loguniform(2, 100),
                      'BitcoinTransformer__sarext_accelerationinitshort': loguniform(2, 100),
                      'BitcoinTransformer__sarext_accelerationlong': loguniform(2, 100),
                      'BitcoinTransformer__sarext_accelerationmaxlong': loguniform(2, 100),
                      'BitcoinTransformer__sarext_accelerationmaxshort': loguniform(2, 100),
                      'BitcoinTransformer__sarext_accelerationshort': loguniform(2, 100),
                      'BitcoinTransformer__sarext_offsetonreverse': loguniform(2, 100),
                      'BitcoinTransformer__sarext_startvalue': loguniform(2, 100),
                      'BitcoinTransformer__sma_close_timeperiod': loguniform(2, 100),
                      'BitcoinTransformer__sma_h_l_c_o_period': loguniform(2, 100),
                      'BitcoinTransformer__sma_handl_period': loguniform(2, 100),
                      'BitcoinTransformer__sma_high_period': loguniform(2, 100),
                      'BitcoinTransformer__sma_low_period': loguniform(2, 100),
                      'BitcoinTransformer__so_d_n': loguniform(2, 100),
                      'BitcoinTransformer__so_n': loguniform(2, 100),
                      'BitcoinTransformer__t3_period': loguniform(2, 100),
                      'BitcoinTransformer__tema_period': loguniform(2, 100),
                      'BitcoinTransformer__trima_period': loguniform(2, 100),
                      'BitcoinTransformer__trix_period': loguniform(2, 100),
                      'BitcoinTransformer__ultosc_period1': loguniform(2, 100),
                      'BitcoinTransformer__ultosc_period2': loguniform(2, 100),
                      'BitcoinTransformer__ultosc_period3': loguniform(2, 100),
                      'BitcoinTransformer__var_close_period': loguniform(2, 100),
                      'BitcoinTransformer__var_open_period': loguniform(2, 100),
                      'BitcoinTransformer__wma_period': loguniform(2, 100),
                      'BitcoinTransformer__wr_lookback_period': loguniform(2, 100)
                      }
    elif typeofrun == 2:
        # ParameterRelationsBTCTrans
        param_dist = {'classify__max_depth': [80, 90, 100, 110],
                      'classify__max_features': [2, 3],
                      'classify__min_samples_leaf': [3, 4, 5],
                      'classify__min_samples_split': [8, 10, 12],
                      'classify__n_estimators': [100, 200, 300, 1000],
                      'BitcoinTransformer__fastperiod': loguniform(2, 100),
                      'BitcoinTransformer__longterm': loguniform(2, 100),
                      'BitcoinTransformer__midterm': loguniform(2, 100),
                      'BitcoinTransformer__shortterm': loguniform(2, 100),
                      'BitcoinTransformer__bb_cci': loguniform(2, 100),
                      'BitcoinTransformer__var_t3': loguniform(2, 100),
                      'BitcoinTransformer__dema_trema': loguniform(2, 100),
                      'BitcoinTransformer__zero': loguniform(2, 100),
                      'BitcoinTransformer__rocperiod': loguniform(2, 100)}
    elif typeofrun == 3:
        # Only Random Forest hyperparameters
        param_dist = {'classify__max_depth': [80, 90, 100, 110],
                      'classify__max_features': [2, 3],
                      'classify__min_samples_leaf': [3, 4, 5],
                      'classify__min_samples_split': [8, 10, 12],
                      'classify__n_estimators': [100, 200, 300, 1000],
                      }
    return param_dist

# Read input txt file, and read line by line the accuracy for classification or MAE for regression
# Classification accuracy input syntax: every line a new accuracy
# Regression MAE input syntax: every line first string seperated by space a new MAE
def readTxtResults(input):
    DEFAULT = []
    with open(input, encoding='utf-8-sig') as fp:
        line = fp.readline()
        cnt = 1
        while line:
            DEFAULT.append(float(line.strip().rsplit(' ')[0]))
            line = fp.readline()
            cnt += 1
    fp.close()
    return DEFAULT


# helper functions to calculate multiple baselines for classification & regression
def calculateBaselines(df, target, intervalType, MachineLearningMethod):
    X = df[["Open", "Low", "High", "Close", "Volume"]]
    X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2, train_size=0.8, shuffle=False,
                                                        random_state=42)

    # BASELINES FOR REGRESSION
    if MachineLearningMethod == "Regression":
        print('------------- No-Change baseline: Predict same Target as last timestamp ------------- ')
        # No-Change baseline
        # Add new column 'Predicted' . True if previous timestamp was also true
        df['Predicted'] = target.shift(1)
        y_predict_NOCHANGE = df['Predicted'][len(X_train):]
        MAE_nochange = metrics.mean_absolute_error(y_test, y_predict_NOCHANGE)
        print('Mean Absolute Error:', MAE_nochange)
        print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_predict_NOCHANGE))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_predict_NOCHANGE)))

        print('------------- DUMMY BASELINE MODEL “mean”: always predicts the mean of the training set ------------- ')
        # Dummy classifier to create baseline to compare to the real models
        dummy_clf = DummyRegressor(strategy="mean")
        dummy_clf_ = dummy_clf.fit(X_train, y_train)
        y_predict_MEAN = dummy_clf_.predict(X_test)

        MAE_mean = metrics.mean_absolute_error(y_test, y_predict_MEAN)
        print('Mean Absolute Error:', MAE_mean)
        print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_predict_MEAN))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_predict_MEAN)))

        print(
            '------------- DUMMY BASELINE MODEL “median”: always predicts the median of the training set ------------- ')
        dummy_clf = DummyRegressor(strategy="median")
        dummy_clf_ = dummy_clf.fit(X_train, y_train)
        y_predict_median = dummy_clf.predict(X_test)

        MAE_median = metrics.mean_absolute_error(y_test, y_predict_median)
        print('Mean Absolute Error:', MAE_median)
        print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_predict_median))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_predict_median)))
    # BASELINES FOR CLASSIFICATION
    elif MachineLearningMethod == "Classification":
        print('------------- No-Change baseline: Predict same Target as last timestamp ------------- ')
        # No-Change baseline
        # Add new column 'Predicted' . True if previous timestamp was also true
        df['Predicted'] = np.where(target.shift(1) == True, True, False)
        y_predict_NOCHANGE = df['Predicted'][len(X_train):]
        print('Accuracy score:')
        accuracy_nochange = round(accuracy_score(y_test, y_predict_NOCHANGE) * 100, 2)
        print(accuracy_nochange)
        print('Classification report:')
        print(classification_report(y_test, y_predict_NOCHANGE))

        print(
            '------------- DUMMY BASELINE MODEL (Stratified) generates predictions by respecting the training set’s class distribution. Random ------------- ')
        # Dummy classifier to create baseline to compare to the real models
        dummy_clf = DummyClassifier(strategy="stratified", random_state=42)
        dummy_clf_ = dummy_clf.fit(X_train, y_train)
        y_predict_STRATIFIED = dummy_clf.predict(X_test)

        print('Accuracy score:')
        accuracy_stratified = round(accuracy_score(y_test, y_predict_STRATIFIED) * 100, 2)
        print(accuracy_stratified)
        print('Classification report:')
        print(classification_report(y_test, y_predict_STRATIFIED))

        print(
            '------------- DUMMY BASELINE MODEL (Most Frequent)  always predicts the most frequent label in the training set. ------------- ')
        dummy_clf = DummyClassifier(strategy="most_frequent", random_state=42)
        dummy_clf_ = dummy_clf.fit(X_train, y_train)
        y_predict_MOSTFREQ = dummy_clf.predict(X_test)

        print('Accuracy score:')
        accuracy_mostfrequent = round(accuracy_score(y_test, y_predict_MOSTFREQ) * 100, 2)
        print(accuracy_mostfrequent)
        print('Classification report:')
        print(classification_report(y_test, y_predict_MOSTFREQ))

        print(
            '------------- DUMMY BASELINE MODEL (Prior) always predicts the class that maximizes the class prior (like “most_frequent”) and predict_proba returns the class prior. ------------- ')
        dummy_clf = DummyClassifier(strategy="prior", random_state=42)
        dummy_clf_ = dummy_clf.fit(X_train, y_train)
        y_predict_PRIOR = dummy_clf.predict(X_test)
        print('Accuracy score:')
        accuracy_prior = round(accuracy_score(y_test, y_predict_PRIOR) * 100, 2)
        print(accuracy_prior)
        print('Classification report:')
        print(classification_report(y_test, y_predict_PRIOR))

    # Build Matplot boxplot from results of baselines + results of all our own modelssssssssssssssssss
    if MachineLearningMethod == "Regression":
        if intervalType == "Day":
            DEFAULT = readTxtResults('../accuraciesOutput/default/BTCUSD_1Day.csv_Regression_Difference_FALSE_RELATIONS.txt')
            RF_OPTIMIZED = readTxtResults('../accuraciesOutput/rf_optimized/BTCUSD_1Day.csv_Regression_Difference_FALSE_RELATIONS.txt')
            NO_RELATIONS = readTxtResults('../accuraciesOutput/no_relations/BTCUSD_1Day.csv_Regression_Difference_FALSE_RELATIONS.txt')
            RELATIONS = readTxtResults('../accuraciesOutput/relations/BTCUSD_1Day.csv_Regression_Difference_TRUE_RELATIONS.txt')
        elif intervalType == "Hour":
            DEFAULT = readTxtResults('../accuraciesOutput/default/bitfinex_tBTCUSD_1h.csv_Regression_Difference_FALSE_RELATIONS.txt')
            RF_OPTIMIZED = readTxtResults('../accuraciesOutput/rf_optimized/bitfinex_tBTCUSD_1h.csv_Regression_Difference_FALSE_RELATIONS.txt')
            NO_RELATIONS = readTxtResults('../accuraciesOutput/no_relations/bitfinex_tBTCUSD_1h.csv_Regression_Difference_FALSE_RELATIONS.txt')
            RELATIONS = readTxtResults('../accuraciesOutput/relations/bitfinex_tBTCUSD_1h.csv_Regression_Difference_TRUE_RELATIONS.txt')
        elif intervalType == "Minute":
            DEFAULT = readTxtResults('../accuraciesOutput/default/bitfinex_tBTCUSD_1m.csv_Regression_Difference_FALSE_RELATIONS.txt')
            RF_OPTIMIZED = readTxtResults('../accuraciesOutput/rf_optimized/bitfinex_tBTCUSD_1m.csv_Regression_Difference_FALSE_RELATIONS.txt')
            NO_RELATIONS = readTxtResults('../accuraciesOutput/no_relations/bitfinex_tBTCUSD_1m.csv_Regression_Difference_FALSE_RELATIONS.txt')
            RELATIONS = readTxtResults('../accuraciesOutput/relations/bitfinex_tBTCUSD_1m.csv_Regression_Difference_TRUE_RELATIONS.txt')
        xLabelNames = ['Default', 'RF Optimized', '75 Hyp..', '14 Hyp..',
                       'No-change', 'Mean', 'Median']
        data = [DEFAULT, RF_OPTIMIZED, NO_RELATIONS, RELATIONS, [MAE_nochange], [MAE_mean], [MAE_median]]
    elif MachineLearningMethod == "Classification":
        if intervalType == "Day":
            DEFAULT = readTxtResults('../accuraciesOutput/default/BTCUSD_1Day.csv_Classification_FALSE_RELATIONS.txt')
            RF_OPTIMIZED = readTxtResults('../accuraciesOutput/rf_optimized/BTCUSD_1Day.csv_Classification_FALSE_RELATIONS.txt')
            NO_RELATIONS = readTxtResults('../accuraciesOutput/no_relations/BTCUSD_1Day.csv_Classification_FALSE_RELATIONS.txt')
            RELATIONS = readTxtResults('../accuraciesOutput/relations/BTCUSD_1Day.csv_Classification_TRUE_RELATIONS.txt')
        elif intervalType == "Hour":
            DEFAULT = readTxtResults('../accuraciesOutput/default/bitfinex_tBTCUSD_1h.csv_Classification_FALSE_RELATIONS.txt')
            RF_OPTIMIZED = readTxtResults('../accuraciesOutput/rf_optimized/bitfinex_tBTCUSD_1h.csv_Classification_FALSE_RELATIONS.txt')
            NO_RELATIONS = readTxtResults('../accuraciesOutput/no_relations/bitfinex_tBTCUSD_1h.csv_Classification_FALSE_RELATIONS.txt')
            RELATIONS = readTxtResults('../accuraciesOutput/relations/bitfinex_tBTCUSD_1h.csv_Classification_TRUE_RELATIONS.txt')
        elif intervalType == "Minute":
            DEFAULT = readTxtResults('../accuraciesOutput/default/bitfinex_tBTCUSD_1m.csv_Classification_FALSE_RELATIONS.txt')
            RF_OPTIMIZED = readTxtResults('../accuraciesOutput/rf_optimized/bitfinex_tBTCUSD_1m.csv_Classification_FALSE_RELATIONS.txt')
            NO_RELATIONS = readTxtResults('../accuraciesOutput/no_relations/bitfinex_tBTCUSD_1m.csv_Classification_FALSE_RELATIONS.txt')
            RELATIONS = readTxtResults('../accuraciesOutput/relations/bitfinex_tBTCUSD_1m.csv_Classification_TRUE_RELATIONS.txt')
        xLabelNames = ['Default', 'RF Optimized', '75 Hyp..', '14 Hyp..',
                       'No-change', 'Stratified', 'Most Frequent', 'Prior']
        data = [DEFAULT, RF_OPTIMIZED, NO_RELATIONS, RELATIONS, [accuracy_nochange], [accuracy_stratified],
                [accuracy_mostfrequent], [accuracy_prior]]

    # Export the predicted model
    df = df[len(X_train):]
    if MachineLearningMethod == "Classification":
        df['Predicted'] = y_predict_MOSTFREQ
    elif MachineLearningMethod == "Regression":
        df['Predicted'] = y_predict_MEAN
    dfClose = df['Close']
    df['Change'] = dfClose.pct_change(periods=1)  # Contains percentage change
    df = df[['Timestamp', 'Close', 'Change', 'Target', 'Predicted']]
    nameOfExportedModel = intervalType + "_" + MachineLearningMethod
    df.to_csv("../PredictedModels/baselineModels/" + nameOfExportedModel + ".csv")

    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
    })

    fig, ax = plt.subplots()
    ax.set_xticklabels(
        xLabelNames
        )
    ax.set_title(MachineLearningMethod + ' Baselines for interval: ' + intervalType)
    ax.boxplot(data)
    plt.savefig('histogram.pgf')
    plt.show()
    exit()


def main(argv):
    RandomState = ''
    Slurm_Task_idx = ''
    CSV_Path = ''
    MachineLearningMethod = ''
    typeRegression = ''
    intervalPeriods = ''
    RunType = ''
    RandomizedSearchCV_Status = ''
    RandomizedSearchCV_n_iter_search = ''
    RandomForest_n_estimators = ''
    shortenDataset = ''

    try:
        opts, args = getopt.getopt(argv, "hx:",
                                   ["task_idx="])
    except getopt.GetoptError:
        print(
            'AliceTasks.py --task_idx <$SLURM_ARRAY_TASK_ID>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(
                'AliceTasks.py --task_idx <$SLURM_ARRAY_TASK_ID>')
            sys.exit()
        elif opt in ("-x", "--task_idx"):
            Slurm_Task_idx = int(arg)

    # Default parameters
    if typeRegression == '':
        typeRegression = "Difference"
    if intervalPeriods == '':
        intervalPeriods = 1

    dateTimeFormat = '%m/%d/%Y %H:%M:%S'  # 1h & 1m dataset
    # Run program 180 times with different variables
    RandomState = Slurm_Task_idx % 10
    TypeOfTest = int(Slurm_Task_idx / 60)  # 0 / 60 = 0 , 59/60  = 0.99, 179/60 = 2.99
    # TypeOfTest = 0 --> Default settings, No Relations, Relations
    # TypeOfTest = 1 --> No Relations
    # TypeOfTest = 2 --> Relations
    # TypeOfTest = 3 --> Only Random Forest optimized hyperparameters
    MLSettings = int((Slurm_Task_idx / 10) % 6)  # 0 = 0, 10 / 10 = 1 % 6 = 1,

    #0 - 59
    if (TypeOfTest == 0):
        # Default settings, no optimalization
        RandomizedSearchCV_Status = "OFF"
        RunType = 1
        settingParams = "default"
    #60 - 119
    elif (TypeOfTest == 1):
        # No relations, hyperparameters optimalization
        RandomizedSearchCV_Status = "ON"
        RunType = 1  # relations off
        settingParams = "norelations"
    #120 - 179
    elif (TypeOfTest == 2):
        # Relations, hyperparameters optimalization
        RandomizedSearchCV_Status = "ON"
        RunType = 2  # relations on
        settingParams = "relations"
    #180 - 239
    elif (TypeOfTest == 3):
        # Only optimize 5 hyperparameters from Random Forest. DEFAULT Technical Indicator hyperparameters
        RandomizedSearchCV_Status = "ON"
        RunType = 1  # relations on
        settingParams = "rfoptimized"

    if (MLSettings == 0):
        CSV_Path = "../inputs/BTCUSD_1Day.csv"
        dateTimeFormat = '%Y-%m-%d %H:%M:%S'  # 1d dataset
        MachineLearningMethod = "Classification"
        RandomForest_n_estimators = 128
        RandomizedSearchCV_n_iter_search = 300
        shortenDataset = False
    elif (MLSettings == 1):
        CSV_Path = "../inputs/BTCUSD_1Day.csv"
        dateTimeFormat = '%Y-%m-%d %H:%M:%S'  # 1d dataset
        MachineLearningMethod = "Regression"
        RandomForest_n_estimators = 128
        RandomizedSearchCV_n_iter_search = 300
        shortenDataset = False
    elif (MLSettings == 2):
        CSV_Path = "../inputs/bitfinex_tBTCUSD_1h.csv"
        dateTimeFormat = '%m/%d/%Y %H:%M:%S'
        MachineLearningMethod = "Classification"
        RandomForest_n_estimators = 128
        RandomizedSearchCV_n_iter_search = 300
        shortenDataset = False
    elif (MLSettings == 3):
        CSV_Path = "../inputs/bitfinex_tBTCUSD_1h.csv"
        dateTimeFormat = '%m/%d/%Y %H:%M:%S'
        MachineLearningMethod = "Regression"
        RandomForest_n_estimators = 128
        RandomizedSearchCV_n_iter_search = 300
        shortenDataset = 'year'
    elif (MLSettings == 4):
        CSV_Path = "../inputs/bitfinex_tBTCUSD_1m.csv"
        dateTimeFormat = '%m/%d/%Y %H:%M:%S'
        MachineLearningMethod = "Classification"
        RandomForest_n_estimators = 128
        RandomizedSearchCV_n_iter_search = 300
        shortenDataset = 'year'
    elif (MLSettings == 5):
        CSV_Path = "../inputs/bitfinex_tBTCUSD_1m.csv"
        dateTimeFormat = '%m/%d/%Y %H:%M:%S'
        MachineLearningMethod = "Regression"
        RandomForest_n_estimators = 128
        RandomizedSearchCV_n_iter_search = 100
        shortenDataset = 'year'

    # For calculating all baselines
    # Regression
    if Slurm_Task_idx == -1:
        CSV_Path = "../inputs/BTCUSD_1Day.csv"
        dateTimeFormat = '%Y-%m-%d %H:%M:%S'  # 1d dataset
        MachineLearningMethod = "Regression"
        shortenDataset = False
        intervalType = "Day"
    elif Slurm_Task_idx == -2:
        CSV_Path = "../inputs/bitfinex_tBTCUSD_1h.csv"
        dateTimeFormat = '%m/%d/%Y %H:%M:%S'
        MachineLearningMethod = "Regression"
        shortenDataset = 'year'
        intervalType = "Hour"
    elif Slurm_Task_idx == -3:
        CSV_Path = "../inputs/bitfinex_tBTCUSD_1m.csv"
        dateTimeFormat = '%m/%d/%Y %H:%M:%S'
        MachineLearningMethod = "Regression"
        shortenDataset = 'month'
        intervalType = "Minute"
    #Classification
    elif Slurm_Task_idx == -4:
        CSV_Path = "../inputs/BTCUSD_1Day.csv"
        dateTimeFormat = '%Y-%m-%d %H:%M:%S'  # 1d dataset
        MachineLearningMethod = "Classification"
        shortenDataset = False
        intervalType = "Day"
    elif Slurm_Task_idx == -5:
        CSV_Path = "../inputs/bitfinex_tBTCUSD_1h.csv"
        dateTimeFormat = '%m/%d/%Y %H:%M:%S'
        MachineLearningMethod = "Classification"
        shortenDataset = False
        intervalType = "Hour"
    elif Slurm_Task_idx == -6:
        CSV_Path = "../inputs/bitfinex_tBTCUSD_1m.csv"
        dateTimeFormat = '%m/%d/%Y %H:%M:%S'
        MachineLearningMethod = "Classification"
        shortenDataset = 'year'
        intervalType = "Minute"

    # Set Variables
    # CSV_Path = read dataframe input open,low,high,close
    # MachineLearningMethod = Classification, Regression
    # typeRegression = handles the "Target" column value
    #                     'Difference' : Target column = the difference between the Close price current interval and 'intervalPeriods' further as a float
    #                     'Percentage' : Target column = the difference between the Close price current interval and 'intervalPeriods' further in percentages
    #                     'ExactPrice' : Target column = the Close price of the current interval
    #                     Options available: "Difference", "Percentage", "ExactPrice"
    # intervalPeriods = If MachineLearningMethod == "Regression" --> How many intervals do we need to look back for target generation?
    # RunType = 1. Pipeline --> BitcoinTransformer()
    #           2. Pipeline --> ParameterRelationsBTCTrans()
    # RandomizedSearchCV_Status = ON,OFF whether to use RandomizedSearchCV
    # RandomizedSearchCV_n_iter_search = 1
    # RandomForest_n_estimators = 100

    relationsHyperparameters = "UNSET"
    if RunType == 1:
        relationsHyperparameters = "FALSE_RELATIONS"
    elif RunType == 2:
        relationsHyperparameters = "TRUE_RELATIONS"

    # ------------------------------------- Workflow, preparing dataset ~ results of models -------------------------------------#
    # 1. Load CSV
    # 2. X = Open, Low, High, Close and Volume columns with data
    # 3. target = target column, calculate with generateTarget()
    # 4. Train test split
    # 5. Pipeline
    # 6. Fit & Predict
    # 7. Results

    # 1. Load CSV
    # __________________________
    df = csvToDF(CSV_Path, dropna=True, timeOutput=False)

    # Check if we want only 2019 and 2020 for training and testing
    # Why? Because rows from 2013 - 2020 is way too much to RandomizedSearchCV
    # Shorten dataset
    if shortenDataset != False:
        if shortenDataset == 'year':
            df1 = df.loc[df['Date'].str.contains('2019')]
            df2 = df.loc[df['Date'].str.contains('2020')]
            frames = [df1, df2]
            df = pd.concat(frames)
        elif shortenDataset == 'month':
            df = df.loc[df['Date'].str.contains('2020')]

    # (optional) Convert datetime to timestamp
    df = datetimeToTimestamp(input=df, Date=df['Date'], Time=df['Time'], Format=dateTimeFormat, timeOutput=False)
    # 2. Prepare X
    # __________________________
    X = df[["Open", "Low", "High", "Close", "Volume"]]
    # 3. Prepare target
    # __________________________
    target = generateTarget(df, method=MachineLearningMethod, typeRegression=typeRegression,
                            intervalPeriods=intervalPeriods)
    '''
    print(target)
    print('Amount of target False/True occurences: {}'.format(Counter(target)))
    print(X)
    exit()
    '''
    
    # Set last row of target on FALSE or 0
    target.at[target.index[-1]] = 0

    # (optional): Calculate baselines
    # Note: Program stops after calculating baselines
    if Slurm_Task_idx < 0:
        calculateBaselines(df, target, intervalType, MachineLearningMethod)

    # 4. Train test split
    # __________________________

    X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2, train_size=0.8, shuffle=False,
                                                        random_state=RandomState)

    # Configuration space.
    # cs = BitcoinTransformer_AutoSk.get_hyperparameter_search_space()
    # print(cs)
    # 5. Start the pipeline
    # __________________________

    if MachineLearningMethod == "Classification":
        RandomForestClas_Reg = RandomForestClassifier(random_state=RandomState)
        scoring_method = 'accuracy'
    elif MachineLearningMethod == "Regression":
        RandomForestClas_Reg = RandomForestRegressor(random_state=RandomState)
        scoring_method = 'neg_mean_absolute_error'

    if RunType == 1:
        pipeline_class = Pipeline(steps=[
            ('imputing', SimpleImputer(missing_values=np.nan, strategy='mean')),
            ('classify', RandomForestClas_Reg)
        ])
    elif RunType == 2:
        pipeline_class = Pipeline(steps=[
            ('imputing', SimpleImputer(missing_values=np.nan, strategy='mean')),
            ('classify', RandomForestClas_Reg)
        ])

    # Auto Sk Learn Classifier Example: pipeline = classifier.AutoSklearnClassifier(
    # time_left_for_this_task=30, per_run_time_limit=5
    #     ,include_estimators=["random_forest"]
    #     ,ml_memory_limit=6000
    #         ,include_preprocessors=['BitcoinTransformer_AutoSk'],exclude_preprocessors=None
    # )

    # Gets all hyperparameter dist for randomizedSearchCV
    if TypeOfTest == 3:
        # Only optimize 5 hyperparameters from Random Forest. DEFAULT Technical Indicator hyperparameters
        param_dist = getHyperParamBTC(3)
    else:
        # Just get all hyperparamers from technical indicators
        param_dist = getHyperParamBTC(RunType)

    # All possible parameters for the pipeline
    # print(pipeline_class.get_params().keys())

    # Run according to options you chose

    if RandomizedSearchCV_Status == "ON":
        pipeline = RandomizedSearchCV(pipeline_class, param_distributions=param_dist, random_state=RandomState,
                                      n_iter=RandomizedSearchCV_n_iter_search, scoring=scoring_method,
                                      cv=3, n_jobs=-1)  # cross validation 10 default is overkill
    else:
        pipeline = pipeline_class


    start = time.time()
    pipeline.fit(X_train, y_train)


    # open file to output
    typeRegression_txt = ''
    if MachineLearningMethod == "Regression":
        typeRegression_txt = typeRegression + "_"
    nameOfExportedModel = str(date.today()) + "_" + settingParams + "_" + str(
        ntpath.basename(
            CSV_Path)) + "_" + MachineLearningMethod + "_" + typeRegression_txt + relationsHyperparameters + str(
        Slurm_Task_idx)

    outputFileName = str(
        ntpath.basename(CSV_Path)) + "_" + MachineLearningMethod + "_" + typeRegression_txt + relationsHyperparameters
    if (TypeOfTest == 0):
        folder = 'default'
    elif (TypeOfTest == 1):
        folder = 'no_relations'
    elif (TypeOfTest == 2):
        folder = 'relations'
    elif (TypeOfTest == 3):
        folder = 'rf_optimized'
    file = open('../accuraciesOutput_NoTI/' + folder + '/' + outputFileName + '.txt', 'a')


    # Tell us what settings we are going to run
    print("___________________________________________")
    print("Task Id: {}".format(Slurm_Task_idx))
    if (RandomizedSearchCV_Status == "ON"):
        print("Hyperparameter optimalization ON")
        if (RunType == 1):
            # Relations for bitcoin transformer
            print("Bitcoin Transformer hyperparameters relations = OFF")
        else:
            print("Bitcoin Transformer hyperparameters relations = ON")
        print("RandomizedSearchCV with n_iter_search = {}, RandomForest n_estimators = {}, dataset: {}".format(
            RandomizedSearchCV_n_iter_search, RandomForest_n_estimators, shortenDataset))

    else:
        print("Default Settings. No hyperparameter optimalizations.")
        print("RandomForest n_estimators = {}".format(RandomForest_n_estimators))

    print("CSV File: {}".format(CSV_Path))
    print("Type Of Machine Learning: {}".format(MachineLearningMethod))
    print("Random State: {}".format(RandomState))

    if RandomizedSearchCV_Status == "ON":
        print("RandomizedSearchCV took %.2f seconds for %d candidates parameter settings." % (
            (time.time() - start), RandomizedSearchCV_n_iter_search))
        # report(pipeline.cv_results_)

    else:
        print("RandomForest took %.2f seconds for %d n_estimators." % (
            (time.time() - start), RandomForest_n_estimators))

    # 6. Fit & Predict the pipeline
    # __________________________
    if RandomizedSearchCV_Status == "ON" and MachineLearningMethod == "Classification":
        # Calculate improvement of models
        # What would the accuracy be if we would run it on a default estimator?
        print("Default random forest, n_estimators=10, random_state=42 :")
        base_model = RandomForestClas_Reg
        base_model.fit(X_train, y_train)
        base_accuracy = evaluate(base_model, X_test, y_test)

        # What would the accuracy be with the optimized model?
        print("RandomizedSearchCV RF:")
        best_random = pipeline.best_estimator_
        random_accuracy = evaluate(best_random, X_test, y_test)

        print('Improvement of {:0.2f}%.'.format(100 * ((random_accuracy - base_accuracy) / base_accuracy)))

    # Predict real model
    y_predict = pipeline.predict(X_test)

    # Export the predicted model
    df = df[len(X_train):]
    df['Predicted'] = y_predict
    dfClose = df['Close']
    df['Change'] = dfClose.pct_change(periods=1)  # Contains percentage change
    df = df[['Timestamp', 'Close', 'Change', 'Target', 'Predicted']]

   # df.to_csv("../PredictedModels/" + nameOfExportedModel + ".csv")

    # 7. Scores
    # __________________________
    if MachineLearningMethod == "Classification":

        print('-------------  Output Classification ------------- ')
        print('Accuracy score:')
        accuracy = round(accuracy_score(y_test, y_predict) * 100, 2)
        print(accuracy)
        file.write(str(accuracy) + '\n')
        # print('Conf matrix:')
        # print(confusion_matrix(y_test, y_predict))
        # print('Classification report:')
        # print(classification_report(y_test, y_predict))

        if RandomizedSearchCV_Status == "ON":
            print("Best params:")
            print(pipeline.best_params_)
        # print(pipeline.sprint_statistics()) # Auto SK learn
        # print(pipeline.show_models()) # Auto SK learn
        # TEMP print(pipeline.cv_results_)
        # print(pipeline.named_steps['classify'].sprint_statistics()) # Auto SK learn
        # print(pipeline.named_steps['classify'].show_models()) #Auto Sk Learn


    elif MachineLearningMethod == "Regression":
        print('-------------  Output Regression -------------')
        mae = str(metrics.mean_absolute_error(y_test, y_predict))
        mse = str(metrics.mean_squared_error(y_test, y_predict))
        rmse = str(np.sqrt(metrics.mean_squared_error(y_test, y_predict)))
        modelscore = str(pipeline.score(X_train, y_train))
        print('Mean Absolute Error:' + mae)
        print('Mean Squared Error:' + mse)
        print('Root Mean Squared Error:' + rmse)
        print('Model Score:' + modelscore)
        file.write(mae + ' ' + mse + ' ' + rmse + ' ' + modelscore + '\n')
        if RandomizedSearchCV_Status == "ON":
            print("Best params:")
            print(pipeline.best_params_)

    file.close()


if __name__ == "__main__":

    main(sys.argv[1:])
