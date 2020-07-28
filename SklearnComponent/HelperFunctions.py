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
import ta


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
        df['Target'] = np.where(df.Close.shift(-1) > df.Close, True, False)
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


def getHyperParamBTC(RunType=1):
    if RunType == 2:
        # ParameterRelationsBTCTrans
        param_dist = {'BitcoinTransformer__fastperiod': loguniform(2, 100),
                      'BitcoinTransformer__longterm': loguniform(2, 100),
                      'BitcoinTransformer__midterm': loguniform(2, 100),
                      'BitcoinTransformer__shortterm': loguniform(2, 100),
                      'BitcoinTransformer__bb_cci': loguniform(2, 100),
                      'BitcoinTransformer__var_t3': loguniform(2, 100),
                      'BitcoinTransformer__dema_trema': loguniform(2, 100),
                      'BitcoinTransformer__zero': loguniform(2, 100),
                      'BitcoinTransformer__rocperiod': loguniform(2, 100)}
    else:
        # BitcoinTransformer
        param_dist = {'BitcoinTransformer__adosc_fastperiod': loguniform(2, 100),
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
    return param_dist


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
        opts, args = getopt.getopt(argv, "hx:c:s:m:t:i:h:r:n:e:",
                                   ["task_idx=", "csv=", "shortendataset=", "mlmethod=", "typeregression=",
                                    "intervalperiod=", "hyperrelations=", "rscv=", "niters=", "rfnest="])
    except getopt.GetoptError:
        print(
            'HelperFunctions.py --task_idx <$SLURM_ARRAY_TASK_ID> --csv <../path/to/csv> --mlmethod <Classification, Regression> --shortendataset <year,month> --typeregression <Difference, Percentage, ExactPrice> --intervalperiod <default:1, abs(int)> '
            '--hyperrelations <1=false, 2=true> --rscv <ON, OFF (ON=RandomizedSearchCV, OFF=RandomForest only) --niters <default: 1, abs(int)> --rfnest <1,10,100>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(
                'HelperFunctions.py --task_idx <$SLURM_ARRAY_TASK_ID> --csv <../path/to/csv> --mlmethod <Classification, Regression> --shortendataset <year,month> --typeregression <Difference, Percentage, ExactPrice> --intervalperiod <default:1, abs(int)> '
                '--hyperrelations <1=false, 2=true> --rscv <ON, OFF (ON=RandomizedSearchCV, OFF=RandomForest only) --niters <default: 1, abs(int)> --rfnest <1,10,100>')
            sys.exit()
        elif opt in ("-x", "--task_idx"):
            Slurm_Task_idx = "_idx_" + arg
            RandomState = int(arg)
        elif opt in ("-c", "--csv"):
            CSV_Path = arg
        elif opt in ("-s", "--shortendataset"):
            shortenDataset = arg
        elif opt in ("-m", "--mlmethod"):
            MachineLearningMethod = arg
        elif opt in ("-t", "--typeregression"):
            typeRegression = arg
        elif opt in ("-i", "--intervalperiod"):
            intervalPeriods = int(arg)
        elif opt in ("-h", "--hyperrelations"):
            RunType = int(arg)
        elif opt in ("-r", "--rscv"):
            RandomizedSearchCV_Status = arg
        elif opt in ("-n", "--niters"):
            RandomizedSearchCV_n_iter_search = int(arg)
        elif opt in ("-e", "--rfnest"):
            RandomForest_n_estimators = int(arg)

    # Default parameters
    if CSV_Path == '':
        CSV_Path = "../inputs/bitfinex_tBTCUSD_1h.csv"
    if MachineLearningMethod == '':
        MachineLearningMethod = "Classification"
    if typeRegression == '':
        typeRegression = "Difference"
    if intervalPeriods == '':
        intervalPeriods = 1
    if RunType == '':
        RunType = 1  # no relations
    if RandomizedSearchCV_Status == '':
        RandomizedSearchCV_Status = "ON"  # use RandomizedSearchCV with RF
    if RandomizedSearchCV_n_iter_search == '':
        RandomizedSearchCV_n_iter_search = 1
    if RandomForest_n_estimators == '':
        RandomForest_n_estimators = 10
    if RandomState == '':
        RandomState = 42
    if shortenDataset == '':
        shortenDataset = False

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
    df = csvToDF(CSV_Path, dropna=True, timeOutput=True)

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
    df = datetimeToTimestamp(input=df, Date=df['Date'], Time=df['Time'], Format='%m/%d/%Y %H:%M:%S', timeOutput=True)
    # 2. Prepare X
    # __________________________
    X = df[["Open", "Low", "High", "Close", "Volume"]]
    # 3. Prepare target
    # __________________________
    target = generateTarget(df, method=MachineLearningMethod, typeRegression=typeRegression,
                            intervalPeriods=intervalPeriods)

    # 4. Train test split
    # __________________________
    # Set last row of target on FALSE or 0
    target.at[target.index[-1]] = 0

    X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2, train_size=0.8, shuffle=False,
                                                        random_state=RandomState)

    # Configuration space.
    # cs = BitcoinTransformer_AutoSk.get_hyperparameter_search_space()
    # print(cs)
    # 5. Start the pipeline
    # __________________________

    if RandomizedSearchCV_Status == "OFF":
        print("Going to run normal RandomForest with n_estimators={}".format(RandomForest_n_estimators))
    else:
        print("Going to run RandomizedSearchCV with n_iter_search={}".format(RandomizedSearchCV_n_iter_search))

    if MachineLearningMethod == "Classification":
        RandomForestClas_Reg = RandomForestClassifier(n_estimators=RandomForest_n_estimators, random_state=RandomState)
    elif MachineLearningMethod == "Regression":
        RandomForestClas_Reg = RandomForestRegressor(n_estimators=RandomForest_n_estimators, random_state=RandomState)

    if RunType == 1:
        print("Going to use 'BitcoinTransformer()'")
        pipeline_class = Pipeline(steps=[
            ('BitcoinTransformer', BitcoinTransformer()
             ),
            ('imputing', SimpleImputer(missing_values=np.nan, strategy='mean')),
            ('classify', RandomForestClas_Reg)
        ])
    elif RunType == 2:
        print("Going to use 'ParameterRelationsBTCTrans()'")
        pipeline_class = Pipeline(steps=[
            ('BitcoinTransformer', ParameterRelationsBTCTrans()
             ),
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
    param_dist = getHyperParamBTC(RunType)

    # All possible parameters for the pipeline
    # print(pipeline_class.get_params().keys())

    # Run according to options you chose

    if RandomizedSearchCV_Status == "ON":
        pipeline = RandomizedSearchCV(pipeline_class, param_distributions=param_dist, random_state=RandomState,
                                      n_iter=RandomizedSearchCV_n_iter_search,
                                      cv=3, n_jobs=-1)  # cross validation 10 default is overkill
    else:
        pipeline = pipeline_class
    start = time.time()
    pipeline.fit(X_train, y_train)

    # open file to output
    typeRegression_txt = ''
    if MachineLearningMethod == "Regression":
        typeRegression_txt = typeRegression + "_"
    nameOfExportedModel = str(date.today()) + "_" + str(
        ntpath.basename(
            CSV_Path)) + "_" + MachineLearningMethod + "_" + typeRegression_txt + relationsHyperparameters + Slurm_Task_idx
    file = open('../outputLogs/' + nameOfExportedModel + '.txt', 'a')

    if RandomizedSearchCV_Status == "ON":
        print("RandomizedSearchCV took %.2f seconds for %d candidates parameter settings." % (
            (time.time() - start), RandomizedSearchCV_n_iter_search))
        report(pipeline.cv_results_)
        file.write("RandomizedSearchCV took %.2f seconds for %d candidates parameter settings." % (
            (time.time() - start), RandomizedSearchCV_n_iter_search))
        report(pipeline.cv_results_)
    else:
        print("RandomForest took %.2f seconds for %d n_estimators." % (
            (time.time() - start), RandomForest_n_estimators))
        file.write("RandomForest took %.2f seconds for %d n_estimators." % (
            (time.time() - start), RandomForest_n_estimators))

    # 6. Fit & Predict the pipeline
    # __________________________

    y_predict = pipeline.predict(X_test)

    # Export the predicted model
    df = df[len(X_train):]
    df['Predicted'] = y_predict
    dfClose = df['Close']
    df['Change'] = dfClose.pct_change(periods=1)  # Contains percentage change
    df = df[['Timestamp', 'Close', 'Change', 'Target', 'Predicted']]

    df.to_csv("../PredictedModels/" + nameOfExportedModel + ".csv")

    # Output to file

    file.write("Program Parameters:\n")
    file.write("MachineLearningMethod = " + MachineLearningMethod + '\n')
    file.write("typeRegression = " + str(typeRegression) + '\n')
    file.write("intervalPeriods = " + str(intervalPeriods) + '\n')
    file.write("RunType = " + str(RunType) + '\n')
    file.write("RandomizedSearchCV_Status = " + RandomizedSearchCV_Status + '\n')
    file.write("RandomizedSearchCV_n_iter_search = " + str(RandomizedSearchCV_n_iter_search) + '\n')
    file.write("RandomForest_n_estimators = " + str(RandomForest_n_estimators) + '\n')
    file.write("\n\n")

    # 7. Scores
    # __________________________
    if MachineLearningMethod == "Classification":

        print('-------------  Output Classification ------------- ')
        print('Accuracy score:')
        print(round(accuracy_score(y_test, y_predict) * 100, 2))
        print('Conf matrix:')
        print(confusion_matrix(y_test, y_predict))
        print('Classification report:')
        print(classification_report(y_test, y_predict))

        if RandomizedSearchCV_Status == "ON":
            print("Best params:")
            print(pipeline.best_params_)
        # print(pipeline.sprint_statistics()) # Auto SK learn
        # print(pipeline.show_models()) # Auto SK learn
        print(pipeline.cv_results_)
        # print(pipeline.named_steps['classify'].sprint_statistics()) # Auto SK learn
        # print(pipeline.named_steps['classify'].show_models()) #Auto Sk Learn

        file.write("-------------  Output Classification ------------- \n")
        file.write("Accuracy score: \n")
        file.write(str(round(accuracy_score(y_test, y_predict) * 100, 2)) + '\n')
        file.write(classification_report(y_test, y_predict) + '\n\n')
        file.write('Conf Matrix:\n')
        file.write(numpy.array_str(confusion_matrix(y_test, y_predict)) + '\n\n')
        if RandomizedSearchCV_Status == "ON":
            file.write("Best params:")
            file.write(json.dumps(pipeline.best_params_) + '\n\n')
        file.write('CV_Results:\n')
        file.write(str(pipeline.cv_results_))


    elif MachineLearningMethod == "Regression":
        file.write('-------------  Output Regression ------------- \n')
        file.write('Mean Absolute Error:' + str(metrics.mean_absolute_error(y_test, y_predict)) + '\n')
        file.write('Mean Squared Error:' + str(metrics.mean_squared_error(y_test, y_predict)) + '\n')
        file.write('Root Mean Squared Error:' + str(np.sqrt(metrics.mean_squared_error(y_test, y_predict))) + '\n')
        file.write('Model Score:' + str(pipeline.score(X_train, y_train)))

    file.close()


if __name__ == "__main__":
    main(sys.argv[1:])
