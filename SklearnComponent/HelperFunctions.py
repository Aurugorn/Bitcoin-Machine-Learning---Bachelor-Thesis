import autosklearn
from BitcoinComponent import BitcoinTransformer
from BitcoinComponent import ParameterRelationsBTCTrans
from BitcoinComponent_AutoSk import BitcoinTransformer_AutoSk
import pandas as pd
import numpy as np
import ta
import time
from datetime import datetime

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# auto sklearn
import autosklearn.classification as classifier
# Set Variables
CSV_Path = "../inputs/bitfinex_tBTCUSD_1h.csv"

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
        print('Reading input...')
        start_time = time.time()
    df = pd.read_csv(input, encoding="ISO-8859-1")
    if dropna:
        # Clean NaN values
        df = ta.utils.dropna(df)

    df.index.names = ['index']
    if timeOutput:
        end_time = time.time()
        print('Input read, took ' + str(end_time - start_time) + ' seconds')
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
df = csvToDF(CSV_Path, dropna=False, timeOutput=True)

# (optional) Convert datetime to timestamp
# inputX = datetimeToTimestamp(input=inputX, Date=inputX['Date'], Time=inputX['Time'], Format='%Y-%m-%d %H:%M:%S')
# 2. Prepare X
# __________________________
X = df[["Open", "Low", "High", "Close", "Volume"]]
# 3. Prepare target
# __________________________
target = generateTarget(df, method="Classification", typeRegression="Difference", intervalPeriods=1)
# 4. Train test split
# __________________________
# Add LDA component to auto-sklearn.
autosklearn.pipeline.components.feature_preprocessing.add_preprocessor(BitcoinTransformer_AutoSk)

X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2, train_size=0.8, shuffle=False,
                                                    random_state=42)
# Configuration space.
cs = BitcoinTransformer_AutoSk.get_hyperparameter_search_space()
print(cs)
# 5. Start the pipeline
# __________________________
'''

pipeline = Pipeline(steps=[
    ('BitcoinTransformer', ParameterRelationsBTCTrans()
     ),
    ('imputing', SimpleImputer(missing_values=np.nan, strategy='mean')),
    ('classify', RandomForestClassifier(n_estimators=10, random_state=42))
])
'''
'''
pipeline = Pipeline(steps=[
    ('BitcoinTransformerWithRelationparameters', ParameterRelationsBTCTrans()),
    ('imputing', SimpleImputer(missing_values=np.nan, strategy='mean')),
    ('classify', classifier.AutoSklearnClassifier(time_left_for_this_task=360, per_run_time_limit=180,include_estimators=["random_forest"]))
])
'''
'''
include_estimators=["random_forest", ], exclude_estimators=None,
include_preprocessors=["no_preprocessing", ], exclude_preprocessors=None

automlclassifier = classifier.AutoSklearnClassifier(time_left_for_this_task=120, per_run_time_limit=60)
automlclassifier_ = automlclassifier.fit(X_train, y_train)
'''
pipeline = classifier.AutoSklearnClassifier(
time_left_for_this_task=360, per_run_time_limit=180,
        include_preprocessors=['BitcoinTransformer_AutoSk'],
    )
# 6. Fit & Predict the pipeline
# __________________________

pipeline.fit(X_train, y_train)
'''
print(automlclassifier_)
'''
y_predict = pipeline.predict(X_test)

# 7. Scores
# __________________________
print('-------------  Real Random Forest Model ------------- ')
print('Accuracy score:')
print(round(accuracy_score(y_test, y_predict) * 100, 2))
print('Conf matrix:')
print(confusion_matrix(y_test, y_predict))
print('Classification report:')
print(classification_report(y_test, y_predict))

print(pipeline.sprint_statistics())
print(pipeline.show_models())

print(pipeline.named_steps['classify'].sprint_statistics())
print(pipeline.named_steps['classify'].show_models())
