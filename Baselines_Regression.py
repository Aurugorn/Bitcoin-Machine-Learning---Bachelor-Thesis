'''
Created on 1 mrt. 2020

@author: stan
'''

# Load the Pandas libraries with alias 'pd' 
import pandas as pd 
import numpy as np
from datetime import datetime
from openpyxl import workbook #For export to Excel
import time
import ta
import matplotlib.pyplot as plt
#new imports

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# Recall
from sklearn.metrics import recall_score

# Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
# Precision
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.dummy import DummyRegressor
import plotly.express as px
import plotly.graph_objects as go
from sklearn import metrics


# Read data from file 'filename.csv' 
# (in the same directory that your python process is based)
# Control delimiters, rows, column names with read_csv (see later) 
df = pd.read_csv("outputsRegression/1HOUR_BTCUSD_ALLTI.csv") 
#df = pd.read_csv("output_AllData_moreTI.csv") 
# Delete columns that contains only NaN values
cols = df.columns[df.isna().all()]
df = df.drop(cols, axis = 1)
# Delete rows that contain at least one NaN
df = df.dropna()
#Machine learning
target = df['Target']



data_cols = df.drop(['Target', 'Timestamp', 'index'], axis='columns').columns.values

X = df[data_cols]

X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2, train_size=0.8, shuffle=False, random_state = 42)

print('------------- No-Change baseline: Predict same Target as last timestamp ------------- ')
# No-Change baseline
# Add new column 'Predicted' . True if previous timestamp was also true
df['Predicted'] = df.Target.shift(1)
y_predict_NOCHANGE = df['Predicted'][len(X_train):]
MAE_nochange = metrics.mean_absolute_error(y_test, y_predict_NOCHANGE)
print('Mean Absolute Error:', MAE_nochange)
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_predict_NOCHANGE))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_predict_NOCHANGE)))




print('------------- DUMMY BASELINE MODEL “mean”: always predicts the mean of the training set ------------- ')
# Dummy classifier to create baseline to compare to the real models
dummy_clf = DummyRegressor(strategy="mean")
dummy_clf_ = dummy_clf.fit(X_train, y_train)
y_predict_MEAN = dummy_clf.predict(X_test)

MAE_mean = metrics.mean_absolute_error(y_test, y_predict_MEAN)
print('Mean Absolute Error:', MAE_mean)
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_predict_MEAN))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_predict_MEAN)))


print('------------- DUMMY BASELINE MODEL “median”: always predicts the median of the training set ------------- ')
dummy_clf = DummyRegressor(strategy="median")
dummy_clf_ = dummy_clf.fit(X_train, y_train)
y_predict_median = dummy_clf.predict(X_test)

MAE_median = metrics.mean_absolute_error(y_test, y_predict_median)
print('Mean Absolute Error:', MAE_median)
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_predict_median))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_predict_median)))

# Several columns are deleted from dataset to prevent training on these columns
# We have now populated the train and test variables

# Exporting the dataframe that was predicted by the classification model
#Only print the predicted rows
df = df[len(X_train):]
#Create empty array filling in with NaNs for the training part of the dataframe
'''NaNArray = []
for i in range(0, len(X_train)):
    NaNArray.append(False)

y_predict = np.insert(y_predict_NOCHANGE, 0, NaNArray)
'''
df['Predicted'] = y_predict_NOCHANGE



#Drop all columns, only need TIMESTAMP, CLOSE, TARGET, PREDICTED
dfClose = df['Close']
df['Change'] = dfClose.pct_change(periods=1) #Contains percentage change
df = df[['Timestamp','Close','Change','Target','Predicted']]


# Create boxplot from accuracy results
fig = go.Figure()
fig.add_trace(go.Box(y=[48.2], name='Random Forest 1HOUR - With TI'))
fig.add_trace(go.Box(y=[43.2], name='Random Forest 1HOUR - No TI'))
fig.add_trace(go.Box(y=[MAE_nochange], name='No-change Baseline'))
fig.add_trace(go.Box(y=[MAE_mean], name='Mean Baseline'))
fig.add_trace(go.Box(y=[MAE_median], name='Median Baseline'))

fig.show()

exit()
df.to_csv("modelOutputs/PredictedModel_BASELINES_1DAY.csv") 


