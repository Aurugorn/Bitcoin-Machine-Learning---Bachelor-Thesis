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
from sklearn.dummy import DummyClassifier
import plotly.express as px
import plotly.graph_objects as go


# Read data from file 'filename.csv' 
# (in the same directory that your python process is based)
# Control delimiters, rows, column names with read_csv (see later) 
df = pd.read_csv("outputs/1DAY_BTCUSD_ALLTI.csv") 
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
df['Predicted'] = np.where(df.Target.shift(1) == True, True, False)
y_predict_NOCHANGE = df['Predicted'][len(X_train):]
print('Accuracy score:')
accuracy_nochange = round(accuracy_score(y_test, y_predict_NOCHANGE)  * 100, 2)
print(accuracy_nochange)
print('Classification report:')
print(classification_report(y_test, y_predict_NOCHANGE))



print('------------- DUMMY BASELINE MODEL (Stratified) generates predictions by respecting the training set’s class distribution. Random ------------- ')
# Dummy classifier to create baseline to compare to the real models
dummy_clf = DummyClassifier(strategy="stratified", random_state=42)
dummy_clf_ = dummy_clf.fit(X_train, y_train)
y_predict_STRATIFIED = dummy_clf.predict(X_test)

print('Accuracy score:')
accuracy_stratified = round(accuracy_score(y_test, y_predict_STRATIFIED)  * 100, 2)
print(accuracy_stratified)
print('Classification report:')
print(classification_report(y_test, y_predict_STRATIFIED))


print('------------- DUMMY BASELINE MODEL (Most Frequent)  always predicts the most frequent label in the training set. ------------- ')
dummy_clf = DummyClassifier(strategy="most_frequent", random_state=42)
dummy_clf_ = dummy_clf.fit(X_train, y_train)
y_predict_MOSTFREQ = dummy_clf.predict(X_test)

print('Accuracy score:')
accuracy_mostfrequent = round(accuracy_score(y_test, y_predict_MOSTFREQ)  * 100, 2)
print(accuracy_mostfrequent)
print('Classification report:')
print(classification_report(y_test, y_predict_MOSTFREQ))

print('------------- DUMMY BASELINE MODEL (Prior) always predicts the class that maximizes the class prior (like “most_frequent”) and predict_proba returns the class prior. ------------- ')
dummy_clf = DummyClassifier(strategy="prior", random_state=42)
dummy_clf_ = dummy_clf.fit(X_train, y_train)
y_predict_PRIOR = dummy_clf.predict(X_test)
print('Accuracy score:')
accuracy_prior = round(accuracy_score(y_test, y_predict_PRIOR)  * 100, 2)
print(accuracy_prior)
print('Classification report:')
print(classification_report(y_test, y_predict_PRIOR))
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
fig.add_trace(go.Box(y=[50], name='Random Forest 1DAY - With TI'))
fig.add_trace(go.Box(y=[53.29], name='Random Forest 1DAY - No TI'))
fig.add_trace(go.Box(y=[accuracy_nochange], name='No-change Baseline'))
fig.add_trace(go.Box(y=[accuracy_stratified], name='Stratified Baseline'))
fig.add_trace(go.Box(y=[accuracy_mostfrequent], name='Most-frequent Baseline'))
fig.add_trace(go.Box(y=[accuracy_prior], name='Prior Baseline'))

fig.show()

exit()
df.to_csv("modelOutputs/PredictedModel_BASELINES_1DAY.csv") 


