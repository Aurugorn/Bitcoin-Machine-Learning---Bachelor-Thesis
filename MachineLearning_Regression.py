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
from sklearn.ensemble import RandomForestRegressor
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
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import plotly.graph_objects as go
import plotly.express as px

# Read data from file 'filename.csv' 
# (in the same directory that your python process is based)
# Control delimiters, rows, column names with read_csv (see later) 
SCALE_DATASET = False
df = pd.read_csv("outputsRegression/1MIN_BTCUSD_ALLTI_Diff.csv") 

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

# Scaledataset
if SCALE_DATASET:
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

model = RandomForestRegressor(n_estimators=10, random_state=42)
classifier = model.fit(X_train, y_train)
y_predict = model.predict(X_test)
#In a good model, 
#the RMSE should be close for both your testing data and your training data.
# If the RMSE for your testing data is higher than the training data, 
#there is a high chance that your model overfit. 
#In other words, your model performed worse during testing than training.
print('-------------  Real Random Forest Model ------------- ')
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_predict))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_predict))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_predict)))
print('Model Score:', model.score(X_train, y_train))


# How important was each TI added to helping our model perform?
# Get numerical feature importances

feature_list = data_cols
importances = list(model.feature_importances_)

dropCols = []
FeatureColums = [(feature, round(importance, 6)) for feature, importance in zip(feature_list, importances)]
FeatureColums = sorted(FeatureColums, key = lambda x: x[1], reverse = True)
count = 0
for i in FeatureColums:
    if count > 10:
        dropCols.append(i[0])    
    count += 1
       

# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 6)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]


#TEMPORARILY
# Save dataframe with optimal features, drop useless features
#df = df.drop(dropCols, axis='columns')
#df.to_csv("outputs/2013_2020_OPTIMIZED_COLS.csv") 
#exit()
# Exporting the dataframe that was predicted by the classification model
#Only print the predicted rows
df = df[len(X_train):]
#Create empty array filling in with NaNs for the training part of the dataframe
'''NaNArray = []
for i in range(0, len(X_train)):
    NaNArray.append(False)

y_predict = np.insert(y_predict_NOCHANGE, 0, NaNArray)
'''
df['Predicted'] = y_predict



#Drop all columns, only need TIMESTAMP, CLOSE, TARGET, PREDICTED
dfClose = df['Close']
df['Change'] = dfClose.pct_change(periods=1) #Contains percentage change
df = df[['Timestamp','Close','Change','Target','Predicted']]

print(df)
x_axis_dates = []
for i, j in df.iterrows(): 
    dateTime_string = datetime.fromtimestamp(j.Timestamp).strftime("%Y-%m-%d")
    x_axis_dates.append(dateTime_string)
#Plot results
#to see the relationship between the training data values
'''
plt.scatter(x_train,y_train,c='red')
plt.show()

#to see the relationship between the predicted 
#brain weight values using scattered graph
plt.plot(x_test,y_pred)   
plt.scatter(x_test,y_test,c='red')
plt.xlabel('headsize')
plt.ylabel('brain weight')
'''
fig = go.Figure()

# Add traces
fig.add_trace(go.Scatter(x=x_axis_dates, y=y_test,
                    mode='lines',
                    name='Actual'))
fig.add_trace(go.Scatter(x=x_axis_dates, y=y_predict,
                    mode='lines',
                    name='Predicted'))
fig.update_layout(
    title = 'Random Forest Regression Actual VS Predicted',
    xaxis_tickformat = '%d %B (%a)<br>%Y'
)


''' 
# Line for breakeven
fig.add_trace(
    go.Scatter(x=x_axis, y=df['Break-even'], name="Break-even")
)

fig.update_layout(
    title = 'BTC/EUR Portfolio value over time',
    xaxis_tickformat = '%d %B (%a)<br>%Y'
)
'''

fig.show()
#exit()

df.to_csv("modelOutputsRegression/PredictedModel_RF_ALLTI_1MIN_DIFF_n10.csv") 


plt.show()
