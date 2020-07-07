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
from collections import Counter

# Read data from file 'filename.csv' 
# (in the same directory that your python process is based)
# Control delimiters, rows, column names with read_csv (see later) 

df = pd.read_csv("outputs/23062020_Classification_1MIN_ALLTI.csv")

#df = pd.read_csv("output_AllData_moreTI.csv") 
# Delete columns that contains only NaN values
cols = df.columns[df.isna().all()]
df = df.drop(cols, axis = 1)
# Delete rows that contain at least one NaN
df = df.dropna()
#Machine learning
target = df['Target']



data_cols = df.drop(['Target', 'index'], axis='columns').columns.values

X = df[data_cols]
start_time = time.time()
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2, train_size=0.8, shuffle=False, random_state = 42)

model = RandomForestClassifier(n_estimators=10, random_state=42)
classifier = model.fit(X_train, y_train)

print('# Time to complete RF: ' + str(time.time() - start_time) + ' seconds')
y_predict = model.predict(X_test)
print('-------------  Real Random Forest Model ------------- ')
print('Amount of target False/True occurences: {}'.format(Counter(target)))
print('Accuracy score:')
print(round(accuracy_score(y_test, y_predict)  * 100, 2))
print('Conf matrix:')
print(confusion_matrix(y_test, y_predict))
print('Classification report:')
print(classification_report(y_test, y_predict))

print('Plotted matrix:')
# Plot non-normalized confusion matrix

disp = plot_confusion_matrix(classifier, X_test, y_test,
                             display_labels=df['Target'],
                             cmap=plt.cm.Blues,
                             normalize='true')
disp.ax_.set_title("Confusion Matrix")

print("Confusion Matrix")
print(disp.confusion_matrix)



# 80/20 : 57%
# 60/40 : 55%
# 40/60 : 55%
# 20/80 : 

# auto-sk learn
'''
automl = autosklearn.classification.AutoSklearnClassifier(
    include_estimators=["random_forest", ], exclude_estimators=None,
    include_preprocessors=["no_preprocessing", ], exclude_preprocessors=None)
automl.fit(X_train, y_train)
y_predict = automl.predict(X_test)
print("Accuracy score", accuracy_score(y_test, y_predict))

''' 



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



#df.to_csv("modelOutputs/PredictedModel_RF_ALLTI_1HOUR.csv")


plt.show()