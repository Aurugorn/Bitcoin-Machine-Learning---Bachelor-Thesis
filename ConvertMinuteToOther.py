'''
Created on 13 feb. 2020

@author: stan
'''
# Load the Pandas libraries with alias 'pd' 
import pandas as pd 
import numpy as np
import ta
import talib
from ta.momentum import StochasticOscillator
from datetime import datetime
from openpyxl import workbook #For export to Excel
import time
from sklearn.linear_model import LogisticRegression

# Configuration
LOAD_CSV = "bitfinex_tBTCUSD_1m.csv"
EXPORT = "EXCEL" #Options: EXCEL , CSV , NONE
EXPORT_NAME_EXCEL = "inputs/bitfinex_tBTCUSD_1hour.xlsx"
EXPORT_NAME_CSV = "inputs/bitfinex_tBTCUSD_1hour.csv"

# Read data from file 'filename.csv' 
# (in the same directory that your python process is based)
# Control delimiters, rows, column names with read_csv (see later) 
print('Loading CSV...')
start_time = time.time()
df = pd.read_csv(LOAD_CSV, encoding = "ISO-8859-1") 
# Clean NaN values
df = ta.utils.dropna(df)
df.index.names = ['index']
end_time = time.time()
print('Time to load and prepare CSV: ' + str(end_time - start_time) + ' seconds')

# Create empty pandas dataframe
dfO = pd.DataFrame(columns=["Date", "Time", "Open", "High", "Low", "Close", "Volume"])

# Loop over every row
# If start of hour --> first row
# Loop untill end of hour - 1
# VARIABLE: HighestVal = Highest value within this interval
# VARIABLE: LowestVal = Lowest value within this interval
# VARIABLE: FullVolume = Sum of Volume over this interval
# VARIABLE: Open = start of hour value Open
# VARIABLE: Close = 00:59:00 value
countRows = 0
hourInt = -1
minuteInt = -1
secondInt = -1
previousHour = -1
rowFormat = {}
HighestVal = 0
LowestVal = 0
FullVolume = 0
CloseVal = 0
for i, j in df.iterrows(): 
    if countRows < 1000:
        hourInt = int(j.Time.split(":")[0])
        minuteInt = int(j.Time.split(":")[1])
        secondInt = int(j.Time.split(":")[2])
        print("{} : {} : {}".format(hourInt, minuteInt, secondInt))
        if hourInt != previousHour:
            previousHour = hourInt
            rowFormat = {
             "Date": j.Date,
             "Time": j.Time,
             "Open": j.Open
              }
            HighestVal = j.High
            LowestVal = j.Low
            FullVolume += j.Volume
            CloseVal = j.Close
        else:   
            # We are in the hour
            # Search for latest value in this hour
            
            dfO = dfO.append({
             "Date": j.Date,
             "Time": j.Time
              }, ignore_index=True)
    else:
        break
    countRows += 1


# Print to Excel dataframe
print(dfO)
exit()
if EXPORT == "EXCEL" or EXPORT == "CSV":
    print('Exporting Excel...')
    start_time = time.time()
    if EXPORT == "EXCEL":
        df.to_excel(EXPORT_NAME_EXCEL)
    if EXPORT == "CSV": 
        df.to_csv(EXPORT_NAME_CSV)
    end_time = time.time()
    print('Time to export excel/CSV: ' + str(end_time - start_time) + ' seconds')

