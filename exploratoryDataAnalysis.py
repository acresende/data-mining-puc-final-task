# Import the `pandas` library as `pd`
import pandas as pd
import numpy as np

# Load in the data with `read_csv()`
horsesDataSet = pd.read_csv("horse.csv", header=0, delimiter=',')

#description of dataSet
descriptionHorsesDataSet = horsesDataSet.describe()

#first 5 and last 5 entries in dataSet
firstRowsDataSet = horsesDataSet.head(5)


lastRowsDataSet = horsesDataSet.tail(5)

# sampling data

# Take a sample of 5
horsesDataSetSample = horsesDataSet.sample(5)

result = pd.isnull(horsesDataSet)

#iterate throught each attribute and define the percentage of missing values

qtd_nan = [0 for x in range(horsesDataSet.shape[1])] #populate array with zeros with column dimensions of dataset
qtd_total = [0 for x in range(horsesDataSet.shape[1])] #populate array with zeros with column dimensions of dataset
i= 0
while i < horsesDataSet.shape[1]:
    attributeLinesIsNA = pd.isna(horsesDataSet.iloc[:, i]) #get array of boolean describing each line as null or not for i attribute
    currentAttributeLabel = list(horsesDataSet)[i] #get current attribute label name
    qtd_nan[i] = horsesDataSet.loc[attributeLinesIsNA, currentAttributeLabel].shape[0]
    qtd_total[i] = horsesDataSet.loc[:, currentAttributeLabel].shape[0]
    i = i+1

percentageArray = np.divide(qtd_nan, qtd_total)
threshold = 0.6
PreProcessedHorseDataSet = horsesDataSet
i = 0
while i < horsesDataSet.shape[1]:
    if percentageArray[i] > threshold:
        currentAttributeLabel = list(horsesDataSet)[i]  # get current attribute label name
        PreProcessedHorseDataSet = PreProcessedHorseDataSet.drop(columns=currentAttributeLabel) # drop attribute column if na values > threshold
    i = i + 1
PreProcessedHorseDataSet = PreProcessedHorseDataSet.fillna(horsesDataSet.mean()) #fill remaining lines with mean values

######

#set index

horsesDataSetWithIndex = horsesDataSet.set_index("hospital_number", drop = False)

# drop columns with missing values

#drop lines with missing values

rectal_tempMean = np.mean(horsesDataSetWithIndex.rectal_temp)

horsesDataSet = horsesDataSetWithIndex.fillna(rectal_tempMean)


