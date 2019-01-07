import sklearn.metrics as metrics
import sklearn.preprocessing as preprocessing
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC


# Load in the data with `read_csv()`
horsesDataSet = pd.read_csv("horse.csv", header=0, delimiter=',')

horsesDataSetTest = pd.read_csv("horseTest.csv", header=0, delimiter=',')

# iterate through each attribute and define the percentage of missing values

# populate array with zeros with column dimensions of dataset
qtd_nan = [0 for x in range(horsesDataSet.shape[1])]

# populate array with zeros with column dimensions of dataset
qtd_total = [0 for x in range(horsesDataSet.shape[1])]

i = 0
while i < horsesDataSet.shape[1]:
    # get array of boolean describing each line as null or not for i attribute
    attributeLinesIsNA = pd.isna(horsesDataSet.iloc[:, i])

    # get current attribute label name
    currentAttributeLabel = list(horsesDataSet)[i]

    qtd_nan[i] = horsesDataSet.loc[attributeLinesIsNA, currentAttributeLabel].shape[0]
    qtd_total[i] = horsesDataSet.loc[:, currentAttributeLabel].shape[0]
    i = i + 1

percentageArray = np.divide(qtd_nan, qtd_total)
threshold = 0.5
PreProcessedHorseDataSet = horsesDataSet
PreProcessedHorseDataSetTest = horsesDataSetTest
i = 0
while i < horsesDataSet.shape[1]:
    if percentageArray[i] > threshold:
        # get current attribute label name
        currentAttributeLabel = list(horsesDataSet)[i]

        # drop attribute column if na values > threshold
        PreProcessedHorseDataSet = PreProcessedHorseDataSet.drop(columns=currentAttributeLabel)

        # drop from test
        PreProcessedHorseDataSetTest = PreProcessedHorseDataSetTest.drop(columns=currentAttributeLabel)

    i = i + 1

# fill remaining lines with mean values
PreProcessedHorseDataSet = PreProcessedHorseDataSet.fillna(horsesDataSet.mean())
PreProcessedHorseDataSetTest = PreProcessedHorseDataSetTest.fillna(horsesDataSetTest.mean())

# Show Statistics of DataSet
StatisticsPreProcessedHorseDataSet = PreProcessedHorseDataSet.describe(include='all')

# Altering Categorical missing values to Mode Value (value that appear the most often)
i = 0
while i < PreProcessedHorseDataSet.shape[1]:
    # return the most frequent value (first index because mode() returns a DataFrame)
    attributeMode = PreProcessedHorseDataSet.mode().iloc[0, i]
    currentAttributeLabel = list(PreProcessedHorseDataSet)[i]
    PreProcessedHorseDataSet[currentAttributeLabel] = PreProcessedHorseDataSet[currentAttributeLabel].fillna(
        attributeMode)
    i = i + 1

# Altering Categorical missing values to Mode Value (value that appear the most often) [DATASET TEST]
i = 0
while i < PreProcessedHorseDataSetTest.shape[1]:
    # return the most frequent value (first index because mode() returns a DataFrame)
    attributeMode = PreProcessedHorseDataSetTest.mode().iloc[0, i]
    currentAttributeLabel = list(PreProcessedHorseDataSetTest)[i]
    PreProcessedHorseDataSetTest[currentAttributeLabel] = PreProcessedHorseDataSetTest[currentAttributeLabel].fillna(
        attributeMode)
    i = i + 1

# categorical attribute binarization

categoricalHorseDataSet = PreProcessedHorseDataSet.select_dtypes(include='object')
categoricalHorseDataSet = categoricalHorseDataSet.drop('outcome', axis=1)
categoricalHorseDataSetDummy = pd.get_dummies(categoricalHorseDataSet)
PreProcessedHorseDataSet = pd.concat([categoricalHorseDataSetDummy, PreProcessedHorseDataSet.loc[:, 'outcome']], axis=1)

# categorical attribute binarization [DATASET TEST]

categoricalHorseDataSetTest = PreProcessedHorseDataSetTest.select_dtypes(include='object')
categoricalHorseDataSetTest = categoricalHorseDataSetTest.drop('outcome', axis=1)
categoricalHorseDataSetDummy = pd.get_dummies(categoricalHorseDataSetTest)
PreProcessedHorseDataSetTest = pd.concat([categoricalHorseDataSetDummy, PreProcessedHorseDataSetTest.loc[:, 'outcome']],
                                         axis=1)

# Change values from euthanized to died
AttributesHorseDataSet = PreProcessedHorseDataSet.drop('outcome', axis=1)

TargetHorseDataSet = PreProcessedHorseDataSet.loc[:, 'outcome']

# mapping 'euthanized' values to 'died' to tune fitting
TargetHorseDataSet = TargetHorseDataSet.map(lambda x: 'died' if x == 'euthanized' else x)

PreProcessedHorseDataSet = pd.concat([AttributesHorseDataSet, TargetHorseDataSet], axis=1)

# Change values from euthanized to died [DATASET TEST]

AttributesHorseDataSetTest = PreProcessedHorseDataSetTest.drop('outcome', axis=1)

TargetHorseDataSetTest = PreProcessedHorseDataSetTest.loc[:, 'outcome']

# mapping 'euthanized' values to 'died' to tune fitting
TargetHorseDataSetTest = TargetHorseDataSetTest.map(lambda x: 'died' if x == 'euthanized' else x)

PreProcessedHorseDataSetTest = pd.concat([AttributesHorseDataSetTest, TargetHorseDataSetTest], axis=1)


#-------------- SVM ---------------------

svmClassifier = SVC(kernel='linear', C = 1.0)
svmClassifier.fit(AttributesHorseDataSet, TargetHorseDataSet)

TargetHorseDataSet_prediction = svmClassifier.predict(AttributesHorseDataSetTest)



# -------------- Metrics ----------------

accuracyScore = metrics.accuracy_score(TargetHorseDataSetTest, TargetHorseDataSet_prediction)
print(accuracyScore)
recallScore = metrics.recall_score(TargetHorseDataSetTest, TargetHorseDataSet_prediction, average=None)
print(recallScore)
kappaScore = metrics.cohen_kappa_score(TargetHorseDataSetTest, TargetHorseDataSet_prediction)
print(kappaScore)

#TargetHorseDataSet_test = labelEncoder.inverse_transform(TargetHorseDataSet_test)
#TargetHorseDataSet_prediction = labelEncoder.inverse_transform(TargetHorseDataSet_prediction)

confusionMatrix = pd.DataFrame(
    metrics.confusion_matrix(TargetHorseDataSetTest, TargetHorseDataSet_prediction),
    columns=['Predicted Died', 'Predicted Lived'],
    index=['True Died', 'True Lived']
)

print(confusionMatrix)