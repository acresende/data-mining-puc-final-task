# Import the `pandas` library as `pd`
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
import sklearn.metrics as metrics
import sklearn.preprocessing as preprocessing


# Load in the data with `read_csv()`
horsesDataSet = pd.read_csv("horse.csv", header=0, delimiter=',')



#description of dataSet
descriptionHorsesDataSet = horsesDataSet.describe()

# --------------------------- Exploratory analysis ---------------------------

#first 5 and last 5 entries in dataSet
firstRowsDataSet = horsesDataSet.head(5)

lastRowsDataSet = horsesDataSet.tail(5)

# sampling data

# Take a sample of 5
horsesDataSetSample = horsesDataSet.sample(5)

result = pd.isnull(horsesDataSet)


# --------------------------- Pre processing ---------------------------


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
    i = i+1

percentageArray = np.divide(qtd_nan, qtd_total)
threshold = 0.5
PreProcessedHorseDataSet = horsesDataSet
i = 0
while i < horsesDataSet.shape[1]:
    if percentageArray[i] > threshold:
        # get current attribute label name
        currentAttributeLabel = list(horsesDataSet)[i]
        # drop attribute column if na values > threshold
        PreProcessedHorseDataSet = PreProcessedHorseDataSet.drop(columns=currentAttributeLabel)
    i = i + 1

# fill remaining lines with mean values
PreProcessedHorseDataSet = PreProcessedHorseDataSet.fillna(horsesDataSet.mean())

# Show Statistics of DataSet
StatisticsPreProcessedHorseDataSet = PreProcessedHorseDataSet.describe(include='all')


# Altering Categorical missing values to Mode Value (value that appear the most often)
i = 0
while i < PreProcessedHorseDataSet.shape[1]:
    # return the most frequent value (first index because mode() returns a DataFrame)
    attributeMode = PreProcessedHorseDataSet.mode().iloc[0, i]
    currentAttributeLabel = list(PreProcessedHorseDataSet)[i]
    PreProcessedHorseDataSet[currentAttributeLabel] = PreProcessedHorseDataSet[currentAttributeLabel].fillna(attributeMode)
    i = i+1

# categorical attribute binarization

categoricalHorseDataSet = PreProcessedHorseDataSet.select_dtypes(include='object')
categoricalHorseDataSet = categoricalHorseDataSet.drop('outcome', axis=1)
categoricalHorseDataSetDummy = pd.get_dummies(categoricalHorseDataSet)
PreProcessedHorseDataSet = pd.concat([categoricalHorseDataSetDummy, PreProcessedHorseDataSet.loc[:, 'outcome']], axis=1)


# --------------------------- Decision Tree ---------------------------

AttributesHorseDataSet = PreProcessedHorseDataSet.drop('outcome', axis=1)

TargetHorseDataSet = PreProcessedHorseDataSet.loc[:, 'outcome']
# mapping 'euthanized' values to 'died' to tune fitting
TargetHorseDataSet = TargetHorseDataSet.map(lambda x: 'died' if x == 'euthanized' else x)

# label encoder
labelEncoder = preprocessing.LabelEncoder()
labelEncoder.fit(TargetHorseDataSet.values)
TargetHorseEncodedArray = labelEncoder.transform(TargetHorseDataSet.values)
TargetHorseEncodedDataSet = pd.DataFrame(TargetHorseEncodedArray, columns=['outcome'])

# split train and test data and target
AttributesHorseDataSet_train, AttributesHorseDataSet_test, TargetHorseDataSet_train, TargetHorseDataSet_test = train_test_split(AttributesHorseDataSet, TargetHorseEncodedDataSet, random_state=1)

# initialize model parameters
decisionTreeModel = tree.DecisionTreeClassifier()

# fit model using training data
decisionTreeModel.fit(AttributesHorseDataSet_train, TargetHorseDataSet_train)

# predict our test data using fitted model
TargetHorseDataSet_prediction = decisionTreeModel.predict(AttributesHorseDataSet_test)

# generate metrics

accuracyScore = metrics.accuracy_score(TargetHorseDataSet_test, TargetHorseDataSet_prediction)
print(accuracyScore)
recallScore = metrics.recall_score(TargetHorseDataSet_test, TargetHorseDataSet_prediction, average=None)
print(recallScore)
kappaScore = metrics.cohen_kappa_score(TargetHorseDataSet_test, TargetHorseDataSet_prediction)
print(kappaScore)

TargetHorseDataSet_test = labelEncoder.inverse_transform(TargetHorseDataSet_test)
TargetHorseDataSet_prediction = labelEncoder.inverse_transform(TargetHorseDataSet_prediction)

confusionMatrix = pd.DataFrame(
    metrics.confusion_matrix(TargetHorseDataSet_test, TargetHorseDataSet_prediction),
    columns=['Predicted Lived', 'Predicted Died'],
    index=['True Lived', 'True Died']
)

print(confusionMatrix)





