
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn import tree

import sklearn.metrics as metrics
import sklearn.preprocessing as preprocessing
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

pd.options.display.max_rows = 999


# In[3]:


# Load in the data with `read_csv()`
horsesDataSet = pd.read_csv("horse.csv", header=0, delimiter=',')

horsesDataSetTest = pd.read_csv("horseTest.csv", header=0, delimiter=',')



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
PreProcessedHorseDataSetTest = horsesDataSetTest
i = 0
while i < horsesDataSet.shape[1]:
    if percentageArray[i] > threshold:
        # get current attribute label name
        currentAttributeLabel = list(horsesDataSet)[i]
        
        # drop attribute column if na values > threshold
        PreProcessedHorseDataSet = PreProcessedHorseDataSet.drop(columns=currentAttributeLabel)
        
        #drop from test
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
    PreProcessedHorseDataSet[currentAttributeLabel] = PreProcessedHorseDataSet[currentAttributeLabel].fillna(attributeMode)
    i = i+1
    
# Altering Categorical missing values to Mode Value (value that appear the most often) [DATASET TEST]
i = 0
while i < PreProcessedHorseDataSetTest.shape[1]:
    # return the most frequent value (first index because mode() returns a DataFrame)
    attributeMode = PreProcessedHorseDataSetTest.mode().iloc[0, i]
    currentAttributeLabel = list(PreProcessedHorseDataSetTest)[i]
    PreProcessedHorseDataSetTest[currentAttributeLabel] = PreProcessedHorseDataSetTest[currentAttributeLabel].fillna(attributeMode)
    i = i+1

# categorical attribute binarization

categoricalHorseDataSet = PreProcessedHorseDataSet.select_dtypes(include='object')
categoricalHorseDataSet = categoricalHorseDataSet.drop('outcome', axis=1)
categoricalHorseDataSetDummy = pd.get_dummies(categoricalHorseDataSet)
PreProcessedHorseDataSet = pd.concat([categoricalHorseDataSetDummy, PreProcessedHorseDataSet.loc[:, 'outcome']], axis=1)


# categorical attribute binarization [DATASET TEST]

categoricalHorseDataSetTest = PreProcessedHorseDataSetTest.select_dtypes(include='object')
categoricalHorseDataSetTest = categoricalHorseDataSetTest.drop('outcome', axis=1)
categoricalHorseDataSetDummy = pd.get_dummies(categoricalHorseDataSetTest)
PreProcessedHorseDataSetTest = pd.concat([categoricalHorseDataSetDummy, PreProcessedHorseDataSetTest.loc[:, 'outcome']], axis=1)

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


# In[4]:


PreProcessedHorseDataSetTest


# In[5]:


# Convertendo objetos para categóricos


# In[6]:



i= 0
while i < PreProcessedHorseDataSet.shape[1]:
    if PreProcessedHorseDataSet[list(PreProcessedHorseDataSet)[i]].dtypes == 'O': 
        PreProcessedHorseDataSet[list(PreProcessedHorseDataSet)[i]] = PreProcessedHorseDataSet[list(PreProcessedHorseDataSet)[i]].astype('category')
    i = i+1


i= 0
while i < PreProcessedHorseDataSetTest.shape[1]:
    if PreProcessedHorseDataSetTest[list(PreProcessedHorseDataSetTest)[i]].dtypes == 'O': 
        PreProcessedHorseDataSetTest[list(PreProcessedHorseDataSetTest)[i]] = PreProcessedHorseDataSetTest[list(PreProcessedHorseDataSetTest)[i]].astype('category')
    i = i+1


# In[7]:


x_train, x_test, y_train, y_test = train_test_split(PreProcessedHorseDataSet.drop('outcome', axis = 1), PreProcessedHorseDataSet['outcome'], random_state = 0)


# In[8]:


# KNN SEM REGRESSÃO LOGÍSTICA


# In[9]:



tamanho = 100
maior_knn = 0
pos = 0
vet =[]
for i in range(1,tamanho):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(x_train, y_train)
    result = knn.score(x_test, y_test)
    vet.append(result)
    if result > maior_knn:
        maior_knn = result
        pos = i
        TargetHorseDataSet_prediction = knn.predict(x_test)
        TargetHorseDataSet_test = y_test


# In[10]:


knn = KNeighborsClassifier(n_neighbors = pos)
knn.fit(x_train, y_train)


# In[11]:


knn.score(PreProcessedHorseDataSetTest.drop('outcome', axis = 1), PreProcessedHorseDataSetTest.outcome)


# In[12]:


TargetHorseDataSet_prediction


# In[13]:


# generate metrics

accuracyScore = metrics.accuracy_score(TargetHorseDataSet_test, TargetHorseDataSet_prediction)
print(accuracyScore)
recallScore = metrics.recall_score(TargetHorseDataSet_test, TargetHorseDataSet_prediction, average=None)
print(recallScore)
kappaScore = metrics.cohen_kappa_score(TargetHorseDataSet_test, TargetHorseDataSet_prediction)
print(kappaScore)

#TargetHorseDataSet_test = labelEncoder.inverse_transform(TargetHorseDataSet_test)
#TargetHorseDataSet_prediction = labelEncoder.inverse_transform(TargetHorseDataSet_prediction)

confusionMatrix = pd.DataFrame(
    metrics.confusion_matrix(TargetHorseDataSet_test, TargetHorseDataSet_prediction),
    columns=['Predicted Died', 'Predicted Lived'],
    index=['True Died', 'True Lived']
)

print(confusionMatrix)


# In[14]:


# KNN COM REGRESSÃO LOGÍSTICA


# In[ ]:


result_lr = 0
melhor_qty = 0
resultados = []

for j in range(1,len(PreProcessedHorseDataSet)):
    model = LogisticRegression()
    feature_qty = j
    dataset = PreProcessedHorseDataSet
    rfe = RFE(model, feature_qty)
    rfe = rfe.fit(dataset.drop('outcome', axis = 1), dataset.outcome.values)

    col = []
    for i in range(0,rfe.support_.size):
        if rfe.support_[i] == True:
            col.append(i)

    dataset_after_rfe = dataset[dataset.columns[col]]
    
    x_train, x_test, y_train, y_test = train_test_split(dataset_after_rfe, dataset['outcome'], random_state = 0)
    
    tamanho = 100
    maior_knn = 0
    pos = 0
    vet =[]
    for i in range(1,tamanho):
        knn = KNeighborsClassifier(n_neighbors = i)
        knn.fit(x_train, y_train)
        result = knn.score(x_test, y_test)
        vet.append(result)
        if result > maior_knn:
            maior_knn = result
            pos = i
    
    knn = KNeighborsClassifier(n_neighbors = pos)
    knn.fit(x_train, y_train)
    
    TESTE = PreProcessedHorseDataSetTest[list(dataset[dataset.columns[col]].columns)]
    
    h = knn.score(TESTE, PreProcessedHorseDataSetTest.outcome)
    resultados.append(h)
    if h > result_lr:
        result_lr = h
        melhor_qty = j
        TargetHorseDataSet_prediction = knn.predict(x_test)
        TargetHorseDataSet_test = y_test
        
        


# In[ ]:


print(result_lr, melhor_qty)


# In[ ]:


TargetHorseDataSet_prediction


# In[ ]:


# generate metrics

accuracyScore = metrics.accuracy_score(TargetHorseDataSet_test, TargetHorseDataSet_prediction)
print(accuracyScore)
recallScore = metrics.recall_score(TargetHorseDataSet_test, TargetHorseDataSet_prediction, average=None)
print(recallScore)
kappaScore = metrics.cohen_kappa_score(TargetHorseDataSet_test, TargetHorseDataSet_prediction)
print(kappaScore)

#TargetHorseDataSet_test = labelEncoder.inverse_transform(TargetHorseDataSet_test)
#TargetHorseDataSet_prediction = labelEncoder.inverse_transform(TargetHorseDataSet_prediction)

confusionMatrix = pd.DataFrame(
    metrics.confusion_matrix(TargetHorseDataSet_test, TargetHorseDataSet_prediction),
    columns=['Predicted Died', 'Predicted Lived'],
    index=['True Died', 'True Lived']
)

confusionMatrix


# In[ ]:


# DECISION TREE


# In[ ]:


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


# In[ ]:


# generate metrics

accuracyScore = metrics.accuracy_score(TargetHorseDataSet_test, TargetHorseDataSet_prediction)
print(accuracyScore)
recallScore = metrics.recall_score(TargetHorseDataSet_test, TargetHorseDataSet_prediction, average=None)
print(recallScore)
kappaScore = metrics.cohen_kappa_score(TargetHorseDataSet_test, TargetHorseDataSet_prediction)
print(kappaScore)

#TargetHorseDataSet_test = labelEncoder.inverse_transform(TargetHorseDataSet_test)
#TargetHorseDataSet_prediction = labelEncoder.inverse_transform(TargetHorseDataSet_prediction)

confusionMatrix = pd.DataFrame(
    metrics.confusion_matrix(TargetHorseDataSet_test, TargetHorseDataSet_prediction),
    columns=['Predicted Lived', 'Predicted Died'],
    index=['True Lived', 'True Died']
)
confusionMatrix


# In[ ]:


# DECISION TREE COM REGRESSÃO LOGÍSTICA


# In[ ]:


result_lr = 0
melhor_qty = 0
resultados = []

for j in range(1,len(PreProcessedHorseDataSet)):
    model = LogisticRegression()
    feature_qty = j
    dataset = PreProcessedHorseDataSet
    rfe = RFE(model, feature_qty)
    rfe = rfe.fit(dataset.drop('outcome', axis = 1), dataset.outcome.values)

    col = []
    for i in range(0,rfe.support_.size):
        if rfe.support_[i] == True:
            col.append(i)

    dataset_after_rfe = dataset[dataset.columns[col]]
    
    
    
    # label encoder
    labelEncoder = preprocessing.LabelEncoder()
    labelEncoder.fit(TargetHorseDataSet.values)
    TargetHorseEncodedArray = labelEncoder.transform(TargetHorseDataSet.values)
    TargetHorseEncodedDataSet = pd.DataFrame(TargetHorseEncodedArray, columns=['outcome'])

    # split train and test data and target
    AttributesHorseDataSet_train, AttributesHorseDataSet_test, TargetHorseDataSet_train, TargetHorseDataSet_test = train_test_split(dataset_after_rfe, TargetHorseEncodedDataSet, random_state=1)

    # initialize model parameters
    decisionTreeModel = tree.DecisionTreeClassifier()

    # fit model using training data
    decisionTreeModel.fit(AttributesHorseDataSet_train, TargetHorseDataSet_train)

    # predict our test data using fitted model
    TargetHorseDataSet_prediction = decisionTreeModel.predict(AttributesHorseDataSet_test)
    
    
    accuracyScore = metrics.accuracy_score(TargetHorseDataSet_test, TargetHorseDataSet_prediction)
    
    if accuracyScore > result_lr:
        result_lr = accuracyScore
        melhor_qty = j
        TargetHorseDataSet_prediction_Store = decisionTreeModel.predict(AttributesHorseDataSet_test)
        TargetHorseDataSet_test_Store = TargetHorseDataSet_test


# In[ ]:


accuracyScore = metrics.accuracy_score(TargetHorseDataSet_test_Store, TargetHorseDataSet_prediction_Store)
print(accuracyScore)
recallScore = metrics.recall_score(TargetHorseDataSet_test_Store, TargetHorseDataSet_prediction_Store, average=None)
print(recallScore)
kappaScore = metrics.cohen_kappa_score(TargetHorseDataSet_test_Store, TargetHorseDataSet_prediction_Store)
print(kappaScore)

#TargetHorseDataSet_test = labelEncoder.inverse_transform(TargetHorseDataSet_test)
#TargetHorseDataSet_prediction = labelEncoder.inverse_transform(TargetHorseDataSet_prediction)

confusionMatrix = pd.DataFrame(
    metrics.confusion_matrix(TargetHorseDataSet_test_Store, TargetHorseDataSet_prediction_Store),
    columns=['Predicted Died', 'Predicted Lived'],
    index=['True Died', 'True Lived']
)

confusionMatrix
