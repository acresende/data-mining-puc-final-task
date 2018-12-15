
# coding: utf-8

# # Import Bibliotecas

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

pd.options.display.max_rows = 999


# # Importação base de dados

# In[2]:


horse = pd.read_csv('horse.csv',sep=',')
Tipos = horse.dtypes #Mostra os tipos das variáveis
Estatisticas = horse.describe(include='all') #Mostra estatisticas das variáveis 


# In[3]:


horse


# In[4]:


Estatisticas


# # Tratamento Missing Values

# In[5]:


qtd_nan = [0 for x in range(horse.shape[1])]
i= 0
while i < horse.shape[1]:
    qtd_nan[i] = horse.loc[ (pd.isna(horse.iloc[:,i])) , list(horse)[i] ].shape[0]
    i = i+1


# In[6]:


#Alterando MissingValues Numéricos para o valor da média
horse_new = horse.fillna(horse.mean())
#alterando MissingValues Categoricos para o valor mais frequente
i= 0
while i < horse.shape[1]:
    horse_new[list(horse)[i]] = horse_new[list(horse)[i]].fillna(Estatisticas.iloc[2,:][i])
    i = i+1

Estatisticas_new = horse_new.describe(include='all') #Mostra estatisticas das variáveis


# In[7]:


horse_new.dtypes


# In[8]:


Estatisticas_new


# # Convertendo categóricos para numéricos

# ### Apenas mudando o tipo da variável para numerico

# In[9]:


#http://pbpython.com/categorical-encoding.html Approach #2
i= 0
while i < horse.shape[1]:
    if horse_new[list(horse)[i]].dtypes == 'O': 
        horse_new[list(horse)[i]] = horse_new[list(horse)[i]].astype('category')
    i = i+1

Estatisticas_new = horse_new.describe(include='all') #Mostra estatisticas das variáveis


# In[10]:


Estatisticas_new


# In[11]:


horse_new.dtypes


# ### Utilizando o metodo de dumies e retirando a coluna de "outcome"

# In[12]:


#http://pbpython.com/categorical-encoding.html Approach #3
horse_dummies = horse_new.drop('outcome', axis = 1)
cols = horse_dummies.columns[horse_dummies.dtypes.eq("category")] #Variavel que possui todas as colunas em formato de Objeto
horse_dummies = pd.get_dummies(horse_dummies, columns=list(cols))
Estatisticas_dummies = horse_dummies.describe(include='all') 
horse_dummies = pd.concat((horse_dummies,horse_new.outcome), axis = 1)


# In[13]:


print(horse_dummies.dtypes)


# # Arrumando base de teste

# In[14]:


horseTest = pd.read_csv('horseTest.csv',sep=',')
EstatisticasTest = horseTest.describe(include='all') #Mostra estatisticas das variáveis 
#Alterando MissingValues Numéricos para o valor da média
horseTest = horseTest.fillna(horseTest.mean())
#alterando MissingValues Categoricos para o valor mais frequente
i= 0
while i < horseTest.shape[1]:
    horseTest[list(horseTest)[i]] = horseTest[list(horseTest)[i]].fillna(EstatisticasTest.iloc[2,:][i])
    i = i+1

#Categorizando
i= 0
while i < horseTest.shape[1]:
    if horseTest[list(horseTest)[i]].dtypes == 'O': 
        horseTest[list(horseTest)[i]] = horseTest[list(horse)[i]].astype('category')
    i = i+1
    
#"Dumificando"
horseTest_dummies = horseTest.drop('outcome', axis = 1)
cols = horseTest_dummies.columns[horseTest_dummies.dtypes.eq("category")] #Variavel que possui todas as colunas em formato de Objeto
horseTest_dummies = pd.get_dummies(horseTest_dummies, columns=list(cols))
EstatisticasTest_dummies = horseTest_dummies.describe(include='all') 
horseTest_dummies = pd.concat((horseTest_dummies,horseTest.outcome), axis = 1)


# In[15]:


EstatisticasTest_dummies


# In[16]:


horseTest_dummies.dtypes


# # KNN

# In[17]:


x_train, x_test, y_train, y_test = train_test_split(horse_dummies.drop('outcome', axis = 1), horse_dummies['outcome'], random_state = 0)


# In[18]:


print(x_train.shape)
print(x_test.shape)


# In[19]:


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
        


# In[20]:


print(maior_knn, pos)


# In[21]:


knn = KNeighborsClassifier(n_neighbors = pos)
knn.fit(x_train, y_train)


# In[22]:


knn.score(horseTest_dummies.drop('outcome', axis = 1), horseTest_dummies.outcome)


# # Adicionando Logistic Regression

# In[23]:


result_lr = 0
melhor_qty = 0
resultados = []

for j in range(1,len(horse_dummies)):
    model = LogisticRegression()
    feature_qty = j
    dataset = horse_dummies
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
    
    TESTE = horseTest_dummies[list(dataset[dataset.columns[col]].columns)]
    
    h = knn.score(TESTE, horseTest_dummies.outcome)
    resultados.append(h)
    if h > result_lr:
        result_lr = h
        melhor_qty = j
        
        


# In[24]:


print(result_lr, melhor_qty)


# In[25]:


model = LogisticRegression()
#Selecao de 10 atributos
feature_qty = melhor_qty
dataset = horse_dummies
rfe = RFE(model, feature_qty)
rfe = rfe.fit(dataset.drop('outcome', axis = 1), dataset.outcome.values)

col = []
for i in range(0,rfe.support_.size):
    if rfe.support_[i] == True:
        col.append(i)

dataset_after_rfe = dataset[dataset.columns[col]]


# In[26]:


dataset[dataset.columns[col]].columns


# In[27]:


dataset_after_rfe.dtypes


# In[28]:


x_train, x_test, y_train, y_test = train_test_split(dataset_after_rfe, dataset['outcome'], random_state = 0)


# In[29]:


print(x_train.shape)
print(x_test.shape)


# In[30]:


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
        


# In[31]:


print(maior_knn, pos)


# In[32]:


knn = KNeighborsClassifier(n_neighbors = pos)
knn.fit(x_train, y_train)


# In[33]:


TESTE = horseTest_dummies[list(dataset[dataset.columns[col]].columns)]


# In[34]:


knn.score(TESTE, horseTest_dummies.outcome)

