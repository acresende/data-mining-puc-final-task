
# coding: utf-8

# # Import Bibliotecas

# In[85]:


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

# In[86]:


horse = pd.read_csv('horse.csv',sep=',')
Tipos = horse.dtypes #Mostra os tipos das variáveis
Estatisticas = horse.describe(include='all') #Mostra estatisticas das variáveis 


# In[87]:


horse


# In[88]:


Estatisticas


# # Tratamento Missing Values

# In[89]:


qtd_nan = [0 for x in range(horse.shape[1])]
i= 0
while i < horse.shape[1]:
    qtd_nan[i] = horse.loc[ (pd.isna(horse.iloc[:,i])) , list(horse)[i] ].shape[0]
    i = i+1


# In[90]:


#Alterando MissingValues Numéricos para o valor da média
horse_new = horse.fillna(horse.mean())
#alterando MissingValues Categoricos para o valor mais frequente
i= 0
while i < horse.shape[1]:
    horse_new[list(horse)[i]] = horse_new[list(horse)[i]].fillna(Estatisticas.iloc[2,:][i])
    i = i+1

Estatisticas_new = horse_new.describe(include='all') #Mostra estatisticas das variáveis


# In[91]:


horse_new.dtypes


# In[92]:


Estatisticas_new


# # Convertendo categóricos para numéricos

# ### Apenas mudando o tipo da variável para numerico

# In[93]:


#http://pbpython.com/categorical-encoding.html Approach #2
i= 0
while i < horse.shape[1]:
    if horse_new[list(horse)[i]].dtypes == 'O': 
        horse_new[list(horse)[i]] = horse_new[list(horse)[i]].astype('category')
    i = i+1

Estatisticas_new = horse_new.describe(include='all') #Mostra estatisticas das variáveis


# In[94]:


Estatisticas_new


# In[95]:


horse_new.dtypes


# ### Utilizando o metodo de dumies e retirando a coluna de "outcome"

# In[96]:


#http://pbpython.com/categorical-encoding.html Approach #3
horse_dummies = horse_new.drop('outcome', axis = 1)
cols = horse_dummies.columns[horse_dummies.dtypes.eq("category")] #Variavel que possui todas as colunas em formato de Objeto
horse_dummies = pd.get_dummies(horse_dummies, columns=list(cols))
Estatisticas_dummies = horse_dummies.describe(include='all') 
horse_dummies = pd.concat((horse_dummies,horse_new.outcome), axis = 1)


# In[97]:


print(horse_dummies.dtypes)


# ### Categorizando com Números

# In[98]:


#categorizando com numeros
horse_cat = horse_new.drop('outcome', axis = 1)
cols = horse_cat.columns[horse_cat.dtypes.eq("category")]
for coluna in cols:
    horse_cat[coluna] = pd.factorize(horse_cat[coluna])[0]
horse_cat = pd.concat((horse_cat,horse_new.outcome), axis = 1)


# In[99]:


horse_cat


# In[100]:


horse_cat.dtypes


# # Arrumando base de teste

# In[101]:


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

#categorizando com numeros
horseTest_cat = horseTest.drop('outcome', axis = 1)
cols = horseTest_cat.columns[horseTest_cat.dtypes.eq("category")]
for coluna in cols:
    horseTest_cat[coluna] = pd.factorize(horseTest_cat[coluna])[0]
horseTest_cat = pd.concat((horseTest_cat,horseTest.outcome), axis = 1)


# In[102]:


EstatisticasTest_dummies


# In[103]:


horseTest_dummies.dtypes


# In[104]:


horseTest_cat


# In[105]:


horseTest_cat.dtypes


# # KNN

# ### horse_dummies

# In[106]:


x_train, x_test, y_train, y_test = train_test_split(horse_dummies.drop('outcome', axis = 1), horse_dummies['outcome'], random_state = 0)


# In[107]:


print(x_train.shape)
print(x_test.shape)


# In[108]:


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
        


# In[109]:


print(maior_knn, pos)


# In[110]:


knn = KNeighborsClassifier(n_neighbors = pos)
knn.fit(x_train, y_train)


# In[111]:


knn.score(horseTest_dummies.drop('outcome', axis = 1), horseTest_dummies.outcome)


# ### horse_cat

# In[112]:


x_train, x_test, y_train, y_test = train_test_split(horse_cat.drop('outcome', axis = 1), horse_cat['outcome'], random_state = 0)


# In[113]:


print(x_train.shape)
print(x_test.shape)


# In[114]:


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
        


# In[115]:


print(maior_knn, pos)


# In[116]:


knn = KNeighborsClassifier(n_neighbors = pos)
knn.fit(x_train, y_train)


# In[117]:


knn.score(horseTest_cat.drop('outcome', axis = 1), horseTest_cat.outcome)


# # Adicionando Logistic Regression

# ### horse_dummies

# In[118]:


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
    
    x_train, x_test, y_train, y_test = train_test_split(dataset_after_rfe, dataset['outcome'], random_state = 5)
    
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
        
        


# In[119]:


print(result_lr, melhor_qty)


# In[120]:


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


# In[121]:


dataset[dataset.columns[col]].columns


# In[122]:


dataset_after_rfe.dtypes


# In[123]:


x_train, x_test, y_train, y_test = train_test_split(dataset_after_rfe, dataset['outcome'], random_state = 0)


# In[124]:


print(x_train.shape)
print(x_test.shape)


# In[125]:


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
        


# In[126]:


print(maior_knn, pos)


# In[127]:


knn = KNeighborsClassifier(n_neighbors = pos)
knn.fit(x_train, y_train)


# In[128]:


TESTE = horseTest_dummies[list(dataset[dataset.columns[col]].columns)]


# In[129]:


knn.score(TESTE, horseTest_dummies.outcome)


# ### horse_cat

# In[130]:


result_lr = 0
melhor_qty = 0
resultados = []

for j in range(1,len(horse_cat)):
    model = LogisticRegression()
    feature_qty = j
    dataset = horse_cat
    rfe = RFE(model, feature_qty)
    rfe = rfe.fit(dataset.drop('outcome', axis = 1), dataset.outcome.values)

    col = []
    for i in range(0,rfe.support_.size):
        if rfe.support_[i] == True:
            col.append(i)

    dataset_after_rfe = dataset[dataset.columns[col]]
    
    x_train, x_test, y_train, y_test = train_test_split(dataset_after_rfe, dataset['outcome'], random_state = 5)
    
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
    
    TESTE = horseTest_cat[list(dataset[dataset.columns[col]].columns)]
    
    h = knn.score(TESTE, horseTest_cat.outcome)
    resultados.append(h)
    if h > result_lr:
        result_lr = h
        melhor_qty = j
        
        


# In[131]:


print(result_lr, melhor_qty)


# In[132]:


model = LogisticRegression()
#Selecao de 10 atributos
feature_qty = melhor_qty
dataset = horse_cat
rfe = RFE(model, feature_qty)
rfe = rfe.fit(dataset.drop('outcome', axis = 1), dataset.outcome.values)

col = []
for i in range(0,rfe.support_.size):
    if rfe.support_[i] == True:
        col.append(i)

dataset_after_rfe = dataset[dataset.columns[col]]


# In[133]:


dataset_after_rfe.dtypes


# In[134]:


x_train, x_test, y_train, y_test = train_test_split(dataset_after_rfe, dataset['outcome'], random_state = 0)


# In[135]:


print(x_train.shape)
print(x_test.shape)


# In[136]:


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
        


# In[137]:


print(maior_knn, pos)


# In[138]:


knn = KNeighborsClassifier(n_neighbors = pos)
knn.fit(x_train, y_train)


# In[139]:


TESTE = horseTest_cat[list(dataset[dataset.columns[col]].columns)]


# In[140]:


knn.score(TESTE, horseTest_cat.outcome)

