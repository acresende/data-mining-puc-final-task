# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 18:06:05 2018

@author: Gabriel
"""

import pandas as pd
import numpy as np

horse = pd.read_csv('horse.csv',sep=',')
Tipos = horse.dtypes #Mostra os tipos das variáveis
Estatisticas = horse.describe(include='all') #Mostra estatisticas das variáveis 

#Missing Values
qtd_nan = [0 for x in range(horse.shape[1])]
i= 0
while i < horse.shape[1]:
    qtd_nan[i] = horse.loc[ (pd.isna(horse.iloc[:,i])) , list(horse)[i] ].shape[0]
    i = i+1
    
#Tentar jogar o vetor qtd_nan dentro do DF Estatisticas
    
#Alterando MissingValues Numéricos para o valor da média
horse_new = horse.fillna(horse.mean())
#alterando MissingValues Categoricos para o valor mais frequente
i= 0
while i < horse.shape[1]:
    horse_new[list(horse)[i]] = horse_new[list(horse)[i]].fillna(Estatisticas.iloc[2,:][i])
    i = i+1

Estatisticas_new = horse_new.describe(include='all') #Mostra estatisticas das variáveis
    










