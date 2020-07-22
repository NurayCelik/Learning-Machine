#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 04:18:20 2018

@author: sadievrenseker
"""

#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder

#2. Veri Onisleme

#2.1. Veri Yukleme
veriler = pd.read_csv('odev_tenis.csv')
#pd.read_csv("veriler.csv")

#veri on isleme
'''
#Bu kodları labelencoding yapma kısayolu apply etmek, o yapıldıgi icin yourm satırı oldu bu kodlar 

#encoder:  Kategorik -> Numeric
play = veriler.iloc[:,-1:].values  # -1 den sona kadar degerleri al
print(play)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
play[:,0] = le.fit_transform(play[:,0])
print(play)

#encoder:  Kategorik -> Numeric
windy = veriler.iloc[:,-2:-1].values  # sondan 2. kolonu al
print(windy)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
windy[:,0] = le.fit_transform(windy[:,0])
print(windy)
'''

#Burada sıkıntı tüm kolonlarda labelencoding oluyor,duzeltmek için şunlar yapıldı:
# 1.sutun birden çok kategeri old. için OneHotÊncoder yaoıldı
# Sayisal colonlar olan sıcaklık ve nem colonları ise veri olarak çekilecek

veriler2 = veriler.apply(LabelEncoder().fit_transform)

#Birden çok kategori var 1. kolonda onun icin OneHotEncoder kullanıldı
c = veriler2.iloc[:,:1] # tum satırların 1 e kadar kolonları al
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder('auto')
c=ohe.fit_transform(c).toarray()
print(c)

#sayisala donusen 0. colon alındı
havadurumu = pd.DataFrame(data = c, index = range(14), columns=['o','r','s'])
 #orjinal sıcaklık ve nem  1. ve 2. kolon degerleri alındı 0, kolonla birlestirildi
sonveriler = pd.concat([havadurumu,veriler.iloc[:,1:3]],axis = 1)
#sondan 2 kolonla birlestirildi sıcaklık ve nem son kolon olarak duzenlendi
sonveriler = pd.concat([veriler2.iloc[:,-2:],sonveriler], axis = 1) 


#verilerin egitim ve test icin bolunmesi
#humidıty nem bagımlı digerleri bagımsız degisken
from sklearn.model_selection import train_test_split
#son kolon harici bütün kolonları al sonveriler.iloc[:,:-1], bağımsız değişken ilk parametre
#son kolon al sonveriler.iloc[:,-1:] nem, bağımlı değişken 2. parametre
#x_train ve x_test bağımlı değişken olmuyor, y_train ve y_text bağımlı değişken içeriyor.
x_train, x_test,y_train,y_test = train_test_split(sonveriler.iloc[:,:-1],sonveriler.iloc[:,-1:],test_size=0.33, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)
print(y_pred)

import statsmodels.api as sm 
X = np.append(arr = np.ones((14,1)).astype(int), values=sonveriler.iloc[:,:-1], axis=1 )
X_l = sonveriler.iloc[:,[0,1,2,3,4,5]].values
r_ols = sm.OLS(sonveriler.iloc[:,-1:], X_l) #son kolonu al bağımlı degiskenin nem oldugu
r = r_ols.fit()
print(r.summary())

# 0. kolon alınmadı
sonveriler = sonveriler.iloc[:,1:]

import statsmodels.api as sm 
X = np.append(arr = np.ones((14,1)).astype(int), values=sonveriler.iloc[:,:-1], axis=1 )
X_l = sonveriler.iloc[:,[0,1,2,3,4]].values
r_ols = sm.OLS(sonveriler.iloc[:,-1:], X_l)
r = r_ols.fit()
print(r.summary())

x_train = x_train.iloc[:,1:] # p degeri yuksek olan 0. kolon burada da atıldı
x_test = x_test.iloc[:,1:] # p degeri yuksek olan 0. kolon burada da atıldı

regressor.fit(x_train,y_train)


y_pred1 = regressor.predict(x_test)

# 0 ve 1. kolon alınmadı
sonveriler = sonveriler.iloc[:,2:]

import statsmodels.api as sm 
X = np.append(arr = np.ones((14,1)).astype(int), values=sonveriler.iloc[:,:-1], axis=1 )
X_l = sonveriler.iloc[:,[0,1,2,3]].values
r_ols = sm.OLS(sonveriler.iloc[:,-1:], X_l)
r = r_ols.fit()
print(r.summary())

x_train = x_train.iloc[:,2:] # p degeri yuksek olan 1. kolon burada da atıldı
x_test = x_test.iloc[:,2:] # p degeri yuksek olan 1. kolon burada da atıldı

regressor.fit(x_train,y_train)


y_pred2 = regressor.predict(x_test)





