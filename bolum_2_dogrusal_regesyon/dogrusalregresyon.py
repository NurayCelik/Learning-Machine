# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#ders6 : kutuphanelerin yuklenmesi
#1 kutuphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#2 Veri Yukleme

veriler = pd.read_csv('satislar.csv')
print(veriler)

#2.Veri Onisleme
aylar=veriler[['Aylar']]
print(aylar)

satislar = veriler[['Satislar']]
print(satislar)

#Verilerin vegitim ve test icin bolme
from sklearn.model_selection import train_test_split
#ilk bağımsız degisken aylar sonrasında bagimli degisken satislar yazildi
x_train, x_test, y_train, y_test = train_test_split(aylar, satislar, test_size=0.33, random_state=0)

'''
#verilerin olceklenmesi/standartlastirma
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test= sc.fit_transform(x_test)
Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)

#model inşası (linear regression)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
# X_train den Y_train i öğrenecek
#model inşa ediyoruz aslında
lr.fit(X_train,Y_train)
#X_test e bakıp Y_text i tahmin edecek
tahmin = lr.predict(X_test)
'''
#model inşası (linear regression)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
# x_train den y_train i öğrenecek
#model inşa ediyoruz aslında
lr.fit(x_train,y_train)
#x_test e bakıp y_text i tahmin edecek
tahmin = lr.predict(x_test)
#aylar random oldugu için sort ile indexe gore sıraldık plot icin
x_train = x_train.sort_index()

y_train = y_train.sort_index()
plt.plot(x_train,y_train)
#x_test deki her bir deger icin bu degerin karsiligi olan 
#predict in grafikte turuncu duz cizgi
#duz cizgi olmasının sebebi tekbir dogru uzerinde
# predict degerlerinin olması y= ax+b dogrusu
plt.plot(x_test,lr.predict(x_test))
plt.title("Aylara göre satış")
plt.xlabel("Aylar")
plt.ylabel("Satışlar")