# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#ders6 : kutuphanelerin yuklenmesi
#1. kutuphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#2 Veri Yukleme

veriler = pd.read_csv('veriler.csv')
print(veriler)

#ilk 5 satırı almadık çocuk yaşları, outline veri olduğu için
x = veriler.iloc[:,1:4].values #bağımsız değişkenler
y = veriler.iloc[:,4:].values #bağımlı degisken
print(y)
#Verilerin egitim ve test icin bolme
from sklearn.model_selection import train_test_split

# test %0.33 train %0.66
# random olarak 2/3 train(egitim), 1/3 test verilerden alacak
# o nedenle her başarım sonrası random olarak aldıgindan farklı %90 bazen de  %95 sonuc cıkar
#train in  alınacağı s, bagimsiz degiskenin alinacagi sonuc3
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

#Verilerin olceklenmesi / standartlasmasi
# standartlastirma
from sklearn.preprocessing import StandardScaler

# sc nesne olust 
sc = StandardScaler()
X_train = sc.fit_transform(x_train) #ogren transfom et
X_test= sc.fit_transform(x_test) #ogrenme transform et

from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train)

y_pred = logr.predict(X_test)
print(y_pred)
print(y_test)
print('LOGIC')
#Bu 2x2 matrisde sag kosegenlerin toplamı dogru sola dogro kosgene toplami yanlıs tahmin veririyor
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test ,y_pred)
print(cm)

from sklearn.neighbors import KNeighborsClassifier
#komsu sayısı artarsa hata artar, azalırsa hata azalır
knn = KNeighborsClassifier(n_neighbors=1, metric='minkowski')
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print('K-NN')
#Bu 2x2 matrisde sag kosegenlerin toplamı dogru sola dogro kosgene toplami yanlıs tahmin veririyor
cm = confusion_matrix(y_test,y_pred)
print(cm)

from sklearn.svm import SVC
svc = SVC(kernel='rbf')
#linear, rbf, poly, rbf
svc.fit(X_train,y_train)
y_pred = svc.predict(X_test)
#Bu 2x2 matrisde sag kosegenlerin toplamı dogru sola dogro kosgene toplami yanlıs tahmin veririyor
cm = confusion_matrix(y_test,y_pred)
print('SVC')
print(cm)























