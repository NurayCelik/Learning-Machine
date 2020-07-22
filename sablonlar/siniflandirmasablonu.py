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
# excel dosylarını calistirma
#veriler = pd.read_excel('veriler.xls') 
print(veriler)

#son colon bağımli öncekiler bagimsiz değişken genelde. Verileri incelemeke gerek
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

#SINIFLANDIRMA ALGORITMALARI

# 1-LOGISTIC ALGORITMASI-SIGMOID

from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train) #EGİTIM (X_train den y_train i ogren)

y_pred = logr.predict(X_test) #tahmim
print(y_pred)
print(y_test)
print('LOGIC')
#Bu 2x2 confusion (karmasıklık) matrisde 11(satır sutun) + 22(satır sutun) toplamı doğru tahmini
# 12(satır sutun) ve 21(satır sutun) matristeki sayıalrın toplamı yanlıs tahmin
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test ,y_pred)
print(cm)

# 2- K-NN ALGORITMASI

from sklearn.neighbors import KNeighborsClassifier
#komsu sayısı artarsa hata artar, azalırsa hata azalır
knn = KNeighborsClassifier(n_neighbors=1, metric='minkowski')
knn.fit(X_train, y_train) #EGİTIM (X_train den y_train i ogren)
y_pred = knn.predict(X_test) #tahmin
print('K-NN')
#Bu 2x2 confusion (karmasıklık) matrisde 11(satır sutun) + 22(satır sutun) toplamı doğru tahmini
# 12(satır sutun) ve 21(satır sutun) matristeki sayıalrın toplamı yanlıs tahmin
cm = confusion_matrix(y_test,y_pred)
print(cm)


# 3- SUPPORT VECTOR (SVC) ALGORITMASI

from sklearn.svm import SVC
svc = SVC(kernel='rbf')
#linear, rbf, poly, rbf
svc.fit(X_train,y_train)
y_pred = svc.predict(X_test)
#Bu 2x2 confusion (karmasıklık) matrisde 11(satır sutun) + 22(satır sutun) toplamı doğru tahmini
# 12(satır sutun) ve 21(satır sutun) matristeki sayıalrın toplamı yanlıs tahmin
cm = confusion_matrix(y_test,y_pred)
print('SVC')
print(cm)

# 4- NAIVE BAYES ALGORITMASI

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

#Bu 2x2 confusion (karmasıklık) matrisde 11(satır sutun) + 22(satır sutun) toplamı doğru tahmini
# 12(satır sutun) ve 21(satır sutun) matristeki sayıalrın toplamı yanlıs tahmin
cm = confusion_matrix(y_test,y_pred)
print('GNB')
print(cm)

# 5- KARAR AGACI ALGORTMASI

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion='entropy')
dtc.fit(X_train,y_train)
y_pred = dtc.predict(X_test)

#Bu 2x2 confusion (karmasıklık) matrisde 11(satır sutun) + 22(satır sutun) toplamı doğru tahmini
# 12(satır sutun) ve 21(satır sutun) matristeki sayıalrın toplamı yanlıs tahmin
cm = confusion_matrix(y_test,y_pred)
print('DTC')
print(cm)

# 6- RASSAL(RANDOM) AGACI ALGORITMASI 

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10, criterion='entropy')
rfc.fit(X_train,y_train)
y_pred = rfc.predict(X_test)
#print(y_pred)
#print(y_test)

#Bu 2x2 confusion (karmasıklık) matrisde 11(satır sutun) + 22(satır sutun) toplamı doğru tahmini
# 12(satır sutun) ve 21(satır sutun) matristeki sayıalrın toplamı yanlıs tahmin
cm = confusion_matrix(y_test,y_pred)
print('RFC')
print(cm)
print('y_test')
print(y_test)


# ROC, TPR, FPR degerleri

y_proba = rfc.predict_proba(X_test)# tahmin olasılıklarını verir, X_test in tahmin olasılıkları
print('iki kolonda -1. kolon dogruluk, 2. kolon yanlış olasılıklar- y_testin olasılık degerleri')
print(y_proba)
print('y_probanın tek dogru kolonda y_test icin olasılık degerleri')#print(y_proba) #Dogru yanlıs 2 kolon gelir % degerleri 
print(y_proba[:,0]) #Doğru tahmin kolonu gelecek.1 olsaydı yanlış kolon degerleri.


from sklearn import metrics 
fpr, tpr, thold = metrics.roc_curve(y_test,y_proba[:,0],pos_label='e') #positifler erkek e
print('False Positive Rate ')
print(fpr)
print('True Positive Rate ')
print(tpr)










































