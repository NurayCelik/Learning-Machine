# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 07:15:37 2020

@author: casper
"""

#1. kutuphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#2 Veri Yukleme

veriler = pd.read_csv('Wine.csv')

# Bu örnekte makine öğrenmesi algoritması ile son kolondaki müsteri segmentasyonu tahmin edilecek
# 13 bağımsız kolon daha az sayıya indirgenecek
X = veriler.iloc[:,0:13].values #bagımsiz degiskenler
y = veriler.iloc[:,13].values #bagimli degisken


#Verilerin egitim ve test icin bolme
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# standartlastirma
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test= sc.fit_transform(x_test)

# PCA 
from sklearn.decomposition import  PCA
# indirgenecek boyut 2
pca = PCA(n_components=2)

# yeni boyuta göre hem egitecek hem transform edicek
X_train2 = pca.fit_transform(X_train) 
# yeni olusan 2 boyutlu uzayda yeni x ve y koordinatlarina X_test donusturecegiz
# X_trainde bulmuş oldugumuz boyutlari X_test de de boyutlandirmis oluyoruz
X_test2 = pca.transform(X_test)

#pca donusumunden once gelen LR
from sklearn.linear_model import  LogisticRegression
#random_state = 0 olması her çalıştığında aynı fonksiyon çalışsın diye
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

#pca donusumunden sonra gelen LR
classifier2 = LogisticRegression(random_state=0)
classifier2.fit(X_train2, y_train)

#Tahminler
y_pred  = classifier.predict(X_test)
y_pred2 = classifier2.predict(X_test2)

from sklearn.metrics import confusion_matrix
#actual / PCA olmadan çıkan sonuç
print('actual(gerçek) / PCA sız')
cm = confusion_matrix(y_test, y_pred)
print(cm)

#actual / PCA sonrası çıkan sonuç
print('gerçek / PCA ile')
cm2 = confusion_matrix(y_test, y_pred2)
print(cm2)

#PCA sonrası / PCA öncesi
print('PCA sız ve PCA lı')
cm3 = confusion_matrix(y_pred, y_pred2)
print(cm3)











