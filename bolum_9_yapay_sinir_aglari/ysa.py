# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
'''
Anaconda Prompt'u açın ve kütüphaneleri şu sırayla yükleyin:

1-Keras: conda install -c conda-forge/label/cf201901 keras

2-Tensorflow: pip3 install –upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.8.0-py3-none-any.whl

3- Theano: conda install theano
'''



#Müşterinin terk edip etmeyeceği
#1. kutuphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#2 Veri Yukleme

veriler = pd.read_csv('Churn_Modelling.csv')
# excel dosylarını calistirma
#veriler = pd.read_excel('veriler.xls') 
#print(veriler)

#Bu örnek ilk 3 kolon id, ad soyad gibi makine öğrenmesinde ezbere kaçacak veriler alınmıyor, ilk 3 kolon alınmadı.
X = veriler.iloc[:,3:13].values #bagımsiz degiskenler
Y = veriler.iloc[:,13].values #bagimli degisken

#encoder : Kategorik -> Numeric
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
# X in 1. kolonu ( ülkeler)alınacak label encoding yapılacak fit transform edilecek 1. kolon
X[:,1] = le.fit_transform(X[:,1])

# X in 2.kolonu alınacak cinsiyet encode edilecek
le2 = LabelEncoder()
X[:,2] = le2.fit_transform(X[:,2])
#label encoding ile ülkeler 1 2 3 diye sıralanır, fakat fit edilirken makina 3'u daha buyuk 1. ülkeyi daha kucuk algılayablir bunu önlemek için one hot encoder yaptık
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [1])],   # The column numbers to be transformed (here is [1] but can be [0, 1, 3])
    remainder='passthrough'                                         # Leave the rest of the columns untouched
)
X = ct.fit_transform(X)
X = X[:,1:] # X'in bütün elemanlarının 1'den sonrası

#Verilerin egitim ve test icin bolme
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=0)

# standartlastirma
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test= sc.fit_transform(x_test)

# 3- YAPAY SİNİR AĞI 
#keras tensorflowun ust mimarisi
import keras
#yapay sinir agı olusuyor sequential ile
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
# giriş ve gizli katmanlarda lineaar fonks ları kullanmak çıkış katmanında sigmoid fonk kullanmak genel kural
# Bu örnek çıkış katmanında bir tane giriş katmanında ise 10 tane nöron var
# init: initial et, ilklendir 
# 6 nöron gizli ktaman: 11+1/2 genel olarak böyle, bağımsız degiskenler+bağımlı degişken / 2 gizli katman eklendi
# input_dim: giriş katman bağımsız degişkenler
classifier.add(Dense(6, init='uniform', activation='relu', input_dim=11))
# 6 nörondan oluşan yine gizli katman eklendi
classifier.add(Dense(6, init='uniform', activation='relu'))
# 1 nöron çıkış katmanında 1 bağımlı 
classifier.add(Dense(1, init='uniform', activation='sigmoid'))
# binary_crossentropy çünkü bağımsız değişken son çıkış katmanı 1 ve 0 lardan oluşuyor, 2 den çok değer varsa categorical_cossentropy kullanılır, eğer çok fazla boşuklardan oluşuyorsa spaerse_categorical_crossentropy kullanılır
#1 ler 1, 0 lar 0 tahmin edilmesi accuracy(başarı)
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
classifier.fit(X_train, y_train, epochs=50) #X_trainden y_train i ogren
y_pred = classifier.predict(X_test)

y_pred = (y_pred >0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print('diagon her zaman doğruları verir. 00 olanlar 0 iken dogru çıkanlar, 11 ise 1 ken dogru çıkanlar')
print(cm)




















