# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 08:49:53 2020

@author: nuraycelik
"""

import pandas as pd
#pandas kütüphanesi url den dosya okutur

url ="https://bilkav.com/satislar.csv"
veriler = pd.read_csv(url)
#datafame olduğu için .values ile array e çevirdik
veriler = veriler.values
#bütün satrıları al 0 dan 1'e kadar olamları
#ayları alıyoruz
X = veriler[:,0:1]
#1. kolonu al
Y  = veriler[:,1]

#test yüzde 33 alınacak
bolme = 0.33

from sklearn import  model_selection
# X ve Y yi yüzde 33 olarak bol
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=bolme)

import pickle

#model.kayit olarak diske kayitli model dosyasi okunacak

yuklenen = pickle.load(open("model.kayit",'rb'))
print(yuklenen.predict(X_test))