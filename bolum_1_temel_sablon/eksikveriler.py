# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#ders6 : kutuphanelerin yuklenmesi
#kutuphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# kod bolumu
veriler = pd.read_csv('eksikveriler.csv')
print(veriler)
boy=veriler[['boy']]
print(boy)
boykilo = veriler[['boy','kilo']]
print(boykilo)

#◙ insan adında class olustu 
class insan:
    # insana ait boy değişkeni
    boy=180
    
    # kosmak adında fonksiyon olustu, self py e ait parametre
    def kosmak(self,b):
        return b + 10
    # y = f(x)
    # f(x) = x + 10

# insan clasından ali adında nesne olustu
ali =insan()
print(ali.boy)
print(ali.kosmak(60))

# eksik veriler  NaN Not availanable
# eksik verileri düzeltme
# sayisal veriler için kolonun ort alıp bu ortalama boş verilere yazmak en guzeli
#sci - kit learn - bilimsel makine ogrenmesi - bilimsel alet kutusu kutuphanesi 
from sklearn.preprocessing import Imputer
#imputer nesne
imputer = Imputer(missing_values='NaN', strategy = 'mean', axis=0)

Yas = veriler.iloc[:,1:4].values
print(Yas)
#Her kolon için ort değer alınıyor, satırda bir sorun yok sayısal olan 1 ve 4 kadar sutunlar sayısal
imputer = imputer.fit(Yas[:,1:4])
#degistir
Yas[:,1:4] = imputer.transform(Yas[:,1:4])
print(Yas)

