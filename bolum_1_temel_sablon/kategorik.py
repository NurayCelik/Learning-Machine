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
# sci - kit learn - bilimsel makine ogrenmesi - bilimsel alet kutusu kutuphanesi 
from sklearn.preprocessing import Imputer
# imputer nesne
imputer = Imputer(missing_values='NaN', strategy = 'mean', axis=0)

Yas = veriler.iloc[:,1:4].values
print(Yas)
# Her kolon için ort değer alınıyor, satırda bir sorun yok sayısal olan 1 ve 4 kadar sutunlar sayısal
imputer = imputer.fit(Yas[:,1:4])
# degistir
Yas[:,1:4] = imputer.transform(Yas[:,1:4])
print(Yas)

# Sayisal olmayan kategorik verilerin sayisala çevrilmesi
# Eger 2 kategori ise evet hayır 1 ya da sıfır 
# eger birden çok kategori varsa t,fr,usa, gr ülkeler gibi 
# Bunların sayısala cevrilemsi

# tum satırlarn ilk sutunu al
ulke=veriler.iloc[:,0:1].values
print(ulke)

# LabelEncoder herbir deger için sayisal deger olusturur 
# verilerde 3 ulke var 1,2,3 iye atar
from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder # Bu sekilde de iki ayrı class eklenebilirdi

# le nsene LabelEncoder'ın
le = LabelEncoder()
# 0.kolonu al 
ulke[:,0] = le.fit_transform(ulke[:,0])
print(ulke)

# Burada 3 ulke için 3 indisli dizi olusur. 
# herbir ulkeyi bir indise 1 olarak diğer ulkeleri 0 olarak atar
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(categorical_features = 'all')
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)

# indis 22 ye kadar
print(list(range(22)))

# sonuc pandas kutuphansesinden nesne 
# data numpy kutuphanesinden
# ulkenin basina index diye kolon ekleyip sayisal degerleri range ile elde etmek
#colums baslarina ulke adlari eklendi
sonuc = pd.DataFrame(data = ulke, index=range(22), columns=['fr','tr','us'] )
print(sonuc)

sonuc2 = pd.DataFrame(data=Yas, index= range(22),columns = ['boy','kilo','yas'])
print(sonuc2)

# son kolonu al
cinsiyet = veriler.iloc[:,-1:].values
print(cinsiyet)

sonuc3 = pd.DataFrame(data = cinsiyet, index= range(22),columns = ['cinsiyet'])
print(sonuc3)

#pandas kutuphanesinde concat kutuphanesi ile satır bazlı verileri birlestirme
s = pd.concat([sonuc, sonuc2],axis=1)
print(s)

s2 = pd.concat([s,sonuc3], axis=1)
print(s2)