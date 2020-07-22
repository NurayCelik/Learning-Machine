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
#print(veriler)

degerler=veriler[['boy','kilo','yas','cinsiyet']]
#encoder: Nominal Ordinal(Kategorik) -> Numeric
# Sayisal olmayan kategorik verilerin sayisala çevrilmesi
# Eger 2 kategori ise evet hayır 1 ya da sıfır 
# eger birden çok kategori varsa t,fr,usa, gr ülkeler gibi 
# Bunların sayısala cevrilemsi

# tum satırlarn ilk sutunu al
ulke=veriler.iloc[:,0:1].values
print(ulke)

# LabelEncoder herbir deger için sayisal deger olusturur 
# verilerde 3 ulke var 1,2,3 diye atar
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

# ohe = OneHotEncoder(categorical_features = 'all')
ohe = OneHotEncoder('auto')
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)
# cinsiyet ->Numerik
c=veriler.iloc[:,-1:].values
print(c)


# LabelEncoder herbir deger için sayisal deger olusturur 
from sklearn.preprocessing import LabelEncoder
# le nsene LabelEncoder'ın
le = LabelEncoder()
# 0.kolonu al 
c[:,0] = le.fit_transform(c[:,0])
print(c)

# Burada 3 ulke için 3 indisli dizi olusur. 
# herbir ulkeyi bir indise 1 olarak diğer ulkeleri 0 olarak atar
from sklearn.preprocessing import OneHotEncoder

# ohe = OneHotEncoder(categorical_features = 'all')
ohe = OneHotEncoder('auto')
c = ohe.fit_transform(c).toarray()
print(c)

sonuc = pd.DataFrame(data = ulke, index=range(22), columns=['fr','tr','us'] )
print(sonuc)

sonuc2 = pd.DataFrame(data=degerler, index= range(22),columns = ['boy','kilo','yas'])
print(sonuc2)

# son kolonu al
cinsiyet = veriler.iloc[:,-1].values
print(cinsiyet)

sonuc3 = pd.DataFrame(data = c[:,:1], index= range(22),columns = ['cinsiyet'])
print(sonuc3)

#pandas kutuphanesinde concat kutuphanesi ile 
#satır bazlı verileri birlestirme
s = pd.concat([sonuc, sonuc2], axis=1)
print(s)

s2 = pd.concat([s,sonuc3], axis=1)
print(s2)

#Verilerin egitim ve test icin bolme
from sklearn.model_selection import train_test_split

# test %0.33 train %0.66
x_train, x_test, y_train, y_test = train_test_split(s, sonuc3, test_size=0.33, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
#x_frainden y_train i ögren
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
#tüm stirlarin 3.koln ile4. kolon arasını al
boy = s2.iloc[:,3:4].values
print(boy)
#s2 nin 3.satirina kadar al
sol = s2.iloc[:,:3]
#s2 nin 3. satirdan sonrasini al
sag = s2.iloc[:,4:]

veri = pd.concat([sol,sag],axis=1)
# test %0.33 train %0.66
#bagimsiz degisken veri leri bagimli degisken boy a gore bol
x_train, x_test, y_train, y_test = train_test_split(veri, boy, test_size=0.33, random_state=0)

regressor1 = LinearRegression()
#x_frainden y_train i ögren
regressor1.fit(x_train, y_train)
y_pred = regressor1.predict(x_test)

#Basarimi göstermek
import statsmodels.api as sm
#basa  22 tane 1 eklenir sebebi carpanin 1 olmasi B0 degeri
#ones 1 lerden olusan dizi olusturur, 22 satir 1 kolon
X = np.append(arr = np.ones((22,1)).astype(int), values=veri, axis=1 )
#Ybütün satirlari al dizi olarak hepsini yazdık cunku daha sonrai işlemdlerde bazılarını cıkrarmak için dizi olarak kolanları yazmasak da olurdu.
X_l = veri.iloc[:,[0,1,2,3,4,5]].values
#1 lerin boy kolonu üerindeki etkisini olcuyoruz, baoy bagimli X_l bagimsiz degisken

model = sm.OLS(boy,X_l).fit()
print(model.summary())

#calistirinca x1 2 3 x4 x5 x6 ccolonlarını görüyoruz.
#backward elimante göre en büyük p ye sahip kolon çıkarılacak
#bakıyoruz x5 en yüke p ye sahip onu cıkarıcagız yani dizide kodda 4. elemean

X_l = veri.iloc[:,[0,1,2,3,5]].values
model = sm.OLS(boy,X_l).fit()
print(model.summary())

#5 elamnın p degeri de buyuk onu da cıkartıyoruz

X_l = veri.iloc[:,[0,1,2,3]].values
model = sm.OLS(boy,X_l).fit()
print(model.summary())


















