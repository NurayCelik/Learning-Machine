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
from sklearn.metrics import r2_score
import statsmodels.api as sm

#2.Veri Onisleme

#2.1 Veri Yukleme
veriler = pd.read_csv('maaslar_yeni.csv')
print(veriler)

#data frame dilimleme(slice)
# maaslar.csv dosyasında egitim seviyesi ve maas bağlantısı incelenecek
#Dolayısıyla unvan kısmını encoding etmeye gerek yok
x = veriler.iloc[:,2:5] #unvan kısmı alınmadı 0. kolon
y = veriler.iloc[:,5:]
# id makine ogrenmesine alınmaz, 11. elemanı maası 50000 diye ezberlerse yanlış sonuc cıkar. 

# x ve y degerleri bölündü Numpy dizi(array) donusumu
X = x.values
Y = y.values

#pandas correlasyon (matrisi) fonk yazdırıyor
#bağımsız degiskenler arsındaki ilişkiyide, tahmini de veriyor
#Hangi kolonların birbirleriyle ilişkisini daha net görebiliriz.Satır sütn ilişkisini verir.
# puan ile maasın ilişkisi 0.201474 tahmini
print(veriler.corr())

#Linear regression
#doğrusal model oluşturma
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y) #x den y yi öğren (train islemi)

r2_err6 = r2_score(Y, lin_reg.predict(X)) # sonuc 1'e yaklaştıkça hata azdır. Cunku tahmin dogruya yakın olur

# OLS p values leri gordugumuz tablo
print("OLS Linear Regresyon")
model1 = sm.OLS(lin_reg.predict(X),X)
sonuc1 = model1.fit().summary()
print(sonuc1)

print("Linear Regresyon R2 degeri")
print(r2_err6)

#polynomial regresson
#nonlinear model / 4.dereceden polinom
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree= 4)
x_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y) #üslü degiskenlerden y yi (B1,B2) öğren

#Tahminler
# OLS p values leri gordugumuz tablo
print("OLS Polynomial Regresyon")
model2 = sm.OLS(lin_reg2.predict(x_poly),X)
sonuc2 = model2.fit().summary()
print(sonuc2)
r2_err5 = r2_score(Y, lin_reg2.predict(poly_reg.fit_transform(X))) # sonuc 1'e yaklaştıkça hata azdır. Cunku tahmin dogruya yakın olur
print("Polynomial Regresyon R2 degeri")
print(r2_err5)


#Verilerin olceklenmesi / standartlasmasi
# standartlastirma
from sklearn.preprocessing import StandardScaler

# sc1 nesne olust 
sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(X)
sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(Y)

#Support Vector Regresyon
from sklearn.svm import SVR

svr_reg = SVR(kernel = 'rbf')
#diğer kernel fonksiyonları: linear, poly, sigmoid
svr_reg.fit(x_olcekli, y_olcekli)
sv_predict = svr_reg.predict(x_olcekli)

print("OLS SVR")
model3 = sm.OLS(svr_reg.predict(x_olcekli),x_olcekli)
sonuc3 = model3.fit().summary()
print(sonuc3)

r2_err4 = r2_score(y_olcekli, sv_predict) # sonuc 1'e yaklaştıkça hata azdır. Cunku tahmin dogruya yakın olur
print("Support Vector Regresyon R2 degeri")
print(r2_err4)

#Decision Tree Regresyon
from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)
r_pred = r_dt.predict(X)

print("OLS Decision Tree")
model4 = sm.OLS(r_dt.predict(X),X)
sonuc4 = model4.fit().summary()
print(sonuc4)

r2_err3 = r2_score(Y, r_pred) # sonuc 1'e yaklaştıkça hata azdır. Cunku tahmin dogruya yakın olur. 
# Decision tree de sonucun Hatalı cıkması yuksek o nedenle başka degerlerle denemek gerek, 
#o nedenle random tree tercih edilir. 
print("Decision Tree R2 degeri")
print(r2_err3)

#Random Forest Regresyon
from sklearn.ensemble import RandomForestRegressor
# n_estimators tane desicion tree cizilecek veriyi bufada 10 bölecek
rf_reg = RandomForestRegressor(n_estimators=10, random_state=0)
rf_reg.fit(X,Y)

print("OLS Random Forest")
model5 = sm.OLS(rf_reg.predict(X),X)
sonuc5 = model5.fit().summary()
print(sonuc5)

rf_reg1 = rf_reg.predict(X)
r2_err = r2_score(Y, rf_reg1) # sonuc 1'e yaklaştıkça hata azdır. Cunku tahmin dogruya yakın olur
print("Random Forest R2 degeri")
print(r2_err)

# Ozet R2 Degerleri
print("--------------------------------")
r2_err6 = r2_score(Y, lin_reg.predict(X)) # sonuc 1'e yaklaştıkça hata azdır. Cunku tahmin dogruya yakın olur
print("Linear Regresyon R2 degeri")
print(r2_err6)

r2_err5 = r2_score(Y, lin_reg2.predict(poly_reg.fit_transform(X))) # sonuc 1'e yaklaştıkça hata azdır. Cunku tahmin dogruya yakın olur
print("Polynomial Regresyon R2 degeri")
print(r2_err5)

r2_err4 = r2_score(y_olcekli, sv_predict) # sonuc 1'e yaklaştıkça hata azdır. Cunku tahmin dogruya yakın olur
print("Support Vector Regresyon R2 degeri")
print(r2_err4)

r2_err3 = r2_score(Y, r_pred) # sonuc 1'e yaklaştıkça hata azdır. Cunku tahmin dogruya yakın olur
print("Decision Tree R2 degeri")
print(r2_err3)

r2_err = r2_score(Y, rf_reg1) # sonuc 1'e yaklaştıkça hata azdır. Cunku tahmin dogruya yakın olur
print("Random Forest R2 degeri")
print(r2_err)



