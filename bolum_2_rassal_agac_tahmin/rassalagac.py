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

#2.Veri Onisleme

#2.1 Veri Yukleme
veriler = pd.read_csv('maaslar.csv')
print(veriler)

#data frame dilimleme(slice)
# maaslar.csv dosyasında egtitim seviyesi ve maas bağlantısı incelenecek
#Dolayısıyla unvan kısmını encoding etmeye gerek yok
x = veriler.iloc[:,1:2] #unvan kısmı alınmadı 0. kolon
y = veriler.iloc[:,2:]

# x ve y degerleri bölündü Numpy dizi(array) donusumu
X = x.values
Y = y.values

#Linear regression
#doğrusal model oluşturma
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y) #x den y yi öğren (train islemi)
#Gorselleştrime
plt.scatter(X, Y, color='red')
plt.plot(X,lin_reg.predict(X), color='blue')
plt.show()


#polynomial regresson
#nonlinear model / 2.dereceden polinom
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree= 2)
x_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y) #üslü degiskenlerden y yi (B1,B2) öğren

#nonlinear model / 4.dereceden polinom
from sklearn.preprocessing import PolynomialFeatures
poly_reg3 = PolynomialFeatures(degree= 4)
x_poly3 = poly_reg3.fit_transform(X)
lin_reg3 = LinearRegression()
lin_reg3.fit(x_poly3,y) #üslü degiskenlerden y yi (B1,B2) öğren

#Gorselleştrime
plt.scatter(X,Y, color = 'red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color = 'blue')
plt.show()

plt.scatter(X,Y, color = 'red')
plt.plot(X,lin_reg3.predict(poly_reg3.fit_transform(X)),color = 'blue')
plt.show()

#Tahminler
#Egitim Seviyesine göre
linpreg = lin_reg.predict(np.array([11]).reshape(1, 1))
print(linpreg)
linpreg2 = lin_reg.predict(np.array([6.6]).reshape(1, 1))
print(linpreg2)

#Polinom seviyesinde ğitim seviyesine gore tahmin
linpreg3 = lin_reg2.predict(poly_reg.fit_transform(np.array([11]).reshape(1, 1)))
print(linpreg3)
linpreg4 = lin_reg2.predict(poly_reg.fit_transform(np.array([6.6]).reshape(1, 1)))
print(linpreg4)


#Verilerin olceklenmesi / standartlasmasi
# standartlastirma
from sklearn.preprocessing import StandardScaler

# sc1 nesne olust 
sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(X)
sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(Y)

from sklearn.svm import SVR

svr_reg = SVR(kernel = 'rbf')
#diğer kernel fonksiyonları: linear, poly, sigmoid
svr_reg.fit(x_olcekli, y_olcekli)
sv_predict = svr_reg.predict(x_olcekli)
plt.scatter(x_olcekli, y_olcekli, color = 'red')
plt.plot(x_olcekli,sv_predict, color='blue')
plt.show()
sv_predict1 = svr_reg.predict(np.array([11]).reshape(1, 1))
print(sv_predict1)
sv_predict2 = svr_reg.predict(np.array([6.6]).reshape(1, 1))
print(sv_predict2)


from sklearn.tree import DecisionTreeRegressor
#random agacin dizilimi ile ilgili rasgeleleik
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)
Z = X + 0.5
K = X - 0.4
r_pred = r_dt.predict(X)
r_pred1 = r_dt.predict(Z)
r_pred2 = r_dt.predict(K)
plt.scatter(X,Y, color='red')
plt.plot(x,r_pred, color='blue')
plt.plot(x,r_pred1, color='green')
plt.plot(x,r_pred2, color='yellow')
plt.show()

r_predict1 = r_dt.predict(np.array([11]).reshape(1, 1))
print(r_predict1)
r_predict2 = r_dt.predict(np.array([6.6]).reshape(1, 1))
print(r_predict2)

from sklearn.ensemble import RandomForestRegressor
# n_estimators tane desicion tree cizilecek veriyi bufada 10 bölecek
rf_reg = RandomForestRegressor(n_estimators=10, random_state=0)
rf_reg.fit(X,Y)
rf_reg1 = rf_reg.predict(X)
rf_reg3 = rf_reg.predict(Z)
rf_reg4 = rf_reg.predict(K)
rf_reg2 = rf_reg.predict(np.array([6.6]).reshape(1, 1))
print(r_pred2)
plt.scatter(X,Y, color='red')
plt.plot(x,rf_reg1, color='blue')
plt.plot(x,rf_reg3, color='green')
plt.plot(x,rf_reg4, color='yellow')




















