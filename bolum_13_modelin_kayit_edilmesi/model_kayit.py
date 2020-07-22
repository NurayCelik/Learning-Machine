# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 08:28:29 2020

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


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, Y_train)
print(lr.predict(X_test))

#bu modeli syretmek için pickle kullanıcaz
import pickle 
#dosya ismimiz model.kayit olacak
dosya = "model.kayit"
#dosya wb (write modunda ve binary) olarak açılacak
#hangi modeli diske kaydetmek istiyorsak burada lr model kullanıldı o yazıldı
pickle.dump(lr, open(dosya,'wb'))
#rb: readbinary
#yuklenen aslında lr bunu başka py dosyalarından çağırıp çalıştırabiliriz 
yuklenen = pickle.load(open(dosya,'rb'))
print(yuklenen.predict(X_test))





