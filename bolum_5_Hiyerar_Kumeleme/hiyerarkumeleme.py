#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 08:10:59 2018

@author: sadievrenseker
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv('musteriler.csv')

X = veriler.iloc[:,3:].values

from sklearn.cluster import KMeans

kmeans = KMeans ( n_clusters = 3, init = 'k-means++')
kmeans.fit(X)

print(kmeans.cluster_centers_)
sonuclar = []
for i in range(1,11):
    kmeans = KMeans (n_clusters = i, init='k-means++', random_state= 123)
    kmeans.fit(X)
    #kmeans ne kadar basarılı oldugu sonuclarda
    sonuclar.append(kmeans.inertia_)

#plot grafige bakarak cluster degeri blirlenir. 2, 3 ya da 4 degeri alınabilir. Dirsek noktasında
plt.plot(range(1,11),sonuclar)
plt.show()

kmeans = KMeans (n_clusters = 4, init='k-means++', random_state= 123)
Y_tahmin = kmeans.fit_predict(X)
print(Y_tahmin)
plt.scatter(X[Y_tahmin==0,0], X[Y_tahmin==0,1], s=100, c='red')
plt.scatter(X[Y_tahmin==1,0], X[Y_tahmin==1,1], s=100, c='blue')
plt.scatter(X[Y_tahmin==2,0], X[Y_tahmin==2,1], s=100, c='yellow')
plt.scatter(X[Y_tahmin==3,0], X[Y_tahmin==3,1], s=100, c='green')
plt.title('KMeans Sonucları')
plt.show()


#Hiyerarşik
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
Y_tahmin = ac.fit_predict(X)
print(Y_tahmin)

plt.scatter(X[Y_tahmin==0,0], X[Y_tahmin==0,1], s=100, c='red')
plt.scatter(X[Y_tahmin==1,0], X[Y_tahmin==1,1], s=100, c='blue')
plt.scatter(X[Y_tahmin==2,0], X[Y_tahmin==2,1], s=100, c='yellow')
plt.scatter(X[Y_tahmin==3,0], X[Y_tahmin==3,1], s=100, c='green')
plt.title('Hiyerarşik Sonuclar')
plt.show()

#hiyerarşik  kumeleme görselleştirilme
import scipy.cluster.hierarchy as sch 
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.show()






