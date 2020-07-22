#Kurulum 
#conda install py-xgboost
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 10:03:40 2018

@author: nuraycelik
"""

#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2. Veri Onisleme

#2.1. Veri Yukleme
veriler = pd.read_csv('Churn_Modelling.csv')
#pd.read_csv("veriler.csv")

#veri on isleme
X= veriler.iloc[:,3:13].values
Y = veriler.iloc[:,13].values

#encoder:  Kategorik -> Numeric
from sklearn.preprocessing import LabelEncoder

labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [1])],   # The column numbers to be transformed (here is [1] but can be [0, 1, 3])
    remainder='passthrough'                                         # Leave the rest of the columns untouched
)
X = ct.fit_transform(X)
X = X[:, 1:] 

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=0)

from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred,y_test)

print(cm)
















