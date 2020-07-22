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
veriler = pd.read_csv('veriler.csv')
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

#liste
l=[1,3,4]