#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 19:03:45 2018

@author: sadievrenseker
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv('Ads_CTR_Optimisation.csv')

import random

N = 10000 #csv dosyası 10000 satır ilan var
d = 10  #10 ilan var, 10 sutun
toplam = 0
secilenler = []
for n in range(0,N):
    ad = random.randrange(d) #10 a kadarolan bir sayı uret bu aslında tıklanan ilan olacak
    secilenler.append(ad)
    # verilerdeki n. satır = 1 ise odul 1
    # random sayı tıklanan reklam ise (veriteinde tıklanan ilanlar 1 diğerleri 0) odul de 1 olacak
    odul = veriler.values[n,ad] 
    toplam = toplam + odul
    
    
plt.hist(secilenler)
plt.show()










