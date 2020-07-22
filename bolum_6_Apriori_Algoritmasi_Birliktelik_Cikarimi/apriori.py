# Apriori

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#csv doysaından colon başlıkları olmadığı için (boy, yas, gelir gibi) header=non yazılıd
veriler = pd.read_csv('sepet.csv', header = None)

cleaned_list = []
for x in t:
    clist = []
    for z in x:
        if str(z) != 'nan':
            clist.append(z)
    cleaned_list.append(clist)
    
 
 
from apyori import apriori,TransactionManager
rules = apriori(cleaned_list, min_support = 0.01, min_confidance = 0.2, min_lift = 3)
rules_list = list(rules)
 
for item in rules_list:
 
    base_items = [x for x in item[2][0][0]]
    add_item, = item[2][0][1]
    print("Rule: " + " + ".join(base_items) + " -> " + str(add_item))
 
    print("Support: " + str(item[1]))
 
    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("=====================================")


t = []
for i in range (0,7501):  #verilerde 7501 satır var
    t.append([str(veriler.values[i,j]) for j in range(0,20)]) #list of list oluştu, range kaç sutun


from apyori import apriori
kurallar = apriori(t,min_support=0.01, min_confidence=0.2, min_lift = 3, min_length=2)

print(list(kurallar))
