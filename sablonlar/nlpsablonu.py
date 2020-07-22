# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

yorumlar = pd.read_csv('Restaurant_Reviews.csv',delimiter=',', error_bad_lines=False)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

# [:, -1:] son sutun
eklenenDegerler = yorumlar.iloc[:, -1:].values
hesaimputer = imputer.fit(eklenenDegerler[:, -1:]) 
eklenenDegerler[:, -1:] = imputer.transform(eklenenDegerler[:, -1:])

sonuc1 = pd.DataFrame(data=eklenenDegerler, index= range(716),columns = ['Liked'])
print(sonuc1)
review = yorumlar.iloc[:,0:1].values

sonuc2 = pd.DataFrame(data = review, index= range(716),columns = ['Review'])
yorumlar1 = pd.concat([sonuc2, sonuc1],axis=1)

# 1- VERİ ÖNİŞLEME - PREPROCESSING
#Eger resim veya sensörle ilgili veriler işliyor olsaydık 1 ve 2 kısım değişecekti. Geri kısım aynı
#reguler expression
import re 

import nltk
#stopwords lerden herhangi birini görürsen onu kullanma
durma = nltk.download('stopwords')

#kelimdeyi gövdelerine köklerine ayırma
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

nltk.download('stopwords')
from nltk.corpus import stopwords

derlem = []
#csv de 100 yorum satırı var
for i in range(716):
    #kucuk harf ve buyuk harf icermeyemnleri filtrele, bunları boş karakterle değiştir
    #imla işaretlerinden kurtarıyoruz yorumları
    yorum = re.sub('[^a-zA-Z]',' ', yorumlar1['Review'][i])
    #kucuk harfe cevirir
    yorum = yorum.lower()
    #metni listeye donusturur
    yorum = yorum.split()
    #stem yorumdaki her kelimenin gövdesini bul
    #stopwordlerin içinde oluşan kumede sayet kelime yoksa o zaman bu kelimeyi stemle, stemlemenin sonucunu gövdeyi listenin ilk elemanı yap
    #stopwordsler is not this gibialınmıyor
    yorum = [ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words('english'))] #turkish de bulunuyor
    #boldugumuz kelimeleri bosluk karakteri ile gövdeler birlesecek.
    yorum = ' '.join(yorum)
    #derleme yorumu ekliyoruz
    derlem.append(yorum)

print("okuma işlemi tamamlandı!")

# 2- FEATURE EXTRACTION - OZNİTELİK ÇIKARIMI
#Bag Of Words (BOW)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2000) #en fazla kullandıgınız 2000 kelimeyi al (X ekseninde colon), Ram iyi ise daha fazla da olabilir
X  = cv.fit_transform(derlem).toarray() # bağımsız değişken her bir satır 2000 tane öznitelik olarak 0 ve 1 degerlerine dönüstu
y  = yorumlar1.iloc[:,1].values # bağımlı değişken


# 3- MACHINE LEARNING

from sklearn.model_selection import train_test_split
# %80 train %20 test
X_train, X_test, y_train, y_test = train_test_split (X,y,test_size = 0.20, random_state=0)
 
from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(X_train,y_train)
 
y_pred=gnb.predict(X_test)
 
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm) # Accuracy %72.9 
#○2 x2 lik matrisin 11 ve 22 toplamı toplam sutun degerleri toplamına yüzdesi alınarak baişarısı bulunur.































