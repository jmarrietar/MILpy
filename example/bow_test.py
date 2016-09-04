# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 22:20:40 2016

@author: josemiguelarrieta
"""

import os
os.chdir('/Users/josemiguelarrieta/Documents/MILpy')
from data import load_data
import numpy as np
from sklearn import cross_validation
from sklearn import metrics
from sklearn.mixture import GMM
from sklearn.linear_model import LogisticRegression
from Algorithms import MIL2SIL

#Load Data 

bags,labels,X = load_data('data_gauss')  #Gaussian data

#

#Funcion que tome los labels de las bolsas y los ponga 

#seed = 66
seed = 90

train_bags, test_bags, train_labels, test_labels = cross_validation.train_test_split(bags, labels, test_size=0.1, random_state=seed)


#TRAINING

k = 10 
covar_type = 'diag'  #Poner esto como un Parametro , tambien son posibles  'spherical', 'diag', 'tied', 'full'
n_iter = 20

X,Y = MIL2SIL(train_bags,train_labels)
X,Y = MIL2SIL(train_bags,[0])

gauss_mix_model= GMM(n_components=k,covariance_type=covar_type, init_params='wc', n_iter=n_iter)

gauss_mix_model.fit(X)

out = gauss_mix_model.predict_proba(X)

#Logistic separate positive histograms from negative histograms

logistic = LogisticRegression()
logistic = logistic.fit(out,Y)


#TESTING

n = len(test_bags)

bags_out_test=[]
for i in range (0,n):
    sil_bag,sil_labels= MIL2SIL(test_bags[i],test_labels[i])
    out_test = gauss_mix_model.predict_proba(sil_bag)
    out_test= np.mean(out_test,axis=0)
    bags_out_test.append(out_test.reshape(1,len(out_test)))

bags_out_test = np.vstack(bags_out_test)

out_predicted = logistic.predict(bags_out_test)

#Retornar out_predicted
