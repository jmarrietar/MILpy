# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 23:17:10 2016

@author: josemiguelarrieta
"""

import os
os.chdir('/Users/josemiguelarrieta/Documents/MILpy')
from data import load_data
import numpy as np
from sklearn import cross_validation
from sklearn import metrics
#Import Algorithms 
from Algorithms import simpleMIL
from Algorithms import MILBoost
from Algorithms import maxDD
from Algorithms import CKNN
from Algorithms import EMDD
from Algorithms import MILES
from Algorithms import bow

#Load Data 
bags,labels,X = load_data('musk1_scaled')  #Musk1 Escalado
#bags,labels,X = load_data('musk1_original')  #Musk1 Original
#bags,labels,X = load_data('data_gauss')  #Gaussian data
#bags,labels,X = load_data('fox_original')  #Musk1 Original


seed = 66
#seed = 70
#Split Data
#seed= 90
train_bags, test_bags, train_labels, test_labels = cross_validation.train_test_split(bags, labels, test_size=0.1, random_state=seed)




                            ################
                            #Bags Of Words #
                            ################
bow = bow() 
bow.fit(train_bags, train_labels,k=100,covar_type = 'diag',n_iter = 20)
predictions = bow.predict(test_bags)
accuracie = np.average(test_labels.T == np.sign(predictions))
print '\n Accuracy: %.2f%%' % (100 * accuracie)
fpr, tpr, thresholds = metrics.roc_curve(test_labels, predictions, pos_label=1.)
metrics.auc(fpr, tpr)



bow = bow()
scores = cross_validation.cross_val_score(bow, bags, labels, cv=5,scoring='accuracy')



                            #####################
                            #simpleMIL [average]#
                            #####################
from Algorithms import simpleMIL
simpleMIL = simpleMIL() 
simpleMIL.fit(train_bags, train_labels, type='average')
predictions = simpleMIL.predict(test_bags)
accuracie = np.average(test_labels.T == np.sign(predictions))
print '\n Accuracy: %.2f%%' % (100 * accuracie)
fpr, tpr, thresholds = metrics.roc_curve(test_labels, predictions, pos_label=1.)
metrics.auc(fpr, tpr)

                            #####################
                            #simpleMIL [extreme]#
                            #####################
from Algorithms import simpleMIL
simpleMIL = simpleMIL() 
simpleMIL.fit(train_bags, train_labels, type='extreme')
predictions = simpleMIL.predict(test_bags)
accuracie = np.average(test_labels.T == np.sign(predictions))
print '\n Accuracy: %.2f%%' % (100 * accuracie)
fpr, tpr, thresholds = metrics.roc_curve(test_labels, predictions, pos_label=1.)
metrics.auc(fpr, tpr)

                            #################
                            #simpleMIL [max]#
                            #################
from Algorithms import simpleMIL
simpleMIL = simpleMIL() 
simpleMIL.fit(train_bags, train_labels, type='max')
predictions = simpleMIL.predict(test_bags)
accuracie = np.average(test_labels.T == np.sign(predictions))
print '\n Accuracy: %.2f%%' % (100 * accuracie)
fpr, tpr, thresholds = metrics.roc_curve(test_labels, predictions, pos_label=1.)
metrics.auc(fpr, tpr)

                            #################
                            #simpleMIL [min]#
                            #################
from Algorithms import simpleMIL
simpleMIL = simpleMIL() 
simpleMIL.fit(train_bags, train_labels, type='min')
predictions = simpleMIL.predict(test_bags)
accuracie = np.average(test_labels.T == np.sign(predictions))
print '\n Accuracy: %.2f%%' % (100 * accuracie)
fpr, tpr, thresholds = metrics.roc_curve(test_labels, predictions, pos_label=1.)
metrics.auc(fpr, tpr)



                            #####
                            #CNN#
                            #####
cknn4 = CKNN() 
cknn4.fit(train_bags, train_labels)
predictions = cknn4.predict(test_bags,3,5)
accuracie = np.average(test_labels.T == np.sign(predictions))
print '\n Accuracy: %.2f%%' % (100 * accuracie)
fpr, tpr, thresholds = metrics.roc_curve(test_labels, predictions, pos_label=1.)
metrics.auc(fpr, tpr)
    
                            #######
                            #MAXDD#
                            #######
maxDD = maxDD() 
maxDD.fit(train_bags=train_bags, train_labels=train_labels)  #Train Classifier
out,predictions = maxDD.predict(test_bags)
accuracie = np.average(test_labels.T == np.sign(predictions))
print '\n Accuracy: %.2f%%' % (100 * accuracie)
fpr, tpr, thresholds = metrics.roc_curve(test_labels, out, pos_label=1.)
metrics.auc(fpr, tpr)

                            ######
                            #EMDD#
                            ######
EMDD = EMDD() 
EMDD.fit(train_bags=train_bags, train_labels=train_labels)  #Train Classifier
out,predictions = EMDD.predict(test_bags)
#Metrics
accuracie = np.average(test_labels.T == np.sign(predictions))
print '\n Accuracy: %.2f%%' % (100 * accuracie)
fpr, tpr, thresholds = metrics.roc_curve(test_labels, 1-out, pos_label=1.)
metrics.auc(fpr, tpr)



                            ##########   
                            #MILBoost#
                            ##########
#Nota Importante:  Solo Funciona Con musk1 original. 
from Algorithms import MILBoost
#Load Data 
bags,labels,X = load_data('musk1_original')  #Musk1 Original
seed = 90
train_bags, test_bags, train_labels, test_labels = cross_validation.train_test_split(bags, labels, test_size=0.1, random_state=seed)

MILBoost = MILBoost() 
MILBoost.fit(train_bags, train_labels)
out = MILBoost.predict(test_bags)
fpr, tpr, thresholds = metrics.roc_curve(test_labels, out, pos_label=1.)
metrics.auc(fpr, tpr)


                            #######
                            #MILES#
                            #######
#Miles, sobre el entrenaod y probado en el training bueno,
#Me hace pensar que depende mucho de datos de entrenamiento
MILES = MILES() 
#MILES.fit(train_bags=bags, train_labels=labels,ktype = 'p',P = 1)  #Train Classifier
MILES.fit(train_bags=train_bags, train_labels=train_labels,ktype = 'p',P = 1)  #Train Classifier
out = MILES.predict(test_bags)
#Metrics
#accuracie = np.average(test_labels.T == np.sign(predictions))
print '\n Accuracy: %.2f%%' % (100 * accuracie)
fpr, tpr, thresholds = metrics.roc_curve(labels, out, pos_label=1.)
metrics.auc(fpr, tpr)

    

      