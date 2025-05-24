# -*- coding: utf-8 -*-
"""

Hold out evaluation is performed in this example with differents MIL Algorithms. 

@author: josemiguelarrieta
"""

#Import Libraries
import sys,os
sys.path.append(os.path.realpath('..'))
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from data import load_data
from sklearn.utils import shuffle
import random as rand

#Import Algorithms 
from MILpy.Algorithms.simpleMIL import simpleMIL
from MILpy.Algorithms.MILBoost import MILBoost
from MILpy.Algorithms.maxDD import maxDD
from MILpy.Algorithms.CKNN import CKNN
from MILpy.Algorithms.EMDD import EMDD
from MILpy.Algorithms.MILES import MILES
from MILpy.Algorithms.BOW import BOW

"""
Note: There is an Issue with regars musk1_original and EMDD and maxDD.
"""

#Load Data 
#bags,labels,_ = load_data('musk1_scaled')  #Musk1 Escalado
#bags,labels,_ = load_data('musk1_original')  #Musk1 Original  ALGO PASA CON ESTA EN EMDD Y MAXDD
bags,labels,_ = load_data('data_gauss')  #Gaussian data
#bags,labels,_ = load_data('fox_original')  #Fox Original
#bags,labels,_ = load_data('fox_scaled')    #Fox Escalado

seed = 66
#seed = 70
#Split Data
#seed= 90

#Shuffle Data
bags,labels = shuffle(bags, labels, random_state=rand.randint(0, 100))

train_bags, test_bags, train_labels, test_labels = train_test_split(bags, labels, test_size=0.1, random_state=seed)

                            ################
                            #Bags Of Words #
                            ################
bow_classifier = BOW() 
bow_classifier.fit(train_bags, train_labels,k=10,covar_type = 'diag',n_iter = 20)
predictions = bow_classifier.predict(test_bags)
accuracie = np.average(test_labels.T == np.sign(predictions))
print('\n Accuracy: %.2f%%' % (100 * accuracie))
fpr, tpr, thresholds = metrics.roc_curve(test_labels, predictions, pos_label=1.)
print('\n Accuracy: %.2f%%' % (100 * accuracie))
SMILmx.fit(train_bags, train_labels, type='max')
predictions = SMILmx.predict(test_bags)
accuracie = np.average(test_labels.T == np.sign(predictions))
print('\n Accuracy: %.2f%%' % (100 * accuracie))
fpr, tpr, thresholds = metrics.roc_curve(test_labels, predictions, pos_label=1.)
print('\n Accuracy: %.2f%%' % (100 * accuracie))
accuracie = np.average(test_labels.T == np.sign(predictions))
print('\n Accuracy: %.2f%%' % (100 * accuracie))
print('\n Accuracy: %.2f%%' % (100 * accuracie))
#Metrics
accuracie = np.average(test_labels.T == np.sign(predictions))
print('\n Accuracy: %.2f%%' % (100 * accuracie))
fpr, tpr, thresholds = metrics.roc_curve(test_labels, 1-out, pos_label=1.)
metrics.auc(fpr, tpr)
train_bags, test_bags, train_labels, test_labels = train_test_split(bags, labels, test_size=0.1, random_state=seed)

milboost_classifier = MILBoost() 
milboost_classifier.fit(train_bags, train_labels)
out = milboost_classifier.predict(test_bags)
fpr, tpr, thresholds = metrics.roc_curve(test_labels, out, pos_label=1.)
metrics.auc(fpr, tpr)

                            #######
                            #MILES#
                            #######
print('\n Accuracy: %.2f%%' % (100 * accuracie))