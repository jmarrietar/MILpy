# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 23:17:10 2016

@author: josemiguelarrieta
"""

#Import Libraries
import sys,os
os.chdir('/Users/josemiguelarrieta/Documents/MILpy')
sys.path.append(os.path.realpath('..'))
import numpy as np
from sklearn import cross_validation
from sklearn import metrics
from data import load_data
from MILpy.functions.mil_cross_val import mil_cross_val

#Import Algorithms 
from MILpy.Algorithms.simpleMIL import simpleMIL
from MILpy.Algorithms.MILBoost import MILBoost
from MILpy.Algorithms.maxDD import maxDD
from MILpy.Algorithms.CKNN import CKNN
from MILpy.Algorithms.EMDD import EMDD
from MILpy.Algorithms.MILES import MILES
from MILpy.Algorithms.BOW import BOW

#Load Data 
bags,labels,X = load_data('musk1_scaled')  #Musk1 Escalado
#bags,labels,X = load_data('musk1_original')  #Musk1 Original
#bags,labels,X = load_data('data_gauss')  #Gaussian data
#bags,labels,X = load_data('fox_original')  #Musk1 Original


## DRAFT 
bow_classifier = BOW() 

cknn_classifier = CKNN()   #Aqui tienes un problema con los Que resiven parametros

maxdd_classifier = maxDD() 

emdd_classifier = EMDD() 

#En este me funciono maxDD porque no tiene problem con parametros 
mil_cross_val(bags=bags,labels=labels, model=emdd_classifier, folds=10)

parameters = 

tel = {'references': 3, 'citers': 5}

tel = {'type': 'max'}

parameters = {'k':100,'covar_type':'diag','n_iter':20}

##
seed = 66
#seed = 70
#Split Data
#seed= 90
train_bags, test_bags, train_labels, test_labels = cross_validation.train_test_split(bags, labels, test_size=0.1, random_state=seed)


                            ################
                            #Bags Of Words #
                            ################
bow_classifier = BOW() 
bow_classifier.fit(train_bags, train_labels,k=100,covar_type = 'diag',n_iter = 20)
predictions = bow_classifier.predict(test_bags)
accuracie = np.average(test_labels.T == np.sign(predictions))
print '\n Accuracy: %.2f%%' % (100 * accuracie)
fpr, tpr, thresholds = metrics.roc_curve(test_labels, predictions, pos_label=1.)
metrics.auc(fpr, tpr)




#scores = cross_validation.cross_val_score(bow, bags, labels, cv=5,scoring='accuracy')



                            #####################
                            #simpleMIL [average]#
                            #####################
SMILa = simpleMIL() 
SMILa.fit(train_bags, train_labels, type='average')
predictions = SMILa.predict(test_bags)
accuracie = np.average(test_labels.T == np.sign(predictions))
print '\n Accuracy: %.2f%%' % (100 * accuracie)
fpr, tpr, thresholds = metrics.roc_curve(test_labels, predictions, pos_label=1.)
metrics.auc(fpr, tpr)

                            #####################
                            #simpleMIL [extreme]#
                            #####################
SMILe = simpleMIL() 
SMILe.fit(train_bags, train_labels, type='extreme')
predictions = SMILe .predict(test_bags)
accuracie = np.average(test_labels.T == np.sign(predictions))
print '\n Accuracy: %.2f%%' % (100 * accuracie)
fpr, tpr, thresholds = metrics.roc_curve(test_labels, predictions, pos_label=1.)
metrics.auc(fpr, tpr)

                            #################
                            #simpleMIL [max]#
                            #################
SMILmx = simpleMIL() 
SMILmx.fit(train_bags, train_labels, type='max')
predictions = SMILmx.predict(test_bags)
accuracie = np.average(test_labels.T == np.sign(predictions))
print '\n Accuracy: %.2f%%' % (100 * accuracie)
fpr, tpr, thresholds = metrics.roc_curve(test_labels, predictions, pos_label=1.)
metrics.auc(fpr, tpr)

                            #################
                            #simpleMIL [min]#
                            #################
SMILmn = simpleMIL() 
SMILmn.fit(train_bags, train_labels, type='min')
predictions = SMILmn.predict(test_bags)
accuracie = np.average(test_labels.T == np.sign(predictions))
print '\n Accuracy: %.2f%%' % (100 * accuracie)
fpr, tpr, thresholds = metrics.roc_curve(test_labels, predictions, pos_label=1.)
metrics.auc(fpr, tpr)



                            #####
                            #CNN#
                            #####

tel = {'references': 3, 'citers': 5}
cknn_classifier = CKNN() 
#cknn_classifier.fit(train_bags, train_labels,**tel)
cknn_classifier.fit(train_bags, train_labels,references = 3, citers = 5)
predictions = cknn_classifier.predict(test_bags)
accuracie = np.average(test_labels.T == np.sign(predictions))
print '\n Accuracy: %.2f%%' % (100 * accuracie)
fpr, tpr, thresholds = metrics.roc_curve(test_labels, predictions, pos_label=1.)
metrics.auc(fpr, tpr)
    
                            #######
                            #MAXDD#
                            #######
maxdd_classifier = maxDD() 
maxdd_classifier.fit(train_bags=train_bags, train_labels=train_labels)  #Train Classifier
out,predictions = maxdd_classifier.predict(test_bags)
accuracie = np.average(test_labels.T == np.sign(predictions))
print '\n Accuracy: %.2f%%' % (100 * accuracie)
fpr, tpr, thresholds = metrics.roc_curve(test_labels, out, pos_label=1.)
metrics.auc(fpr, tpr)

                            ######
                            #EMDD#
                            ######
emdd_classifier = EMDD() 
emdd_classifier.fit(train_bags=train_bags, train_labels=train_labels)  #Train Classifier
out,predictions = emdd_classifier.predict(test_bags)
#Metrics
accuracie = np.average(test_labels.T == np.sign(predictions))
print '\n Accuracy: %.2f%%' % (100 * accuracie)
fpr, tpr, thresholds = metrics.roc_curve(test_labels, 1-out, pos_label=1.)
metrics.auc(fpr, tpr)



                            ##########   
                            #MILBoost#
                            ##########
#Nota Importante:  Solo Funciona Con musk1 original. 
#Load Data 
bags,labels,X = load_data('musk1_original')  #Musk1 Original
seed = 90
train_bags, test_bags, train_labels, test_labels = cross_validation.train_test_split(bags, labels, test_size=0.1, random_state=seed)

milboost_classifier = MILBoost() 
milboost_classifier.fit(train_bags, train_labels)
out = milboost_classifier.predict(test_bags)
fpr, tpr, thresholds = metrics.roc_curve(test_labels, out, pos_label=1.)
metrics.auc(fpr, tpr)


                            #######
                            #MILES#
                            #######
bags,labels,X = load_data('data_gauss')  #Gaussian data
seed = 66
train_bags, test_bags, train_labels, test_labels = cross_validation.train_test_split(bags, labels, test_size=0.1, random_state=seed)

#Miles, sobre el entrenaod y probado en el training bueno,
#Me hace pensar que depende mucho de datos de entrenamiento
miles_classifier = MILES() 
#MILES.fit(train_bags=bags, train_labels=labels,ktype = 'p',P = 1)  #Train Classifier
miles_classifier.fit(train_bags=train_bags, train_labels=train_labels,ktype = 'p',P = 1)  #Train Classifier
out = miles_classifier.predict(test_bags)
#Metrics
#accuracie = np.average(test_labels.T == np.sign(predictions))
print '\n Accuracy: %.2f%%' % (100 * accuracie)
fpr, tpr, thresholds = metrics.roc_curve(labels, out, pos_label=1.)
metrics.auc(fpr, tpr)

    

      