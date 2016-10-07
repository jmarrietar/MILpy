# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 20:20:43 2016

k Fold Stratified Evaluation is done in this example. 

@author: josemiguelarrieta
"""

#Import Libraries
import sys,os
os.chdir('/Users/josemiguelarrieta/Documents/MILpy')
sys.path.append(os.path.realpath('..'))
from sklearn.utils import shuffle
import random as rand
from data import load_data
from MILpy.functions.mil_cross_val import mil_cross_val

#Import Algorithms 
from MILpy.Algorithms.simpleMIL import simpleMIL
from MILpy.Algorithms.MILBoost import MILBoost
from MILpy.Algorithms.maxDD import maxDD
from MILpy.Algorithms.CKNN import CKNN
from MILpy.Algorithms.EMDD import EMDD
from MILpy.Algorithms.BOW import BOW
from MILpy.Algorithms.MILES import MILES

"""
Note: There is an Issue with regars musk1_original and EMDD and maxDD.
"""

#Load Data 
#bags,labels,_ = load_data('musk1_scaled')
#bags,labels,_ = load_data('musk2_scaled') 
#bags,labels,_ = load_data('fox_scaled') 
#bags,labels,_ = load_data('tiger_scaled')  
#bags,labels,_ = load_data('elephant_scaled') 
bags,labels,_ = load_data('data_gauss')


#Shuffle Data
bags,labels = shuffle(bags, labels, random_state=rand.randint(0, 100))

#Number of Folds 
folds=5

bow_classifier = BOW() 
#parameters_bow = {'k':100,'covar_type':'diag','n_iter':20}
parameters_bow = {'k':10,'covar_type':'diag','n_iter':20}
accuracie, results_accuracie, auc,results_auc  = mil_cross_val(bags=bags,labels=labels, model=bow_classifier, folds=folds, parameters=parameters_bow)

SMILa = simpleMIL()
parameters_smil = {'type': 'max'}
#En este me funciono maxDD porque no tiene problem con parametros 
accuracie, results_accuracie, auc,results_auc,elapsed  = mil_cross_val(bags=bags,labels=labels, model=SMILa, folds=folds, parameters=parameters_smil,timer=True)

parameters_smil = {'type': 'min'}
#En este me funciono maxDD porque no tiene problem con parametros 
accuracie, results_accuracie, auc,results_auc  = mil_cross_val(bags=bags,labels=labels, model=SMILa, folds=folds, parameters=parameters_smil)

parameters_smil = {'type': 'extreme'}
#En este me funciono maxDD porque no tiene problem con parametros 
accuracie, results_accuracie, auc,results_auc  = mil_cross_val(bags=bags,labels=labels, model=SMILa, folds=folds, parameters=parameters_smil)

parameters_smil = {'type': 'average'}
#En este me funciono maxDD porque no tiene problem con parametros 
accuracie, results_accuracie, auc,results_auc  = mil_cross_val(bags=bags,labels=labels, model=SMILa, folds=folds, parameters=parameters_smil)

cknn_classifier = CKNN() 
parameters_cknn = {'references': 3, 'citers': 5}
accuracie, results_accuracie, auc,results_auc  = mil_cross_val(bags=bags,labels=labels, model=cknn_classifier, folds=folds, parameters=parameters_cknn)

maxDD_classifier = maxDD()
accuracie, results_accuracie, auc,results_auc  = mil_cross_val(bags=bags,labels=labels, model=maxDD_classifier, folds=folds, parameters={})

emdd_classifier = EMDD()
accuracie, results_accuracie, auc,results_auc  = mil_cross_val(bags=bags,labels=labels, model=emdd_classifier, folds=folds, parameters={})

#STOP: Fix This. 
#MIL BOOST: Tiene Out y no tiene predicted. 

milboost_classifier = MILBoost() 
accuracie, results_accuracie, auc,results_auc  = mil_cross_val(bags=bags,labels=labels, model=milboost_classifier, folds=10,parameters={})


#MIL BOOST: Tiene Out y no tiene predicted. 

#Stop Here: 
#MILES: Tiene Out no tiene predicted. 
