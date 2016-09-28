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
import numpy as np
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

"""
Note: There is an Issue with regars musk1_original and EMDD and maxDD.
"""

#Load Data 
bags,labels,X = load_data('musk1_scaled')

bow_classifier = BOW() 
parameters_bow = {'k':100,'covar_type':'diag','n_iter':20}
mil_cross_val(bags=bags,labels=labels, model=bow_classifier, folds=10,parameters=parameters_bow)

SMILa = simpleMIL()
parameters_smil = {'type': 'max'}
#En este me funciono maxDD porque no tiene problem con parametros 
mil_cross_val(bags=bags,labels=labels, model=SMILa, folds=10,parameters=parameters_smil)

parameters_smil = {'type': 'min'}
#En este me funciono maxDD porque no tiene problem con parametros 
mil_cross_val(bags=bags,labels=labels, model=SMILa, folds=10,parameters=parameters_smil)

parameters_smil = {'type': 'extreme'}
#En este me funciono maxDD porque no tiene problem con parametros 
mil_cross_val(bags=bags,labels=labels, model=SMILa, folds=10,parameters=parameters_smil)

parameters_smil = {'type': 'average'}
#En este me funciono maxDD porque no tiene problem con parametros 
mil_cross_val(bags=bags,labels=labels, model=SMILa, folds=10,parameters=parameters_smil)

cknn_classifier = CKNN() 
parameters_cknn = {'references': 3, 'citers': 5}
mil_cross_val(bags=bags,labels=labels, model=cknn_classifier, folds=10,parameters=parameters_cknn)

maxDD_classifier = maxDD()
mil_cross_val(bags=bags,labels=labels, model=maxDD_classifier, folds=10,parameters={})

emdd_classifier = EMDD()
mil_cross_val(bags=bags,labels=labels, model=emdd_classifier, folds=10,parameters={})

milboost_classifier = MILBoost() 
mil_cross_val(bags=bags,labels=labels, model=milboost_classifier, folds=10,parameters={})

#MIL BOOST: Tiene Out y no tiene predicted. 

#MILES: Tiene Out no tiene predicted. 
