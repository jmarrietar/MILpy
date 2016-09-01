# -*- coding: utf-8 -*-
"""
Implements Simple Multiple Instance Learning 

Copyright: A.R. Jose, jmarrietar@unal.edu.co
Departamento de Ciencias de la Computación y de la Decisión
Universidad Nacional de Colombia - Sede Medellín
"""


import numpy as np
from sklearn import svm

class simpleMIL(object):
    
    
    def _init_(self):
        self._model = None
        self._type = None

    def fit(self,train_bags,train_labels,type): 
        """
        @param train_bags : a sequence of n bags; each bag is an m-by-k array-like
                      object containing m instances with k features
        @param train_labels : an array-like object of length n containing -1/+1 labels
        """        
        self._type = type
        if self._type == 'average':
            bag_mean = np.asarray([np.mean(bag, axis=0) for bag in train_bags])
            bag_modified = bag_mean
        elif self._type == 'extreme':
            bag_max = np.asarray([np.amax(bag,axis=0) for bag in train_bags])
            bag_min = np.asarray([np.amin(bag,axis=0) for bag in train_bags])
            bag_extreme = np.concatenate((bag_max,bag_min),axis=1)
            bag_modified = bag_extreme
        elif self._type == 'max':
            bag_max = np.asarray([np.amax(bag,axis=0) for bag in train_bags])
            bag_modified = bag_max
        elif self._type == 'min':     
            bag_min = np.asarray([np.amin(bag,axis=0) for bag in train_bags])
            bag_modified = bag_min
        else:
            print 'No exist'
        self._model = svm.SVC()
        self._model.fit(bag_modified, train_labels) 
        
    def predict(self,test_bags):
        """
        @param test_bags : a sequence of n bags; each bag is an m-by-k array-like
                      object containing m instances with k features
        """
        bag_modified_test = None
        
        if self._type == 'average':
            bag_mean_test=np.asarray([np.mean(bag, axis=0) for bag in test_bags])
            bag_modified_test = bag_mean_test
        elif self._type == 'extreme':
            bag_max_test = np.asarray([np.amax(bag,axis=0) for bag in test_bags])
            bag_min_test = np.asarray([np.amin(bag,axis=0) for bag in test_bags])
            bag_extreme_test = np.concatenate((bag_max_test,bag_min_test),axis=1)
            bag_modified_test = bag_extreme_test
        elif self._type == 'max':
            bag_max_test = np.asarray([np.amax(bag,axis=0) for bag in test_bags])
            bag_modified_test = bag_max_test
        elif self._type == 'min':
            bag_min_test = np.asarray([np.amin(bag,axis=0) for bag in test_bags])
            bag_modified_test = bag_min_test
        else:
            print 'No exist'
        predictions = self._model.predict(bag_modified_test)
        return predictions