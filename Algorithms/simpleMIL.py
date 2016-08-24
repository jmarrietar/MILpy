# -*- coding: utf-8 -*-
"""
Implements Simple Multiple Instance Learning 

@author: josemiguelarrieta
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
        else:
            print 'No exist'
        self._model = svm.SVC()
        self._model.fit(bag_modified, train_labels) 
        
    def predict(self,test_bags):
        """
        @param test_bags : a sequence of n bags; each bag is an m-by-k array-like
                      object containing m instances with k features
        """
        if self._type == 'average':
            bag_mean_test=np.asarray([np.mean(bag, axis=0) for bag in test_bags])
            bag_modified_test = bag_mean_test
        else:
            print 'No exist'
        predictions = self._model.predict(bag_modified_test)
        return predictions