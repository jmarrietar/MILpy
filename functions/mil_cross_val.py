# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 20:09:20 2016

MIL K stratified fold representation

@author: josemiguelarrieta
"""

from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np
import sys

def mil_cross_val(bags,labels,model,folds,parameters={}):    
    skf = StratifiedKFold(labels.reshape(len(labels)), n_folds=folds)
    results_accuracie = []
    results_auc = []
    fold = 0
    for train_index, test_index in skf:
        X_train = [bags[i] for i in train_index]
        Y_train = labels[train_index]
        X_test  = [bags[i] for i in test_index]
        Y_test  = labels[test_index]
        sys.stdout.write('Fold# '+str(fold)+'...')
        if len(parameters) > 0: 
            model.fit(X_train, Y_train, **parameters)
        else: 
            model.fit(bags, labels)
        predictions = model.predict(X_test)           
        if (isinstance(predictions, tuple)):
            predictions = predictions[0]
        accuracie = np.average(Y_test.T == np.sign(predictions)) 
        results_accuracie.append(100 * accuracie)
        auc_score = roc_auc_score(Y_test,predictions)  
        results_auc.append(100 * auc_score)
        fold = fold+1
    return np.mean(results_accuracie), results_accuracie, np.mean(results_auc), results_auc