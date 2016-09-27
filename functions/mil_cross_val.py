# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 20:09:20 2016

@author: josemiguelarrieta
"""

from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np

"""
NOTE: AUX is missing Here. 
"""

def mil_cross_val(bags,labels,model,folds):

    skf = StratifiedKFold(labels.reshape(len(labels)), n_folds=folds)
    results_accuracie = []
    results_auc = []
    run = 0
    for train_index, test_index in skf:
        X_train = [bags[i] for i in train_index]
        Y_train = labels[train_index]
        X_test  = [bags[i] for i in test_index]
        Y_test  = labels[test_index]
        model.fit(X_train, Y_train)
        print 'Run # '+str(run)
        _, predictions = model.predict(X_test) #Aqui tuve que hacer algo tricky ya que me devuelve 2 predict y predict_proba? 
        accuracie = np.average(Y_test.T == np.sign(predictions)) 
        results_accuracie.append(100 * accuracie)
        auc_score = roc_auc_score(Y_test,predictions)  
        results_auc.append(100 * auc_score)
        print 'roc_auc_score'
        print auc_score
        print 'predictions' 
        print predictions 
        print 'real'
        run = run+1
        
    #modify below to Return AUC
    return np.mean(results_accuracie), results_accuracie, results_auc, np.mean(results_auc)



    
    