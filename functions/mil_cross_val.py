# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 20:09:20 2016

@author: josemiguelarrieta
"""

from sklearn.cross_validation import StratifiedKFold
from sklearn import metrics
import numpy as np

"""
NOTE: AUX is missing Here. 
"""

def mil_cross_val(bags,labels,model,folds):

    
    skf = StratifiedKFold(labels.reshape(len(labels)), n_folds=folds)

    results_accuracie = []

    run = 0
    for train_index, test_index in skf:
        X_train = [bags[i] for i in train_index]
        Y_train = labels[train_index]
        X_test  = [bags[i] for i in test_index]
        Y_test  = labels[test_index]
        model.fit(X_train, Y_train)
        print 'Run # '+str(run)
        _,predictions = model.predict(X_test) #Aqui tuve que hacer algo tricky ya que me devuelve 2 predict y predict_proba? 
        #out = model.predict_proba(X_test)
        accuracie = np.average(Y_test.T == np.sign(predictions)) 
        print '\n Accuracy: %.2f%%' % (100 * accuracie)
        results_accuracie.append(100 * accuracie)
        #IMPORTANT FIX AUX
        #fpr, tpr, thresholds = metrics.roc_curve(Y_test, out[:,0], pos_label=1) #this is Wrong 
        print 'predictions' 
        print predictions 
        print 'real'
        print Y_test
        #print metrics.auc(fpr, tpr)
        run = run+1
        
    #modify below to Return AUC
    return np.mean(results_accuracie), results_accuracie, 0, 0 



    
    