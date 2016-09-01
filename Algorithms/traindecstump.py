# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 23:50:13 2016

Copyright: A.R. Jose, jmarrietar@unal.edu.co
Departamento de Ciencias de la Computación y de la Decisión
Universidad Nacional de Colombia - Sede Medellín
"""
import sys 
import numpy as np

def traindecstump(X,w):
    
    #      [H,BESTERR] = TRAINDECSSTUMP(X,W)
    #
    # INPUT
    #   X          Dataset
    #   W          Weight per object
    # DESCRIPTION
    # Train a decision stump on dataset X. Each object in X is weighted by a
    # weight W. Objects from the positive class have a positive weight, and
    # otherwise the weights should be negative.
    #
    # The result is returned in vector H:
    #   H(1)    the feature to threshold
    #   H(2)    the threshold set on that feature
    #   H(3)    the sign (+: right side is positive class, -: neg. side)
    # Also the minimum error is returned in BESTERR.
#

    n,dim = X.shape
    sumneg = w[w< 0].sum()
    sumpos = w[w> 0].sum()
    besterr =  float('Inf')
    bestfeat = 0
    bestthr = 0
    bestsgn = 0

    for i in range  (0,dim-1):
        # find the best threshold for feature i
        # assume that the positive class is on the right of the decision
        # threshold:
        
        sx= np.sort(X[:,i],axis=0)
        J=np.argsort(X[:,i],axis=0)
        z = np.cumsum(w[J])
        
        err1 = -sumneg + z
        minerr=min(err1)
        I=np.argmin(err1)
        
        if (minerr<besterr):
            besterr = minerr
            bestfeat = i
            if (I==n-1):
                bestthr = sx[I]+10*sys.float_info.epsilon
            else:
                bestthr = (sx[I]+sx[I+1])/2 + sys.float_info.epsilon
            bestsgn = 1
        
           #Now assume that the positive class is on the left of the decision
           #threshold:
        err2 =  sumpos - z
        minerr=min(err2)
        I=np.argmin(err2)
        if (minerr<besterr):
            besterr = minerr
            bestfeat = i;
        if (I==n-1):
            bestthr = sx[I]+10*sys.float_info.epsilon
        else:
            bestthr = (sx[I]+sx[I+1])/2 + sys.float_info.epsilon
        bestsgn = -1
    
    return {'bestfeat':bestfeat, 'bestthr':float(bestthr),'bestsgn':bestsgn,'besterr':besterr}
