# -*- coding: utf-8 -*-

"""
Implements MILBoost

DESCRIPTION

Copyright: A.R. Jose, jmarrietar@unal.edu.co
Departamento de Ciencias de la Computación y de la Decisión
Universidad Nacional de Colombia - Sede Medellín
"""

import numpy as np
from scipy.optimize import fmin_bfgs
from MILpy.functions.noisyORlossWeights import noisyORlossWeights
from MILpy.functions.noisyORlossAlphas import noisyORlossAlphas
from MILpy.functions.traindecstump import traindecstump

class MILBoost(object):
    
    def __init__(self):
        self._alpha = None
        self._H = None 
        self._T = None
        
    def fit(self,train_bags,train_labels, errtol = 1e-15,T=100,**kwargs):
        """
        @param train_bags : a sequence of n bags; each bag is an m-by-k array-like
        object containing m instances with k features
        @param train_labels : an array-like object of length n containing -1/+1 labels
        """
        self._T = T 
        bagSil = [np.asmatrix(bag) for bag in train_bags]
        baglab = np.asmatrix(train_labels).reshape((-1, 1))
        X = np.vstack(bagSil)
                               
        A = np.array([])  
        Ibag=[]        

        for index in range (0,len(bagSil)):
            A=np.append(A, index*(np.ones(len(bagSil[index]))))
            Ibag.append(np.array(np.where(A==index)))

        bagy =baglab
        N= len(X)

        #init
        BestFeature=[]
        h = np.zeros((T,3))
        H=[]
        self._alpha = np.zeros((T,1))
        prev_out = np.zeros((N,1))
    
        for t in range (0,T):
            w = noisyORlossWeights(prev_out,bagy,Ibag)
            h = traindecstump(X,w)
            BestFeature.append(h['bestfeat'])
            H.append(h)
        
            this_out=np.array(h['bestsgn']*np.sign(X[:,h['bestfeat']]-h['bestthr']))
        
      
            xopt = fmin_bfgs(noisyORlossAlphas,1,args=(prev_out,this_out,bagy,Ibag))
            self._alpha[t]=xopt[0]
            # update output full classifier:
            prev_out = prev_out + self._alpha[t]*this_out;
            besterr=h['besterr']
            if (besterr<=errtol):
                self._H = H 
                break
        self._H = H 

    def predict(self,test_bags):
        """
        @param test_bags : a sequence of n bags; each bag is an m-by-k array-like
            object containing m instances with k features
        """        
        T = self._T 
        H = self._H

        bagSilT = [np.asmatrix(bag) for bag in test_bags]
                               
        AT=np.array([])
        IbagT=[]

        for index in range (0,len(bagSilT)):
            AT=np.append(AT, index*(np.ones(len(bagSilT[index]))))
            IbagT.append(np.array(np.where(AT==index)))

        bagSilT = [np.asmatrix(bag) for bag in test_bags]

        Z = np.vstack(bagSilT)

        pij=[]
        n = len(Z)
        out = np.zeros((n,1))
        for i in range(0,T-1):
             out = out + float(self._alpha[i])*H[i]['bestsgn']*np.sign(Z[:,H[i]['bestfeat']]-H[i]['bestthr'])
             pij = 1/(1+np.exp(-out))

        B = len(test_bags);
        out = np.zeros((B,1))
        for i in range (0,B-1):
            out[i]=1-np.prod([1-np.asarray(pij[IbagT[i]])])
        
        return out