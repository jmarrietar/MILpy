# -*- coding: utf-8 -*-
"""
Implements Max Diverse Density 

DESCRIPTION
 Maximum diverse density Multi-instance learner. This implementation is
 completely inspired by MATLAB  Udelft MIL toolbox by D.M.J. Tax 
 It optimizes the diverse density using gradient descent, starting from
 initial points SPOINTS and initial scales SCALES. Then the
 optimization is run for EPOCHS epochs, and it is stopped when the
 likelihood changes less than TOL.

Copyright: A.R. Jose, jmarrietar@unal.edu.co
Departamento de Ciencias de la Computación y de la Decisión
Universidad Nacional de Colombia - Sede Medellín
"""
from MILpy.functions.bagprob import bagprob
from MILpy.functions.maxdd2 import maxdd
import numpy as np
from numpy import inf
from sklearn.linear_model import LogisticRegression

class maxDD(object):
    
    
    def __init__(self):
        self._spoints = None
        self._epochs = None
        self._frac = None
        self._tol = None
        self._maxConcept = None
        self._end = None
        self._model = None
        
        
    def fit(self,train_bags,train_labels,spoints = 10,epochs = np.array([4,4]),frac = 1,tol=[1e-5,1e-5,1e-7,1e-7],**kwargs):
        """
        @param train_bags : a sequence of n bags; each bag is an m-by-k array-like
                      object containing m instances with k features
        @param train_labels : an array-like object of length n containing -1/+1 labels
        """        
        self._spoints = spoints
        self._epochs = epochs
        self._frac = frac
        self._tol = tol
        index=np.array(np.where(train_labels == 1))
        index=index.transpose()[0]
        bagI = train_labels  #Labels de las Bolsas
        pbags=[]            #positive Bags
        for i in range(0,len(index)):
            indice=index[i]
            pbags.append(train_bags[indice])
        _,dim = pbags[0].shape

        #Missing condition If Spoints Empty choose all.

        #PENDING: ADD MORE CONDITIONS IF IT FAILS
        
        tmp = np.vstack(pbags)
        I = np.random.permutation(len(tmp))
      
        #Missing Spoints conditionals
        spoints = tmp[I[0:spoints]]

        #Missing scales conditionals
        scales = 0.1*np.ones(dim)
        epochs = epochs*dim

        # begin diverse density maximization
        self._maxConcept,concepts = maxdd(spoints,scales,train_bags,bagI,epochs,tol)
 
        #Invent a threshold...:
        self._end=len(self._maxConcept[0])  
        n = len(train_bags)
        out = np.zeros(n)
     
        for i in range (0,n):
            out[i], _ = bagprob(train_bags[i],1,self._maxConcept[0][0:dim],self._maxConcept[0][dim:self._end])
            
        out=out.reshape(len(out),1)
        train_labels=train_labels.reshape(len(train_labels),1)
        train_labels=np.ravel(train_labels)
        model = LogisticRegression()
        self._model = model.fit(out,train_labels)
      
     
    def predict(self,test_bags):
        """
        @param test_bags : a sequence of n bags; each bag is an m-by-k array-like
                      object containing m instances with k features
        """
        pbagsT=[]
        for i in range(0,len(test_bags)):
            pbagsT.append(test_bags[i])
        _,dimT = pbagsT[0].shape
        nT = len(pbagsT)
        outT = np.zeros(nT)
        for i in range (0,nT):
        	 # check if any objects fall inside the bounds
            outT[i], _ = bagprob(pbagsT[i],1,self._maxConcept[0][0:dimT],self._maxConcept[0][dimT:self._end])
        	
        outT=outT.reshape(len(outT),1)
        predicted = self._model.predict(outT)
        return outT, predicted
        
