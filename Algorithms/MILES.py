# -*- coding: utf-8 -*-

"""
Implements MILES

Copyright: A.R. Jose, jmarrietar@unal.edu.co
Departamento de Ciencias de la Computación y de la Decisión
Universidad Nacional de Colombia - Sede Medellín
"""

import numpy as np
from scipy.sparse import identity
from scipy.optimize import linprog
from scipy.stats import logistic
from dd_kernel import dd_kernel 
from MIL2SIL import MIL2SIL

class MILES(object):
    
    def _init_(self):
        self._w = None
        self._w0 = None
        self._sva = None
        self._ktype = None
        self._P = None
        self._I = None
        
    def fit(self,train_bags,train_labels,ktype = 'p',P = 1):
        """
        @param train_bags : a sequence of n bags; each bag is an m-by-k array-like
            object containing m instances with k features
        @param train_labels : an array-like object of length n containing -1/+1 labels
        """
        self._ktype = ktype
        self._P = P
        
        
        bagT = train_bags
        baglabT = np.asmatrix(train_labels).reshape((-1, 1))
        nrbags=len(bagT)
        Xtrain,Ytrain = MIL2SIL(train_bags,train_labels)

        ########
        #MAPING#
        ######## 
        nrbags=len(bagT)
        
        m = np.zeros((len(bagT),len(Xtrain)))
        for i in range (0,nrbags):
            d=dd_kernel(bagT[i],Xtrain,ktype,self._P)
            m[i]=np.amax(d, axis=0)
            
        #########
        #linprog#
        #########    
        #% setup the linprog:
        _,nrcon=m.shape #number of potential concepts
        y=2*baglabT-1
        C=1 #Esto es un Parametro que se debe pasar.
        Cweights = np.matlib.repmat(float(C),nrbags,1)
         # reweigh to cope with class imbalance:
        Ipos = [y==+1];
        Ineg = [y==-1];
        Cweights[Ipos]=np.divide(Cweights[Ipos],float(np.count_nonzero(y == 1)))
        Cweights[Ineg]=np.divide(Cweights[Ineg],float(np.count_nonzero(y == -1)))
            
        f=np.squeeze(np.vstack((np.divide(np.ones((2*nrcon,1)), float((2*nrcon))),
                     Cweights,
                        0)))
                        
        A = -np.hstack((np.multiply(np.matlib.repmat(y,1,nrcon),m),
                        -np.multiply(np.matlib.repmat(y,1,nrcon),m),
                        identity(nrbags).toarray(),
                        y))          
        
        b = np.squeeze(-np.ones((nrbags,1))) 
        
        #lb=np.squeeze(np.vstack((np.zeros((2*nrcon+nrbags,1)),-inf)))
        #ub=np.squeeze(np.matlib.repmat(inf,2*nrcon+nrbags+1,1))
             
        res = linprog(f, A_ub=A, b_ub=b, bounds=([0, None]),options=dict(bland=True))
        u = res['x']
        v = u[0:nrcon]-u[nrcon:2*nrcon]
        I = np.where(abs(v)>1e-9)
        if len(I) == 0:
            print 'All weights are zero.'
            I = 1; 
        
        w = v[I]
        self._w = np.array([w])
        self._w0 = u[-1]
        self._sva = Xtrain[I]
        ktype = ktype
        self._I = I
        

    
    def predict(self,test_bags):
        """
        @param test_bags : a sequence of n bags; each bag is an m-by-k array-like
            object containing m instances with k features
        """        
        bagTest = [np.asmatrix(bag) for bag in test_bags]
        Xtest = np.vstack(test_bags)
        
        n = len(bagTest)
        out = np.zeros((n,1))
         
        for i in range (0,n):
            d=dd_kernel(bagTest[i],Xtest,self._ktype,self._P)
            out[i]=np.amax(d[:,self._I], axis=0)*self._w.T + self._w0
        s_out = logistic.cdf(out)

        return s_out