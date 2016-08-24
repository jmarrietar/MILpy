# -*- coding: utf-8 -*-

"""
Implements Expectation Maximization Maximum Diverse Density

DESCRIPTION
 Expectation Maximization Maximum Diverse Density This implementation is
 completely inspired by MATLAB Udelft MIL toolbox by D.M.J. Tax 
 Use the Expectation Maximization version of the Maximum Diverse
 Density. It is an iterative EM algorithm, requiring a sensible
 initialisation. 


@author: josemiguelarrieta
"""
import numpy as np
from numpy import inf
from maxdd2 import maxdd
from log_DD import log_DD
from sklearn.linear_model import LogisticRegression

class EMDD(object):
    
    def _init_(self):
        self._spoints = None
        self._epochs = None
        self._frac = None
        self._tol = None
        self._maxConcept = None
        self._end = None
        self._alf = None
        self._model = None
        
    def fit(self,train_bags,train_labels,alf = 10,spoints = 10,epochs = np.array([4,4]),frac = 1,tol=[1e-5,1e-5,1e-7,1e-7]):
        """
        @param train_bags : a sequence of n bags; each bag is an m-by-k array-like
            object containing m instances with k features
        @param train_labels : an array-like object of length n containing -1/+1 labels
        """
        self._spoints = spoints
        self._epochs = epochs
        self._frac = frac
        self._tol = tol
        self._alf = alf
        bagI = train_labels
        nrobags = len(train_bags)
        #_, dim = X.shape
        _,dim = train_bags[0].shape
        self._epochs = self._epochs*dim

        index = np.array(np.where(train_labels == 1))
        index = index.transpose()

        pbags = []            #positive Bags
        for l in range(0,len(index)):
            indice = index[l][0]
            pbags.append(train_bags[indice])
    
        #Poner la Condicion de if spoints is Empty Coger Todos        
        #FALTA PONER TODOS LOS CONDICIONALES QUE TIENE EN MATLAB
        startpoint = np.vstack(pbags)
        I = np.random.permutation(len(startpoint))

        #PONER AQUI TODAS LAS CONDICIONALES DE SPOINTS 

        if alf<1:
            print 'Hacer Algo' #k = max(round(alf*length(I)),1);
        else:
            k = alf
    
        if k>len(startpoint):
        	k = len(startpoint)
        else:
        	startpoint = startpoint[I[0:k]]

        scales = np.matlib.repmat(0.1,k,dim)
        pointlogp = np.matlib.repmat(inf,k,1)

        #start the optimization k times:
        for i in range (0,k):
            bestinst = []
            logp1,_ = log_DD( np.concatenate((startpoint[i,:],scales[i,:])),train_bags,bagI)
             #do a few runs to optimize the concept and scales in an EM fashion:
            for r in range(0,10):
                    # find the best fitting instance per bag [ Los mas cercanos a ese starting point en cada Bolsa]]
                bestinst = []        
                for j in range (0,nrobags):
                    
                    dff = train_bags[j] - np.matlib.repmat(startpoint[i],len(train_bags[j]),1)
                    dff=np.dot(dff**2,scales[i].reshape(len(scales[i]),1))
                    J = np.argmin(dff)
                    bestinst.append(np.array([train_bags[j][J]]))
                #run the maxDD on only the best instances
                self._maxConcept,concepts = maxdd(np.array([startpoint[i]]),scales[i],bestinst,bagI,self._epochs,tol)    
                end=len(self._maxConcept[0])    
                startpoint[i] = self._maxConcept[0][0:dim]
                scales[i] = self._maxConcept[0][dim:end]
                # do we improve?
                logp0 = logp1
                logp1,_ = log_DD(self._maxConcept[0],train_bags,bagI)
                if (abs(np.exp(-logp1)-np.exp(-logp0))<0.01*np.exp(-logp0)):
                    break
            pointlogp[i] = logp1
        # now we did it k times, what is the best one?
        J = np.argmin(pointlogp)
        self._maxConcept=np.concatenate((startpoint[J],scales[J]))
        # invent a threshold...:
        out = np.zeros(nrobags)
        for i in range (0,nrobags):    
            out[i],_ = log_DD(self._maxConcept,[train_bags[i]],[1])
        model = LogisticRegression()
        out=out.reshape(len(out),1)
        self._model = model.fit(out,train_labels)

    
    
    def predict(self,test_bags):
        """
        @param test_bags : a sequence of n bags; each bag is an m-by-k array-like
            object containing m instances with k features
        """        
        bagsT = []
        for i in range(0,len(test_bags)):
            bagsT.append(test_bags[i])
        _,dimT = bagsT[0].shape
        nT = len(bagsT)
        outT = np.zeros(nT)
        for i in range (0,nT):    
            outT[i],_ = log_DD(self._maxConcept,[test_bags[i]],[1])
        outT = outT.reshape(len(outT),1)
        predicted = self._model.predict(outT)
        return outT, predicted
    
    
    
    
    
    