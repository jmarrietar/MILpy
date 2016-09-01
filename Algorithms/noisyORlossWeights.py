# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 23:42:34 2016

Copyright: A.R. Jose, jmarrietar@unal.edu.co
Departamento de Ciencias de la Computación y de la Decisión
Universidad Nacional de Colombia - Sede Medellín
"""

import numpy as np

def noisyORlossWeights(prev_out,bagy,Ibag):
    B = len(Ibag)
    N = len(prev_out)

    pij = 1/(1+np.exp(-prev_out))
        
    pi = np.zeros((B,1))
    w = np.zeros((N,1))

    #run over the bags
    for i in range (0,B-1):
        pi[i] = 1 - np.prod(1-pij[Ibag[i]])
        if (bagy[i]==1):
            w[Ibag[i]] = (1-pi[i])*pij[Ibag[i]]/pi[i]
        else:
            w[Ibag[i]] = -pij[Ibag[i]]
		
    # I run into problems when the weights are virtually zero, so avoid that
    # it really becomes too small:
    
    tol = 1e-10
    d=np.where(abs(w)<tol)
       
    if (len(d)>0):
         a,b=(np.where(abs(w)<tol))
         #A[a]=np.sign(a)*tol; 
    return w