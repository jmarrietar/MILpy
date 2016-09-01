# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 23:55:49 2016

Copyright: A.R. Jose, jmarrietar@unal.edu.co
Departamento de Ciencias de la Computación y de la Decisión
Universidad Nacional de Colombia - Sede Medellín
"""

import numpy as np 
import sys 

def noisyORlossAlphas(alpha,prev_out,this_out,bagy,Ibag):
    B = len(Ibag)
    
    pij=[]
    pij = 1/(1+np.exp(-prev_out-alpha*this_out))
    pi = np.zeros((B,1))
    logL = 0
    
    # run over the bags
    for i in range (0,B-1):
        
        pi[i] = 1 - np.prod(1-pij[Ibag[i]])

        if (bagy[i]==1):
            logL = logL - np.log(pi[i]+sys.float_info.epsilon)
        else:
            logL = logL - np.log(1-pi[i]+sys.float_info.epsilon)
    return logL