# -*- coding: utf-8 -*-
"""
Copyright: A.R. Jose, jmarrietar@unal.edu.co
Departamento de Ciencias de la Computación y de la Decisión
Universidad Nacional de Colombia - Sede Medellín
"""
import numpy as np
import numpy.matlib
from scipy.sparse import identity
from scipy import sparse
from numpy import inf
from scipy.optimize import fmin_bfgs   
import numpy as np
from MILpy.functions.bagprob import bagprob

def log_DD(*args):
    #Parameters
    args = list(args)
    pars = args[0] 
    bags = args[1]
    bagslabs = args[2]

    dim = len(pars)
    n = len(bagslabs)
    _,d2=bags[0].shape
    concept = pars[0:d2]
    end=len(pars)
    s = pars[d2:end]
    prob = np.zeros(n)
    der = np.zeros(shape=(n,dim))

    for i in range (0,n): 
        prob[i], der[i] = bagprob(bags[i],bagslabs[i],concept,s)
    
    prob = np.maximum(prob,1e-12)
    der = -(1/prob).dot(der)
    p = -np.sum(np.log(prob))
    return p,der