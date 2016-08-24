# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 20:44:27 2016

@author: josemiguelarrieta
"""

import numpy as np
import numpy.matlib
from scipy.sparse import identity
from scipy import sparse
from numpy import inf

def bagprob(*args):
    args = list(args)
    bagi = args[0] 
    labi = args[1]
    concept = args[2]
    concept = np.array([concept]) 
    s = args[3] 
        
    nrpar = 2*np.shape(concept)[1]
    m = np.shape(bagi)[0]
    dff = bagi - np.matlib.repmat(concept,m,1) 
    dff2 = dff**2
    s2 = s**2

    #First the probability:
    p = np.array([np.exp(np.dot(-dff2,s2))])
    p = p.transpose()
    p1minp = np.prod(1-p)

    # then the derivative:
    der = 2*np.matlib.repmat(p,1,nrpar)* np.append(dff*np.matlib.repmat(s2,m,1),-dff2*np.matlib.repmat(s,m,1),axis=1)
    der = np.sum(der/np.matlib.repmat(np.maximum(1-p,[1e-12]),1,nrpar),axis=0)

    if labi > 0:
    	#here is the OR function:
    	bagp = 1-p1minp
    	derp = p1minp*der
    else:
    	bagp = p1minp
    	derp = -p1minp*der
     
    return bagp, derp