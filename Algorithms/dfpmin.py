# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 20:57:44 2016

@author: josemiguelarrieta
"""

import numpy as np
import numpy.matlib
from scipy.sparse import identity
from scipy import sparse
from numpy import inf
from scipy.optimize import fmin_bfgs   
import numpy as np
from mil_lnsrch import mil_lnsrch
from log_DD import log_DD

def dfpmin(*args):

    #Parameters
    args = list(args)
    xold = args[0] 
    n = args[1] 
    tolx = args[2] 
    gtol = args[3] 
    itmax = args[4] 
    bags = args[5] 
    baglabs = args[6] 

    xnew = xold
    
    fp,g = log_DD(xold,bags,baglabs)
    hessin = np.eye(2 * n)
    
    xi = - g
    sum = np.dot(xold , xold.T)
    stpmax = 100 * np.maximum(np.sqrt(sum),2 * n)
    for its in range(0,itmax):  
        iter = its
        pnew,fret,check = mil_lnsrch(xnew,n,fp,g,xnew,tolx,stpmax,bags,baglabs)
        fp = fret
        xi = pnew - xnew      #Update the line direction
        xnew = pnew           #Update the current pint
        test= np.max(abs(xi) / np.maximum(abs(xnew),1))   #Test for convergence on delta x
        if (test < tolx):
            return xnew,fret,iter
        dg = g                 #Save the old gradient
        dummy,g = log_DD(xnew,bags,baglabs)
        g = np.array([g])
        den = max(fret,1)     #Test for convergence on zero gradient
        test = np.max(np.abs(g)*(np.maximum(abs(xnew),1))) / den
        if (test < gtol):
            return xnew,fret,iter
        dg = g - dg
        hdg = np.dot(hessin, dg.T)
        fac = np.dot(dg , xi.T)
        fae = np.dot(dg , hdg)
        sumdg = np.dot(dg, dg.T)
        sumxi = np.dot(xi, xi.T)
        if (fac > np.sqrt(3e-08 * sumdg * sumxi)):
            fac = 1 / fac
            fad = 1 / fae
            dg = fac * xi - np.dot(fad, hdg.T)
            hessin = hessin + fac * (xi.T * xi) - fad * (hdg * hdg.T) + fae * (g.T * g)
        xi = ( np.dot(- hessin, g.T)).T
        