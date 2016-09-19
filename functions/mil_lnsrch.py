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

from log_DD import log_DD

def mil_lnsrch(*args):

    #Parameters
    args = list(args)
    xold = args[0] 
    n = args[1]
    fold = args[2]
    g = args[3]
    p = args[4]
    tolx = args[5]
    stpmax = args[6]
    bags = args[7]
    baglabs = args[8]

    
    ALF = 0.0001
    check = 0
    sum = np.sqrt(np.dot(p,p.T))
    if (sum > stpmax):
        p = p*(stpmax / sum)
    
    slope = np.dot(g, p.T)
    if (slope >= 0):
        xnew = xold
        fnew = fold 
        check = 1
        return xnew,fnew,check
    
    test=np.max(abs(p) / np.maximum(abs(xold),1))
    
    if (test == 0):
        xnew = xold
        fnew = fold
        check = 1
        return xnew,fnew,check
    
    alamin = tolx / test
    alam = 1
    
    while (1):    
        xnew = xold + alam * p
        fnew,_ = log_DD(xnew,bags,baglabs)
        if (alam < alamin):         #Convergence on delta x
            xnew = xold
            check = 1
            return xnew,fnew,check
        else:
            if (fnew <= fold + ALF * alam * slope):   #Sufficient function decrease
                return xnew,fnew,check
            else:
                if (alam == 1):
                    tmplam= -slope / (2 * (fnew - fold - slope))
                else:
                    rhs1=fnew - fold - alam * slope             #Subsequent backtracks
                    rhs2=f2 - fold - alam2 * slope
                    if (alam == alam2):
                        check=1
                        xnew=copy(xold)
                        fnew=copy(fold)
                        return xnew,fnew,check
                    a=((rhs1 / (alam * alam)) - (rhs2 / (alam2 * alam2))) / (alam - alam2)
                    b=(((- alam2 * rhs1) / (alam * alam)) + ((alam * rhs2) / (alam2 * alam2))) / (alam - alam2)
                    if (a == 0):
                        tmplam = -slope / (2 * b)
                    else:                                   
                        disc = b * b - 3 * a * slope
                        if (disc < 0):
                            tmplam = 0.5 * alam
                        else:
                            if (b <= 0):
                                tmplam = (- b + np.sqrt(disc)) / (3 * a)
                            else:
                                tmplam = -slope / (b + np.sqrt(disc))
                    if (tmplam > 0.5 * alam):
                        tmplam = 0.5 * alam
                alam2 = alam
                f2 = fnew
                alam = max(tmplam,0.1 * alam)