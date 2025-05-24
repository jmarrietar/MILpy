# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 20:51:51 2016

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
from MILpy.functions.mil_lnsrch import mil_lnsrch
from MILpy.functions.dfpmin import dfpmin
from MILpy.functions.log_DD import log_DD

def maxdd (*args):
    #Parameters
    args = list(args)
    spoints = args[0] 
    scales = args[1]
    bags = args[2]
    bagI = args[3]
    epochs = args[4]
    tol = args[5]

# initialize some parameters and storage

    num_start_points,dim = spoints.shape
    _ , dim = spoints.shape
    concepts = []
    maxConcept = []
    maxConcept.insert(0, np.concatenate((numpy.zeros(shape=(1,dim)) , numpy.ones(shape=(1,dim))),axis = 1))
    maxConcept.insert(1,0) 
    

# make several runs, starting with another startingpoint spoint.

    for i in range(0,num_start_points):
            #Meterle el if aqui del Print
        xold = np.concatenate((spoints[i],scales))
        fold, g = log_DD(xold,bags,bagI)
        p = -g
        sumx = np.dot(xold,xold.T)    #%Upper bound on step size
        stpmax = 100*max(np.sqrt(sumx),2*dim) #upper bound on step size
        #% Now do an iterative line-search to find the global minimum
        for iter in (0 ,epochs[0]):
            xnew,fnew,check = mil_lnsrch(xold,dim,fold,g,p,tol[3],stpmax,bags,bagI)
            xi = xnew-xold
            tst = max(abs(xi)/np.maximum(abs(xnew),1))
            if tst<tol[1]:
                break
            #Store for the next step
            p = -g
            xold = xnew
            fold = fnew
            sumx = np.dot(xold,xold.T)
            stpmax = 100*max(np.sqrt(sumx),2*dim)
    
        iterations=np.zeros(num_start_points)
    
        xnew,fret,iterations[i] = dfpmin(xnew,dim,tol[2],tol[3],epochs[1],bags,bagI)
    
        concepts.append([])
    
        concepts[i].append(xnew)
        concepts[i].append(np.exp(-fret))
        
        if concepts[i][1] > maxConcept[1]:
            maxConcept[0] = concepts[i][0]
            maxConcept[1] = concepts[i][1]
    return maxConcept, concepts
