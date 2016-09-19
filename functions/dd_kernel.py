# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 21:09:38 2016

@author: josemiguelarrieta
"""
    
def dd_kernel(*args):
    # INPUT
#   A,B      Data matrices
#   KTYPE    Kernel type
#   P        Kernel parameter
#
# OUPUT
#   K        Matrix with kernel values

    #Parameters
    args = list(args)
    A = args[0] 
    B = args[1] 
    KTYPE = args[2] 
    P = args[3] 

    
    if KTYPE=='p' and P==1: 
        K=A*B.transpose()
    else: 
        return 1
    return K

