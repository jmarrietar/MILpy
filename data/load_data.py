# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 10:30:26 2016

@author: josemiguelarrieta
"""

from scipy.io import loadmat
import numpy as np

def load_data(data):
    
    if data == 'musk1_scaled':  
        filename_bag = 'data/musk1_scaled/Bag2_mus_escal.mat'
        filename_labels = 'data/musk1_scaled/bagI_mus_escal.mat'
        X_g = loadmat('data/musk1_scaled/X_mus_escal.mat')
    elif data == 'data_gauss': 
        filename_bag = 'data/gauss_data/bag_g.mat'
        filename_labels = 'data/gauss_data/bagI_gauss.mat'
        X_g = loadmat('data/gauss_data/X_g.mat')
    elif data == 'musk1_original':
        filename_bag = 'data/musk1_unscaled/Bag2_musk_original.mat'
        filename_labels = 'data/musk1_unscaled/bagI_musk1_original.mat'
        X_g = loadmat('data/musk1_unscaled/X_musk_original.mat')
    else:
        file = data
        filename_bag = 'MILpy/data/'+file+'/Bag2.mat'
        filename_labels = 'MILpy/data/'+file+'/bagI.mat'
        X_g = loadmat('MILpy/data/'+file+'/X.mat')
    
        
    bag_g = loadmat(filename_bag)
    labels = loadmat(filename_labels)
    try: 
        Bag = bag_g['Bag2']
    except KeyError: 
        Bag = bag_g['Bag']
    labels = labels['bagI']
    X = X_g['X']
    Bag = np.squeeze(Bag-1)
    nrobags = max(Bag+1)
    bags = []
    for i in range(0,nrobags): 
        index = np.where( Bag == i )
        bag = X[index]
        bags.append(bag)
    return bags,labels,X


    
    
    

