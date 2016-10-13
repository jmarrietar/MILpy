# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 21:55:58 2016

@author: josemiguelarrieta
"""
#Import Libraries
import scipy.spatial.distance as dist
from sklearn.neighbors.kde import KernelDensity
from sklearn import svm
import numpy as np
import sys,os
os.chdir('/Users/josemiguelarrieta/Documents/MILpy')
sys.path.append(os.path.realpath('..'))
import numpy as np
from data import load_data
from MILpy.functions.dd_kernel import dd_kernel
from MILpy.functions.MIL2SIL import MIL2SIL

###########
#LOAD DATA#
###########
bags,labels,_ = load_data('data_gauss')
train_bags = bags
train_labels = labels

###################
#BORRADOR DE MILIS#
###################
#Hasta la Parte del primer SVM 
ktype = 'p'
P = 1
baglabT = np.asmatrix(train_labels).reshape((-1, 1))
nrbags=len(train_bags)

#get negative instances
negative_bags_index = np.where(train_labels == 0)[0]
positive_bags_index = np.where(train_labels == 1)[0]

negative_bags = []
for i in range (len(negative_bags_index)):
    negative_bags.append(train_bags[negative_bags_index[i]])
    
negative_instances = np.vstack(negative_bags)

#####
#KDE#
#####
#NOTA: ¿Como calcular Bandwidth?
kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(negative_instances)

#Get Prototypes
prototypes = []
for i in range (len(train_bags)):
    if train_labels[i][0]==1:
        prototypes.append(get_prototype(train_bags[i],kde,'positive'))
    else: 
        prototypes.append(get_prototype(train_bags[i],kde,'negative'))
prototypes = np.vstack(prototypes)

lamb = 1 #NOTA. ESTE LAMBDA DE DONDE??

Z = np.zeros([nrbags,len(prototypes)])
for i in range(len(train_bags)):
    for j in range(len(prototypes)):
        Z[i,j] = np.exp(-lamb * _min_hau_bag(train_bags[i],[prototypes[j]]))
        
#Training and testing on Trainingd Data
lin_svc = svm.LinearSVC().fit(Z, train_labels.reshape(len(train_labels)))
predicted = lin_svc.predict(Z)

    
###########
#FUNCTIONS#
###########

def get_prototype(bag,kde,bag_type):
    if bag_type == 'positive': 
        prototype =  bag[np.argmax(kde.score_samples(bag))]
    elif bag_type == 'negative':
        prototype =  bag[np.argmin(kde.score_samples(bag))]
    return prototype.reshape(1,len(prototype))

def _min_hau_bag(X,Y):
    """
    @param  X : a sequence of n bags; each bag is an m-by-k array-like
                      object containing m instances with k features
    @param  Y : a sequence of n bags; each bag is an m-by-k array-like
                      object containing m instances with k features
    @return :  Hausdorff_distance
    """
    
    Hausdorff_distance = max(min((min([list(dist.euclidean(x, y) for y in Y) for x in X]))),
                               min((min([list(dist.euclidean(x, y) for x in X) for y in Y]))))
    return Hausdorff_distance



