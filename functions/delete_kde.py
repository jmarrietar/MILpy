# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 21:55:58 2016

@author: josemiguelarrieta
"""
#Import Libraries
import scipy.spatial.distance as dist
from sklearn.neighbors.kde import KernelDensity
from sklearn import svm, cross_validation, metrics
import numpy as np
import sys,os
os.chdir('/Users/josemiguelarrieta/Documents/MILpy')
sys.path.append(os.path.realpath('..'))
from sklearn.metrics import hinge_loss
from data import load_data
from sklearn.utils import shuffle
import random as rand
from copy import deepcopy

###########
#FUNCTIONS#
###########

def feature_update(x,v,j,Z,phi_prototypes,prototypes,labels,bag_index):
    lamb = 1                                          #NOTA: Este Lambda de donde?. 
    Z_copy= deepcopy(Z)
    #phi_prototypes_prim = deepcopy(phi_prototypes)
    #phi_prototypes_prim[bag_index] = j
    #prototypes_prim[bag_index] = deepcopy(train_bags[bag_index][j])
    for i_index_bag in range (0,len(train_bags)):     #Para Cada Bolsa.
        Z_copy[i_index_bag,j] = np.exp(-lamb * _min_hau_bag(train_bags[i_index_bag],[x]))
        pred_decision = lin_svc.decision_function(Z_copy)
        v_prim = hinge_loss(labels, pred_decision)
        if v_prim > v: 
            v_prim = np.inf
            break
    return v_prim, Z_copy, phi_prototypes_prim

def get_prototype(bag,kde,bag_type):
    if bag_type == 'positive': 
        index_prototype = np.argmax(kde.score_samples(bag))
        prototype =  bag[index_prototype]
    elif bag_type == 'negative':
        index_prototype = np.argmin(kde.score_samples(bag))
        prototype =  bag[index_prototype]
    return prototype.reshape(1,len(prototype)), index_prototype

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

################################################################################################

# Nota Importante antes de Usar:
# Estos SVMs, el label 'debe ser', (1,-1)

###########
#LOAD DATA#
###########
#bags,labels,_ = load_data('data_gauss')
bags,labels,_ = load_data('musk1_scaled')  
labels = 2.0*labels-1

#Shuffle Data
bags,labels = shuffle(bags, labels, random_state=rand.randint(0, 100))
#bags,labels = shuffle(bags, labels, random_state=66)
labels = labels.reshape(len(labels))

seed = 66

train_bags, test_bags, train_labels, test_labels = cross_validation.train_test_split(bags, labels, test_size=0.1, random_state=seed)

     

###################
#BORRADOR DE MILIS#
###################

#get negative instances
negative_bags_index = np.where(train_labels == -1)[0]
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
phi_prototypes =[]
for i in range (len(train_bags)):
    if train_labels[i]==1:
        prototype, index_prototype = get_prototype(train_bags[i],kde,'positive')
        prototypes.append(prototype)
        phi_prototypes.append(index_prototype)
    else: 
        prototype, index_prototype = get_prototype(train_bags[i],kde,'negative')
        prototypes.append(prototype)
        phi_prototypes.append(index_prototype)
        
prototypes = np.vstack(prototypes)
phi_prototypes = np.vstack(phi_prototypes).reshape(len(phi_prototypes))

lamb = 1 # NOTA. ESTE LAMBDA DE DONDE!!

Z = np.zeros([len(train_bags),len(prototypes)])
for i in range(len(train_bags)):
    for j in range(len(prototypes)):
        Z[i,j] = np.exp(-lamb * _min_hau_bag(train_bags[i],[prototypes[j]]))
        
"""      
################################################################################
# PARENTESIS 
###############################################################################
lin_svc = svm.LinearSVC().fit(Z, train_labels.reshape(len(train_labels)))
lin_svc.coef_
predicted = lin_svc.predict(Z)
pred_decision = lin_svc.decision_function(Z)
hinge_loss(train_labels, pred_decision)

#0.52298114303397236

accuracie = np.average(train_labels.T == np.sign(predicted))
print '\n Accuracy: %.2f%%' % (100 * accuracie)
fpr, tpr, thresholds = metrics.roc_curve(train_labels, predicted, pos_label=1.)
metrics.auc(fpr, tpr)
#################################################################################
#################################################################################
"""
     
#Training and testing on Trainingd Data
        
#Importante= No le he dado los parametros al LinearSVC.


#Crear Funcion Instupdate Input = Bags,LabelOfBags,Z,phi,w(mi svm)
    #1)Me calcule el loss de mi svm con las Bags
lin_svc = svm.LinearSVC().fit(Z, train_labels.reshape(len(train_labels)))
print 'Coef antes de todos updates'
print lin_svc.coef_
pred_decision = lin_svc.decision_function(Z)
margin = train_labels * pred_decision
losses = 1 - margin
v = np.average(losses)
phi_prototypes_prim = np.zeros(len(train_bags))
print 'Loss antes de update es '+ str(v)
for i in range (0,len(train_bags)):
    if losses[i] > 0:
        for j in range (0,len(train_bags[i])):
            if j != phi_prototypes[i]: 
                x = train_bags[i][j]
                v_prim, Z_prim, phi_prototypes_prim = feature_update(x,v,j,Z,phi_prototypes,prototypes,train_labels,i)
                if v_prim < v:
                    v = v_prim
                    Z = Z_prim
                    prototypes[i] = train_bags[i][j]
                    phi_prototypes = phi_prototypes_prim
                    lin_svc = svm.LinearSVC().fit(Z, train_labels)

print 'Loss despues del update es '+ str(v)

#Actualizar Prototipos prototypes con phi_prototypes

"""
######################################################
lin_svc.coef_
predicted = lin_svc.predict(Z)
pred_decision = lin_svc.decision_function(Z)
hinge_loss(train_labels, pred_decision)


accuracie = np.average(train_labels.T == np.sign(predicted))
print '\n Accuracy: %.2f%%' % (100 * accuracie)
fpr, tpr, thresholds = metrics.roc_curve(train_labels, predicted, pos_label=1.)
metrics.auc(fpr, tpr)
#####################################################
"""

#TESTING

Z_test = np.zeros([len(test_bags),len(prototypes)])
for i in range(len(test_bags)):
    for j in range(len(prototypes)):
        Z_test[i,j] = np.exp(-lamb * _min_hau_bag(test_bags[i],[prototypes[j]]))
        

predicted = lin_svc.predict(Z_test)
pred_decision = lin_svc.decision_function(Z_test)
hinge_loss(test_labels, pred_decision)


accuracie = np.average(test_labels.T == np.sign(predicted))
print '\n Accuracy: %.2f%%' % (100 * accuracie)
fpr, tpr, thresholds = metrics.roc_curve(test_labels, predicted, pos_label=1.)
metrics.auc(fpr, tpr)