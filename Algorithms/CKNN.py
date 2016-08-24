"""
Implements Citation-KNN
"""
import numpy as np
import scipy.spatial.distance as dist
import inspect


class CKNN(object):
    """
    Citation-KNN
    """

    def __init__(self):
        
        self._bags = None
        self._bag_predictions = None
        self._labels=None
        self._full_bags=None
        self._DM=None

    def fit(self, bags, y):
        """
        @param bags : a sequence of n bags; each bag is an m-by-k array-like
                      object containing m instances with k features
        @param y : an array-like object of length n containing -1/+1 labels
        """
        self._bags = bags
        self._labels=y

    def predict(self, Testbags,R,C):
        """
        @param bags : a sequence of n bags; each bag is an m-by-k array-like
                      object containing m instances with k features
        @return : an array of length n containing real-valued label predictions
        """
        #Unir Bolsas de Training and Testing 
        train_bags=self._bags
        full_bags=self._bags+Testbags    
        pred_labels=np.array([])
        self._DM=self.DistanceMatrix(full_bags)
        
        for num in range(len(self._bags),len(full_bags) ):
            number=num
            REFERENCES= self._DM[number,0:R]
            CiteMatrix=self._DM[:,0:C]
            CITERS,j=np.where(CiteMatrix == number)
            
            LabelsTrainCiters=self._labels[CITERS[CITERS<len(train_bags)]]
            LabelsTrainRef=self._labels[REFERENCES[REFERENCES<len(train_bags)]]
            
            Rp= np.count_nonzero(LabelsTrainRef == 1) 
            Rn=np.count_nonzero(LabelsTrainRef == 0)      
            Cp= np.count_nonzero(LabelsTrainCiters == 1)    
            Cn=np.count_nonzero(LabelsTrainCiters == 0)
            
            if Rp+Cp> Rn+Cn:
                label_out = 1
            else:
                label_out = 0
            pred_labels=np.append(pred_labels,label_out)       
        return pred_labels
        
        #Distancias de las Bolsas 
        #Se hallan las distancias de las Bolsas a todas las demas. 

    def DistanceMatrix (self,bags): 
        BagDistances ={}
        count=0        
        
        #Bucle para recorrer todas las Bolsas
        for bag in bags:
            #Hallar la distancia Hausdorr de Todas las bolsas con todas
                for i in range(0, len(bags)):
                    BagDistances[i] = _min_hau_bag(bags[i],bag)
                references_bag_={}
                references_bag=sorted(BagDistances.items(), key=lambda x: x[1]) #Ordeno las bolsas referentes de la Bolsa seleccionada
                REF_Bag_p=[]
                for j in range(0, len(references_bag)):
                    REF_Bag_p.append(references_bag[j][0])
            
                if count==0:
                    DistanceMatrix=np.matrix(REF_Bag_p)
                else:  
                    DistanceMatrix = np.vstack([DistanceMatrix, REF_Bag_p])
                count=count+1
        return DistanceMatrix       


def _hau_bag(X,Y):
    """
    @param  X : a sequence of n bags; each bag is an m-by-k array-like
                      object containing m instances with k features
    @param  Y : a sequence of n bags; each bag is an m-by-k array-like
                      object containing m instances with k features
    @return :  Hausdorff_distance
    """
    
    Hausdorff_distance=max(max((min([list(dist.euclidean(x, y) for y in Y) for x in X]))),
                               max((min([list(dist.euclidean(x, y) for x in X) for y in Y]))))
    return Hausdorff_distance


def _min_hau_bag(X,Y):
    """
    @param  X : a sequence of n bags; each bag is an m-by-k array-like
                      object containing m instances with k features
    @param  Y : a sequence of n bags; each bag is an m-by-k array-like
                      object containing m instances with k features
    @return :  Hausdorff_distance
    """
    
    Hausdorff_distance=max(min((min([list(dist.euclidean(x, y) for y in Y) for x in X]))),
                               min((min([list(dist.euclidean(x, y) for x in X) for y in Y]))))
    return Hausdorff_distance



