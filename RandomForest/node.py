import json
import sys

import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, lNode=None, rNode=None, value=None) -> None:
        self.feature = feature
        self.threshold = threshold
        self.lNode = lNode
        self.rNode = rNode
        self.value = value
        
    def predict(self,x):
        """ Prediction de l'arbre pour une donnee

        :param x: la donnee a predire
        :type x: np.array
        :return: label
        :rtype: str
        """        
        if self.value != None:
            return self.value
        if x[self.feature] <= self.threshold:
            return self.lNode.predict(x)
        else:
            return self.rNode.predict(x)
        
    def get_custom_dict(self):
        """Retourne un dictionnaire (pour la serialisation)

        :return: dictionnaire decrivant le Node
        :rtype: dict
        """        
        if self.value != None:
            custom_dict = {
                "value" : self.value,  
            }
        elif self.feature is None:
            return {}
        else:
            custom_dict = {
                "feature" : self.feature,
                "threshold" : self.threshold, 
            }
            
        if self.lNode:
            custom_dict['lNode'] = self.lNode.get_custom_dict()  
        if self.rNode:
            custom_dict["rNode"] = self.rNode.get_custom_dict()
        
            
        return custom_dict
    
    def serialize(self):
        """Retourne le dictionnaire serialise

        :return: json
        :rtype: dict
        """        
        return self.get_custom_dict()
    
    @staticmethod
    def deserialize(tree_dict):
        """ Deserialise un json pour creer un Node

        :param tree_dict: description du Node
        :type tree_dict: dict
        :return: racine de l'arbre
        :rtype: Node
        """    
        if len(tree_dict) == 0:
            return Node()
        
        new_tree = Node()
        for f in tree_dict:
            if f == "lNode":
                new_tree.lNode = Node.deserialize(tree_dict[f])
            elif f == "rNode":
                new_tree.rNode = Node.deserialize(tree_dict[f])
            else:
                new_tree.__dict__[f] = tree_dict[f]
                
        return new_tree
    
    def get_current_node_data(self, dataset, labels):
        
        # Si c'est une feuille retourner null
        if not self.value is None:
            return None, None
        
        # Si c'est un noeud non initialise correctement, garder le dataset
        if self.feature is None:
            return dataset, labels
        
        
        i_l = np.where(dataset[self.feature].values <= self.threshold)
        i_r = np.where(dataset[self.feature].values > self.threshold)
        ldf = dataset.iloc[i_l]
        rdf = dataset.iloc[i_r]
        
        llables = labels[i_l]
        rlables = labels[i_r]        
        
        # si le noeud gauche est present, l'explorer et retourner ce qu'il retourne si ce n'est pas nul
        ldf, l_new_labels = self.lNode.get_current_node_data(ldf,llables)
        if not ldf is None:
            return ldf,l_new_labels
        
        rdf, r_new_labels = self.rNode.get_current_node_data(rdf,rlables)
        if not rdf is None:
            return rdf, r_new_labels
        
        return None, None
        
    def right_most_leaf(self):
        current_node = self
        while current_node.value is None:
            current_node = current_node.rNode
            
        return current_node.value
    
    def left_most_leaf(self):
        current_node = self
        while current_node.value is None:
            current_node = current_node.rNode
            
        return current_node.value