import requests
from flask import jsonify
import json
from multiprocessing import Pool
import numpy as np


class ServerManager():
    def __init__(self, clients) -> None:
        self.clients = clients
        
    def __get(self, data, uri):
        """Effectue un HTTP GET pour chaque client et retourne leurs reponses

        :param data: Json a envoyer au client
        :type data: dict
        :param uri: ressource a acceder
        :type uri: str
        :return: Reponse des clients
        :rtype: list
        """        
        values = []
        for client in self.clients:
            r = requests.get(f'{client}/{uri}',json=data, headers={"Content-Type":"application/json; charset=utf-8"})
            
            values.append(r.json())
            
        return np.array(values)
        
            
    def send_dataset_to_client(self,dataset, labels):
        """Envoie les sous-dataset aux clients

        :param dataset: Liste de plusieurs sous-datasets
        :type dataset: list
        :param labels: Liste de labels
        :type labels: list
        """  
        
        for client in range(len(self.clients)):
            requests.post(f'{self.clients[client]}/dataset', json={'dataset': dataset[client].to_dict(), 'labels': labels[client].to_dict()}, headers={"Content-Type":"application/json; charset=utf-8"})
            
    def get_thresholds(self, features, current_tree):
        
        data = {"features" : features, "current_tree":current_tree.serialize()}
        return self.__get(data,'thresholds')
            
    def get_best_threshold_from_clients(self,features, thresholds, current_tree):
        """Obtient les thresholds optimaux des clients

        :param features: Liste des features a evaluer
        :type features: list
        :param thresholds: Liste des thresholds associes aux features decider par Master
        :type thresholds: ist
        :param current_tree: arbre actuellement construit
        :type current_tree: Node
        :return: Liste des thresholds optimaux des clients
        :rtype: np.array
        """        
        data = {"features" : features.tolist(), "thresholds": thresholds.tolist(), "current_tree":current_tree.serialize()}
        return self.__get(data,'best-threshold')
    
    def get_leafs(self, current_tree):
        data = {"current_tree":current_tree.serialize()}
        return self.__get(data,'leaf')
    
    def get_clients_local_accuracy(self,test_dataset,test_labels):
        data = {'dataset': test_dataset.to_dict(), 'labels': test_labels.to_dict()}
        return self.__get(data,'local-accuracy')
        
    