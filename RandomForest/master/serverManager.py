import requests
from flask import jsonify
import json
from multiprocessing import Pool
import numpy as np


class ServerManager():
    def __init__(self, clients) -> None:
        self.clients = clients
        
    def __get(self, data, endpoint):
        values = []
        for client in self.clients:
            r = requests.get(f'{client}/{endpoint}',json=data, headers={"Content-Type":"application/json; charset=utf-8"})
            
            values.append(r.json())
            
        return np.array(values)
        
            
    def send_dataset_to_client(self,dataset, labels):
        
        for client in range(len(self.clients)):
            requests.post(f'{self.clients[client]}/dataset', json={'dataset': dataset[client].to_dict(), 'labels': labels[client].to_dict()}, headers={"Content-Type":"application/json; charset=utf-8"})
            
    def get_thresholds(self, features):
        
        data = {"features" : features}
        return self.__get(data,'thresholds')
            
    def get_best_threshold_from_clients(self,features, thresholds, current_tree):
        
        data = {"features" : features.tolist(), "thresholds": thresholds.tolist(), "current_tree":current_tree.serialize()}
        return self.__get(data,'best-threshold')
    
    