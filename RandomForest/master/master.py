from flask import jsonify
import numpy as np
from serverManager import ServerManager
class Master():
    def __init__(self, dataset, labels) -> None:
        self.dataset = dataset
        self.labels = labels
        self.test_dataset = None
        self.test_labels = None
        self.server_manager = ServerManager()
    
    def split_dataset(self,n_clients):
        train_idx = np.random.choice(len(self.dataset)-1, replace=False ,size = int(len(self.dataset)*0.8))
        
        self.test_dataset = self.dataset.loc[~self.dataset.index.isin(train_idx)]
        self.test_labels = self.labels.loc[~self.labels.index.isin(train_idx)]
        
        step = int(len(self.labels)/n_clients)
        stop = len(self.labels)
        
        train_datasets = np.split(self.dataset, [x for x in range(step,stop,step)])
        train_labels = np.split(self.labels, [x for x in range(step,stop,step)])
        
        self.server_manager.send_dataset_to_client(train_datasets,train_labels, n_clients)
        
        
        
        
        