import requests
from flask import jsonify
import json

class ServerManager():
    def __init__(self) -> None:
        self.clients = ["http://localhost:5001","http://localhost:5002"]
        
    def send_dataset_to_client(self,dataset, labels, n_clients):
        
        for client in range(n_clients):
            
            r = requests.post(f'{self.clients[client]}/dataset', json={'dataset': dataset[client].to_dict(), 'labels': labels[client].to_dict()}, headers={"Content-Type":"application/json; charset=utf-8"})
            print(r)