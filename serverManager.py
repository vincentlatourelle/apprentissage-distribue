import io
from joblib import dump, load
import requests
from multiprocessing import Pool
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def inner_get(url, data):
    return requests.get(url, json=data, headers={"Content-Type": "application/json; charset=utf-8"})


def inner_post(url, data):
    return requests.post(url, json=data, headers={"Content-Type": "application/json; charset=utf-8"})


def inner_post_file(url, file):
    return requests.post(url, files=file)
class ServerManager():
    def __init__(self, clients) -> None:
        self.clients = clients
        self.pool = Pool(len(self.clients))

    def get(self, data, uri):
        """Effectue un HTTP GET pour chaque client et retourne leurs reponses

        :param data: Json a envoyer au client
        :type data: dict
        :param uri: ressource a acceder
        :type uri: str
        :return: Reponse des clients
        :rtype: list
        """
        r = self.pool.starmap(inner_get, zip([f'{x}/{uri}' for x in self.clients], [data] * len(self.clients)))
        json = [resp.json() for resp in r]
        return np.array(json,dtype=object)

    def post(self, data, uri):
        """Effectue un HTTP POST pour chaque client et retourne leurs reponses

        :param data: Json a envoyer au client
        :type data: dict
        :param uri: ressource a acceder
        :type uri: str
        :return: Reponse des clients
        :rtype: list
        """
        r = self.pool.starmap(inner_post, zip([f'{x}/{uri}' for x in self.clients], data))
        return r
    
    def post_model(self, uri, file):
        """Effectue un HTTP POST pour chaque client et retourne leurs reponses

        :param data: Json a envoyer au client
        :type data: dict
        :param uri: ressource a acceder
        :type uri: str
        :return: Reponse des clients
        :rtype: list
        """
        
        r = self.pool.starmap(inner_post_file, zip([f'{x}/{uri}' for x in self.clients], [file] * len(self.clients)))
        return r
    
    def get_models(self,data,uri):
        r = self.pool.starmap(inner_get, zip([f'{x}/{uri}' for x in self.clients], [data] * len(self.clients)))
        models = [load(io.BytesIO(resp.content)) for resp in r]
        return models
        

    def __del__(self):
        self.pool.close()
        self.pool.join()
