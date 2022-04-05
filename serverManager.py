import io
from joblib import dump, load
import requests
from multiprocessing import Pool
import numpy as np


def inner_get(url, data):
    return requests.get(url, json=data, headers={"Content-Type": "application/json; charset=utf-8"})


def inner_post(url, data):
    return requests.post(url, json=data, headers={"Content-Type": "application/json; charset=utf-8"})


def inner_post_file(url, file):
    return requests.post(url, files=file)


class ServerManager():
    """Gere l'envoye de requete aux clients
    """

    def __init__(self, clients) -> None:
        self.clients = clients
        self.pool = Pool(len(self.clients))

    def get(self, data, uri):
        """Effectue un HTTP GET pour chaque client et retourne leurs reponses

        Args:
            data (dict): json a envoyer au client
            uri (str): ressource a acceder

        Returns:
            list: Reponse des clients
        """

        response = self.pool.starmap(inner_get, zip(
            [f'{x}/{uri}' for x in self.clients], [data] * len(self.clients)))
        json = [resp.json() for resp in response]
        return np.array(json, dtype=object)

    def post(self, data, uri):
        """Effectue un HTTP POST pour chaque client et retourne leurs reponses

        Args:
            data (dict): Json a envoyer au client
            uri (str): ressource a acceder

        Returns:
            list: Reponse des clients
        """

        response = self.pool.starmap(inner_post, zip(
            [f'{x}/{uri}' for x in self.clients], data))
        return response

    def post_model(self, uri, model):
        """Effectue un HTTP POST pour envoyer un model aux clients

        Args:
            uri (str): ressource a acceder
            model (any): Model scikit-learn a envoyer

        Returns:
            list: reponse des clients
        """

        # Ajoute le modele dans un bytestream
        bytes_io = io.BytesIO()
        dump(model, bytes_io)
        bytes_io.seek(0)
        file = {'file': ('file', bytes_io)}

        response = self.pool.starmap(inner_post_file, zip(
            [f'{x}/{uri}' for x in self.clients], [file] * len(self.clients)))
        return response

    def get_models(self, data, uri):
        """Effectue un HTTP GET pour recevoir les modeles locals entraines chez les clients

        Args:
            data (dict): Json a envoyer au client
            uri (str): ressource a acceder

        Returns:
            list: listes des modeles entraines chez les clients
        """
        response = self.pool.starmap(inner_get, zip(
            [f'{x}/{uri}' for x in self.clients], [data] * len(self.clients)))
        
        # Load chacun des modeles dans une liste
        models = [load(io.BytesIO(resp.content)) for resp in response]
        return models

    def __del__(self):
        self.pool.terminate()
        self.pool.close()
        self.pool = None
