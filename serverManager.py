import requests
from multiprocessing import Pool
import numpy as np


def inner_get(url, data):
    return requests.get(url, json=data, headers={"Content-Type": "application/json; charset=utf-8"}).json()


def inner_post(url, data):
    return requests.post(url, json=data, headers={"Content-Type": "application/json; charset=utf-8"})


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

        return np.array(r)

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

    def __del__(self):
        self.pool.close()
        self.pool.join()
