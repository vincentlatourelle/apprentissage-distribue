from collections import defaultdict
import numpy as np

import os
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
rootdir = os.path.join(currentdir, "../../")
sys.path.append(rootdir)

from RandomForest.randomForest import RandomForest
from RandomForest.node import Node

class FederatedRandomForest:
    """Classe qui gere le processus d'entrainement d'une random forest en mode federe
    """

    def __init__(self, server_manager) -> None:
        self.server_manager = server_manager
        self.forest = RandomForest()
        self.features = None

    def select_features(self):
        """Choisi aleatoirement entre 2 et racine carre du nombre de feature + 4 features parmis les features des clients, sans repetition de feature.

        Returns:
            np.array: Array contenant les features (str)
        """

        return np.random.choice(self.features, replace=False,
                                size=np.random.randint(2, np.sqrt(len(self.features)) + 4))

    def get_thresholds(self, features, thresholds):
        """ Pour chaque features, recupere le min et le max, puis definit le threshold qui
            est un valeur entre le min et le max

        Args:
            features (list): Liste des features selectionnes
            thresholds (np.array): thresholds selectionnes par les clients pour chaque feature

        Returns:
            np.array: thresholds selectionnes pour chaque feature
        """

        values = np.array([])

        for f in range(len(features)):
            col = thresholds[:, f]
            min = np.nanmin(col)
            max = np.nanmax(col)

            # Prend une valeur entre la valeur min et max obtenue
            values = np.append(
                values, np.random.default_rng().uniform(low=min, high=max))

        return values

    def get_label(self, current_tree):
        """Recupere les labels des clients 

        Args:
            current_tree (Node): Racine de l'arbre en developpement

        Returns:
            list: liste des labels (cibles)
        """

        # Recoit une liste des labels de chaque client pour le noeud courant
        data = {"current_tree": current_tree.serialize()}
        labels = np.concatenate(self.server_manager.get(
            data, 'rf/leaf').tolist(), axis=0)

        # print("<-- Le master recoit les labels des clients")
        # print(labels)
        # print("**************************************************************************************")

        # Fait un vote majoritaire
        result, count = np.unique(labels, return_counts=True)

        return result[np.argmax(count)]

    def get_label_vote(self, current_tree):
        """Fait un vote majoritaire selon les cibles majoritaires chez les clients
           Un vote et un poids est obtenu de chaque client avec rf/leaf-vote et cette fonction
           concatene ces votes pour construire la feuille.

        Args:
            current_tree (Node): Racine de l'arbre en developpement

        Returns:
            str: label (cible) obtenu par le vote majoritaire
        """
        
        # Recoit un couple (label, nombre de donnees) de chaque client
        data = {"current_tree": current_tree.serialize()}
        labels = self.server_manager.get(data, 'rf/leaf-vote')

        # print("<-- Le master recoit les labels des clients")
        # print(labels)
        # print("**************************************************************************************")
        
        # Fait un vote majoritaire
        votes = defaultdict(int)
        for v in labels:
            votes[v["label"]] += v["count"]

        label = max(votes, key=votes.get)
        return label

    def build_tree(self, current_node, current_root, depth=15):
        """Construit un arbre decisionnel de facon federe, recursivement

        Args:
            current_node (Node): Noeud ou assigner un separation (threshold) ou une valeur de feuille
            current_root (Node): racine de l'arbre en developpement
            depth (int, optional): profondeur de l'arbre (condition d'arret). Defaults to 15.

        Returns:
            Node: racine de l'arbre developpe
        """

        # Si la limite de profondeur est atteinte
        if depth == 0:
            current_node.value = self.get_label_vote(current_root)
            return current_node.value

        features = self.select_features()

        # print("--> Le master envoie les features aux clients")
        # print(features)

        thresholds = self.server_manager.get({"features": features.tolist(
        ), "current_tree": current_root.serialize()}, 'rf/thresholds')

        # print("<-- Le master recoit les thresholds des clients")
        # print(thresholds)
        # print("**************************************************************************************")

        thresholds = self.get_thresholds(features, thresholds)

        # print("--> Le master envoie les thresholds selectionnes aux clients")
        # print(thresholds)

        best_threshold = self.server_manager.get({"features": features.tolist(
        ), "thresholds": thresholds.tolist(), "current_tree": current_root.serialize()}, 'rf/best-threshold')

        # print("<-- Le master recoit les meilleurs features et le nombre de donnees actuels des clients")
        # print(best_threshold)
        # print("**************************************************************************************")
        
        # Vote majoritaire pour avoir la meilleure separation
        votes = dict.fromkeys(features, 0)
        votes['pure'] = 0
        votes['no-data'] = 0
        votes["no-gain"] = 0

        for c in best_threshold:
            votes[c['feature']] += c["n_data"]

        best_feature = max(votes, key=votes.get)

        # Si personne ne vote
        if votes[best_feature] == 0:
            current_node.value = self.get_label_vote(current_root)
            return current_node.value

        # Ajouter le meilleur feature et separation a "current_tree"
        current_node.feature = best_feature
        current_node.threshold = thresholds[np.where(
            features == best_feature)][0]

        # Construit l'arbre de gauche
        lNode = Node()
        current_node.lNode = lNode
        self.build_tree(lNode, current_root, depth - 1)
        # Construit l'arbre de droite
        rNode = Node()
        current_node.rNode = rNode
        self.build_tree(rNode, current_root, depth - 1)

        return current_root

    def train(self, n=100, depth=3):
        """Entraine le model en contruisant n arbre de facon distribuee

        Args:
            n (int, optional): nombre d'arbres. Defaults to 100.
            depth (int, optional): profondeur maximal des arbres. Defaults to 3.
        """

        self.get_clients_features()

        for t in range(n):
            current_tree = Node()
            self.build_tree(current_tree, current_tree, depth=depth)

            # Ajouter current_tree a la foret
            self.forest.add(current_tree)

        self.send_forest()

    def send_forest(self):
        """ Envoie la foret construite aux clients
        """
        # Envoyer la foret aux clients
        json_forest = self.forest.serialize()

        self.server_manager.post(
            [{'forest': json_forest}] * len(self.server_manager.clients), 'rf/random-forest')

    def get_clients_features(self):
        """Récupère les features des clients
        """

        self.features = self.server_manager.get(None, 'rf/features')[0]
