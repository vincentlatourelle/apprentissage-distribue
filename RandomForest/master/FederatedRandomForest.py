import numpy as np

import os
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
rootdir = os.path.join(currentdir, "../../")
sys.path.append(rootdir)

from RandomForest.randomForest import RandomForest
from RandomForest.node import Node


class FederatedRandomForest():
    def __init__(self, server_manager) -> None:
        self.server_manager = server_manager
        self.forest = RandomForest()
        self.features = None

    def select_features(self):
        """ Choisi un nombre aléatoire de features parmi la liste de features

        :return: Liste de n features
        :rtype: np.array de dimension n_features
        """

        return np.random.choice(self.features, replace=False, size=np.random.randint(2, np.sqrt(len(self.features)) + 4))

    def get_thresholds(self, features, thresholds):
        """ Pour chaque features, recupere le min et le max, puis definit le threshold qui
            est un valeur entre le min et le max

        :param features: Liste des features selectionnes
        :type features: list
        :param thresholds: thresholds selectionnes par les clients pour chaque feature
        :type thresholds: np.array de dimension n_client x n_feature
        :return: thresholds selectionnes pour chaque feature
        :rtype: np.array de dimension n_feature
        """
        values = np.array([])

        for f in range(len(features)):
            col = thresholds[:, f]
            min = np.nanmin(col)
            max = np.nanmax(col)

            values = np.append(
                values, np.random.default_rng().uniform(low=min, high=max))

        return values

    def get_label(self, current_tree):
        data = {"current_tree": current_tree.serialize()}
        labels = np.concatenate(self.server_manager.get(data, 'leaf').tolist(), axis=0)
        
        print("<-- Le master recoit les labels des clients")
        print(labels)
        print("**************************************************************************************")
        
        result, count = np.unique(labels, return_counts=True) 
        if len(result) == 0:
            return "resultat_invalid"
        return result[np.argmax(count)]
    
    def build_tree(self, current_node,current_root, depth=15):
        """Construit un arbre de facon distribuee

        :param current_tree: arbre actuellement en construction
        :type current_tree: Node
        """
        if depth == 0:
            current_node.value = self.get_label(current_root)
            return current_node.value

        features = self.select_features()
        
        print("--> Le master envoie les features aux clients")
        print(features)
        
        
        thresholds = self.server_manager.get({"features": features.tolist(), "current_tree": current_root.serialize()},'thresholds')

        print("<-- Le master recoit les thresholds des clients")
        print(thresholds)
        print("**************************************************************************************")
        

        thresholds = self.get_thresholds(features, thresholds)
        
        print("--> Le master envoie les thresholds selectionnes aux clients")
        print( thresholds )
        
        best_threshold = self.server_manager.get({"features": features.tolist(), "thresholds": thresholds.tolist(),
                "current_tree": current_root.serialize()},'best-threshold')
        
        print("<-- Le master recoit les meilleurs features et le nombre de donnees actuels des clients")
        print( best_threshold )
        print("**************************************************************************************")
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
            current_node.value = self.get_label(current_root)
            return current_node.value
        
        # Ajouter le meilleur feature et separation a "current_tree"
        current_node.feature = best_feature
        current_node.threshold = thresholds[np.where(features == best_feature)][0]
        
        # Construit l'arbre de gauche
        lNode = Node()
        current_node.lNode = lNode
        lres = self.build_tree(lNode, current_root, depth - 1)
        # Construit l'arbre de droite
        rNode = Node()
        current_node.rNode = rNode
        rres = self.build_tree(rNode, current_root, depth - 1)

        # Fusionne si la separation etait invalide
        # Devrait pas etre utilise
        if lres == "resultat_invalid":
            current_node.lNode.value = current_node.rNode.left_most_leaf()
        elif rres == "resultat_invalid":
            current_node.rNode.value = current_node.lNode.right_most_leaf()

        return current_root

    def train(self, n=100,depth=3):
        """ Entraine le modele en construisant un arbre de facon distribuee puis en
            l'ajoutant au Random Forest
        """
        self.get_clients_features()
        
        for t in range(n):
            current_tree = Node()
            self.build_tree(current_tree, current_tree, depth=depth)

            # Ajouter current_tree a la foret
            self.forest.add(current_tree)

        # Envoyer la foret aux clients
        json_forest = self.forest.serialize() 
        
        self.server_manager.post([{'forest':json_forest}]*len(self.server_manager.clients), 'random-forest')

    def get_clients_features(self):
        self.features = self.server_manager.get({}, 'features')[0]

