import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from serverManager import ServerManager

import os
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
rootdir = os.path.join(currentdir, "../../")
sys.path.append(rootdir)

from RandomForest.randomForest import RandomForest
from RandomForest.node import Node


class Master():
    def __init__(self, dataset, labels, server_manager: ServerManager) -> None:
        self.dataset = dataset
        self.labels = labels
        self.test_dataset = None
        self.test_labels = None
        self.server_manager = server_manager
        self.forest = RandomForest()

    def select_features(self):
        """ Choisi un nombre aléatoire de features parmi la liste de features

        :return: Liste de n features
        :rtype: np.array de dimension n_features
        """

        features = self.dataset.columns
        return np.random.choice(features, replace=False, size=np.random.randint(2, np.sqrt(len(features)) + 4))

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
            min = np.min(col)
            max = np.max(col)

            values = np.append(
                values, np.random.default_rng().uniform(low=min, high=max))

        return values

    def get_label(self, current_tree):
        labels = np.concatenate(self.server_manager.get_leafs(current_tree).tolist(), axis=0)
        # print(labels)
        result, count = np.unique(labels, return_counts=True)
        if len(result) == 0:
            return "resultat_invalid"
        return result[np.argmax(count)]

    def build_tree(self, current_node, current_root, depth=5):
        """Construit un arbre de facon distribuee

        :param current_tree: arbre actuellement en construction
        :type current_tree: Node
        """
        if depth == 0:
            current_node.value = self.get_label(current_root)
            return current_node.value

        features = self.select_features()

        thresholds = self.server_manager.get_thresholds(features.tolist(), current_root)

        # print(thresholds)

        thresholds = self.get_thresholds(features, thresholds)

        # print(thresholds)

        best_threshold = self.server_manager.get_best_threshold_from_clients(
            features, thresholds, current_root)

        # print(best_threshold)

        # Vote majoritaire pour avoir la meilleure separation
        votes = dict.fromkeys(features, 0)

        for c in best_threshold:
            votes[c['feature']] += c["n_data"]

        best_feature = max(votes, key=votes.get)

        # Si personne ne vote
        if votes[best_feature] == 0:
            current_node.value = self.get_label(current_root)
            return current_node.value

        # print(best_feature)
        # Ajouter le meilleur feature et separation a "current_tree"
        current_node.feature = best_feature
        current_node.threshold = thresholds[np.where(features == best_feature)][0]

        # print(features)
        # print(thresholds)
        # print(current_node.threshold)
        # Construit l'arbre de gauche
        lNode = Node()
        current_node.lNode = lNode
        lres = self.build_tree(lNode, current_root, depth - 1)
        # Construit l'arbre de droite
        rNode = Node()
        current_node.rNode = rNode
        rres = self.build_tree(rNode, current_root, depth - 1)

        # Fusionne si la separation etait invalide (surement un probleme qui fait qu'on a des ensembles vide relativement souvent)
        if lres == "resultat_invalid":
            current_node.lNode.value = current_node.rNode.left_most_leaf()
        elif rres == "resultat_invalid":
            current_node.rNode.value = current_node.lNode.right_most_leaf()

        return current_root

    def train(self, n=100):
        """ Entraine le modele en construisant un arbre de facon distribuee puis en
            l'ajoutant au Random Forest
        """
        for t in range(n):
            current_tree = Node()
            self.build_tree(current_tree, current_tree, depth=15)

            # Ajouter current_tree a la foret
            self.forest.add(current_tree)

        # Envoyer la foret aux clients

    def get_federated_accuracy(self):
        res = [self.forest.predict(row) for index, row in self.test_dataset.iterrows()]
        print(sum([int(value != self.test_labels.values[x]) for x, value in enumerate(res)]) / len(self.test_labels))

    def get_centralised_accuracy(self):
        dt = ExtraTreesClassifier()
        dt.fit(self.dataset.values, self.labels.values)
        res = dt.predict(self.test_dataset)
        print(sum([int(value != self.test_labels.values[x]) for x, value in enumerate(res)]) / len(self.test_labels))

    def get_local_accuracy(self):
        print(np.mean(self.server_manager.get_clients_local_accuracy(self.test_dataset, self.test_labels)))
