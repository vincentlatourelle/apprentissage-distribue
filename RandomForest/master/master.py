from flask import jsonify
import numpy as np
from serverManager import ServerManager

import os
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
rootdir = os.path.join(currentdir, "../../")
sys.path.append(rootdir)

from RandomForest.node import Node

class Master():
    def __init__(self, dataset, labels, server_manager: ServerManager) -> None:
        self.dataset = dataset
        self.labels = labels
        self.test_dataset = None
        self.test_labels = None
        self.server_manager = server_manager

    def split_dataset(self):
        n_clients = len(self.server_manager.clients)
        train_idx = np.random.choice(
            len(self.dataset)-1, replace=False, size=int(len(self.dataset)*0.8))

        self.test_dataset = self.dataset.loc[~self.dataset.index.isin(
            train_idx)]
        self.test_labels = self.labels.loc[~self.labels.index.isin(train_idx)]

        step = int(len(self.labels)/n_clients)
        stop = len(self.labels)

        train_datasets = np.split(
            self.dataset, [x for x in range(step, stop, step)])
        train_labels = np.split(
            self.labels, [x for x in range(step, stop, step)])

        self.server_manager.send_dataset_to_client(
            train_datasets, train_labels)

    def select_features(self):
        features = self.dataset.columns
        return np.random.choice(features, replace=False, size=np.random.randint(2, np.sqrt(len(features))+2))

    def get_thresholds(self, features, thresholds):

        values = np.array([])
        for f in range(len(features)):
            col = thresholds[:, f]
            min = np.min(col)
            max = np.max(col)

            values = np.append(
                values, np.random.default_rng().uniform(low=min, high=max))

        return values

    def build_tree(self, current_tree):
        features = self.select_features()

        thresholds = self.server_manager.get_thresholds(features.tolist())

        print(thresholds)

        thresholds = self.get_thresholds(features, thresholds)

        print(thresholds)

        best_threshold = self.server_manager.get_best_threshold_from_clients(
            features, thresholds, current_tree)

        # Vote majoritaire pour avoir la meilleure separation

        # Ajouter le meilleur feature et separation a "current_tree"

        # Construit l'arbre de gauche

        # Construit l'arbre de droite

    def train(self):
        current_tree = Node()
        current_tree = self.build_tree(current_tree)

        # Ajouter current_tree a la foret
