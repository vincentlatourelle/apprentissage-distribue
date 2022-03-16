import numpy as np
from RandomForest.node import Node


class RandomForest:
    def __init__(self) -> None:
        self.forest = []

    def add(self, node):
        """Ajoute un arbre a la randomForest

        :param node: racine de l'arbre
        :type node: Node
        """
        self.forest.append(node)

    def predict(self, x):
        """ Prediction de la randomforest pour une donnee

        :param x: la donnee a predire
        :type x: np.array
        :return: label
        :rtype: str
        """
        f_result = []
        for index, x_i in x.iterrows():
            result = [tree.predict(x_i) for tree in self.forest]
            result, count = np.unique(result, return_counts=True)
            f_result.append(result[np.argmax(count)])
        return f_result
    
    def serialize(self):
        """
        Serialisation du Random Forest (dictionnaire --> json)

        :return: Arbre serialise
        :rtype: str (json)
        """
        return [x.serialize() for x in self.forest]
    
    def deserialize(self, forest):
        """
        Deserialisation du Random Forest (json --> dictionnaire)

        :param forest:
        :type forest:
        :return: Arbre deserialise
        :rtype: dict
        """
        for x in forest:
            self.add(Node.deserialize(x))
