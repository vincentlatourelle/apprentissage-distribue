import numpy as np


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
        result = [tree.predict(x) for tree in self.forest]
        result, count = np.unique(result, return_counts=True)
        return result[np.argmax(count)]
