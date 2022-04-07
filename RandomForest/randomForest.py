import numpy as np
from RandomForest.node import Node


class RandomForest:
    """Random Forest (collection d'arbre)
    """

    def __init__(self) -> None:

        self.forest = []

    def add(self, node):
        """Ajoute un arbre a la randomForest

        Args:
            node (Node): noeud a la racine de l'arbre
        """
        
        self.forest.append(node)

    def predict(self, x):
        """Prediction de la randomforest pour une donnee

        Args:
            x (pd.DataFrame): la donnee a predire

        Returns:
            str: label de la valeur predite
        """

        f_result = []
        # Fait une prediction pour chaque donnee du dataframe en predisant 
        # avec chaque arbre de la foret
        for index, x_i in x.iterrows():
            result = [tree.predict(x_i) for tree in self.forest]
            result, count = np.unique(result, return_counts=True)
            f_result.append(result[np.argmax(count)])
        return f_result

    def serialize(self):
        """Serialisation de la foret aleatore

        Returns:
            list: structure json contenant une liste des arbres de la foret
        """

        return [x.serialize() for x in self.forest]

    def deserialize(self, forest):
        """Deserialisation du Random Forest (json --> dictionnaire)

        Args:
            forest (list): Liste des arbres de la foret
        """

        for x in forest:
            self.add(Node.deserialize(x))
