import numpy as np


class Node:
    """Noeud d'un arbre de decision
    """

    def __init__(self, feature=None, threshold=None, lNode=None, rNode=None, value=None) -> None:
        self.feature = feature
        self.threshold = threshold
        self.lNode = lNode
        self.rNode = rNode
        self.value = value

    def predict(self, x):
        """Prediction de l'arbre en parcourant l'arbre de decision pour une donnee x

        Args:
            x (pd.Series): la donnee a predire

        Returns:
            str: label qui correspond a la valeur predite
        """
        
        # Si c'est une feuille
        if self.value is not None:
            return self.value
        
        # Si c'est un noeud
        if x[self.feature] <= self.threshold:
            return self.lNode.predict(x)
        else:
            return self.rNode.predict(x)

    def get_custom_dict(self):
        """Retourne un dictionnaire pour la serialisation

        Returns:
            dict: dictionnaire decrivant le Node
        """

        # Si le noeud est une feuille
        if self.value is not None:
            custom_dict = {
                "value": self.value,
            }
        # Si c'est le noeud actuellement developpe
        elif self.feature is None:
            return {}
        else:
            custom_dict = {
                "feature": self.feature,
                "threshold": self.threshold,
            }

        # Parcourt recursivement les noeuds enfants (gauche et droite)
        if self.lNode:
            custom_dict['lNode'] = self.lNode.get_custom_dict()
        if self.rNode:
            custom_dict["rNode"] = self.rNode.get_custom_dict()

        return custom_dict

    def serialize(self):
        """Retourne le dictionnaire a serialiser

        Returns:
            dict: Structure en json du noeud
        """

        return self.get_custom_dict()

    @staticmethod
    def deserialize(tree_dict):
        """Deserialise un json pour creer un Node

        Args:
            tree_dict (dict): Structure en json du noeud

        Returns:
            Node: racine de l'arbre deserialise
        """

        if len(tree_dict) == 0:
            return Node()

        new_tree = Node()
        # Pour chaque attribut (feature) de l'objet (chaque cle du dict)
        for f in tree_dict:
            if f == "lNode":
                new_tree.lNode = Node.deserialize(tree_dict[f])
            elif f == "rNode":
                new_tree.rNode = Node.deserialize(tree_dict[f])
            else:
                new_tree.__dict__[f] = tree_dict[f]

        return new_tree

    def get_current_node_data(self, dataset, labels):
        """ Obtient les donnees separees en fonction de l'arbre actuel et du noeud courant

        Args:
            dataset (pd.DataFrame): Donnees a separer
            labels (pd.Series): Labels associes aux donnees

        Returns:
            tuple: tuple contenant :
                df (pd.DataFrame): 
                    Sous-ensemble du dataset initial contenant les donnees a evaluer au noeud courant
                new_labels (pd.Series): 
                    Sous-ensemble des labels initials contenant les labels a evaluer au noeud courant
        """

        # Si c'est une feuille retourner null
        if not self.value is None:
            return None, None

        # Si c'est un noeud non initialise correctement, garder le dataset
        if self.feature is None:
            return dataset, labels

        # Separe en deux dataset selon le threshold
        i_l = np.where(dataset[self.feature].values <= self.threshold)
        i_r = np.where(dataset[self.feature].values > self.threshold)
        ldf = dataset.iloc[i_l]
        rdf = dataset.iloc[i_r]

        llabels = labels[i_l]
        rlabels = labels[i_r]

        # si le noeud gauche est present, l'explorer et retourner ce qu'il retourne si ce n'est pas null
        ldf, l_new_labels = self.lNode.get_current_node_data(ldf, llabels)
        if not ldf is None:
            return ldf, l_new_labels

        # si le noeud droite est present, l'explorer et retourner ce qu'il retourne si ce n'est pas null
        rdf, r_new_labels = self.rNode.get_current_node_data(rdf, rlabels)
        if not rdf is None:
            return rdf, r_new_labels

        return None, None
