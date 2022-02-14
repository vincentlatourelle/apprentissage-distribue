import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier


class Client():
    def __init__(self, dataset=None) -> None:
        self.dataset = dataset
        self.forest = None
        self.labels = None

    def __bootstrap(self, x):
        """ Effectue un bootstap sur le dataset x

        :param x: le dataset pour le bootstrap
        :type x: pd.Dataframe
        :return: Donnees selectionnees
        :rtype: pd.Dataframe
        """
        idx = np.random.choice(len(x) - 1, replace=True, size=len(x))
        return x.iloc[idx]

    def get_best_threshold(self, features, splits, current_tree):
        """Determine le threshold qui permet de mieux separer les donnees, selon la separation actuelle

        :param features: Liste des features a evaluer
        :type features: list
        :param splits: liste des thresholds associes aux features
        :type splits: list
        :param current_tree: arbre actuellement construit
        :type current_tree: Node
        """
        # Separer les donnees en fonctions de l'arbre courant 
        labels = self.labels.copy()
        dataset = self.dataset.copy()
        if current_tree is not None:
            dataset, labels = current_tree.get_current_node_data(dataset, labels)

        # Calcul du gini de l'ensemble actuel
        total_gini = Client.gini_impurity(labels)

        # Si rien est a separer
        if total_gini == 0 or len(dataset) <= 4:
            return features[0], 0

        # calcul de gini pour chaque feature
        ds_star = dataset[features]
        thresholds = pd.DataFrame([splits], columns=features)
        ginis = ds_star.apply(lambda col: Client.gini_gain(col, labels, thresholds, total_gini), 0).values

        # retourne l'attribut permetant d'avoir le meilleur "gain de gini", ainsi que le nombre 
        # donnees dans le dataset courant

        i_best_gini = np.argmax(ginis)
        best_gini_feature = features[i_best_gini]
        n_data = len(labels)

        return best_gini_feature, n_data

    def get_leaf(self, current_tree):
        """Obtient la distribution des classes pour un dataset 
           (possiblement juste la classe majoritaire si on decide d'utiliser un vote)

        :param current_tree: arbre actuellement evalue
        :type current_tree: Node
        """
        # Separer les donnees en fonctions de l'arbre courant 
        labels = self.labels.copy()
        dataset = self.dataset.copy()
        if current_tree is not None:
            dataset, labels = current_tree.get_current_node_data(dataset, labels)

        print(labels, file=sys.stderr)
        # retourner le nombre de valeurs perturbees pour chaque classe dans le dataset courant
        return labels

    def set_new_forest(self, random_forest):
        """Modifie la randomForest du client

        :param random_forest: nouvelle randomForest
        :type random_forest: RandomForest
        """
        self.forest = random_forest

    def get_local_accuracy(self, test_dataset, test_labels):

        # Entrainer un modele de randomForest (scikit-learn) et retourner l'accuracy

        dt = ExtraTreesClassifier()
        dt.fit(self.dataset.values, self.labels)
        res = dt.predict(test_dataset)
        return sum([int(value != test_labels[x]) for x, value in enumerate(res)]) / len(test_labels)

    def get_thresholds(self, features, current_tree):
        """ Pour chaque features, recupere le min et le max, puis definit le threshold qui
            est un valeur entre le min et le max

        :param features: Liste des features selectionnes
        :type features: list
        :param thresholds: thresholds selectionnes par les clients pour chaque feature
        :type thresholds: np.array de dimension n_client x n_feature
        :return: thresholds selectionnes pour chaque feature
        :rtype: np.array de dimension n_feature
        """
        labels = self.labels.copy()
        dataset = self.dataset.copy()
        if current_tree is not None:
            dataset, labels = current_tree.get_current_node_data(dataset, labels)

        values = []
        for f in features:
            col = self.dataset[f]
            min = col.min()
            max = col.max()
            values.append(np.random.default_rng().uniform(low=min, high=max))

        return values

    @staticmethod
    def gini_impurity(y):
        """ Calcul l'impureté de Gini

        :param y: Liste des différents labels
        :type y: list
        :return: L'impureté de Gini (entre 0 et 1)
        :rtype: float
        """

        l, count = np.unique(y, return_counts=True)
        prob = count / len(y)
        return 1 - np.sum(np.power(prob, 2))

    @staticmethod
    def gini_gain(col, y, thresholds, total_gini):
        threshold = thresholds[col.name][0]
        i_l = np.where(col <= threshold)[0]
        i_r = np.where(col > threshold)[0]

        l_gini = Client.gini_impurity(y[i_l])
        r_gini = Client.gini_impurity(y[i_r])
        sum_gini = total_gini - (len(i_l) / len(y)) * l_gini - (len(i_r) / len(y)) * r_gini

        return sum_gini
