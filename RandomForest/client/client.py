import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


class Client:
    def __init__(self, dataset=None) -> None:
        self.dataset = dataset
        self.forest = None
        self.labels = None
        self.test_dataset = None
        self.test_labels = None

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
        if total_gini == 0:
            return "pure", 0
        if len(dataset) <= 2:
            return "no-data", 0

        # calcul de gini pour chaque feature
        ds_star = dataset[features]
        thresholds = pd.DataFrame([splits], columns=features)
        ginis = ds_star.apply(lambda col: Client.gini_gain(col, labels, thresholds, total_gini), 0).values

        # retourne l'attribut permetant d'avoir le meilleur "gain de gini", ainsi que le nombre 
        # donnees dans le dataset courant

        i_best_gini = np.argmax(ginis)
        best_gini_feature = features[i_best_gini]
        n_data = len(labels)
        
        if ginis[i_best_gini] <= 0:
            return "no-gain", 0

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

        # retourner le nombre de valeurs perturbees pour chaque classe dans le dataset courant
        return labels

    def set_new_forest(self, random_forest):
        """Modifie la randomForest du client

        :param random_forest: nouvelle randomForest
        :type random_forest: RandomForest
        """
        self.forest = random_forest

    def get_federated_accuracy(self):
        """
        Calcule la précision de l'algo pour une distribution federee
        :return: Precision federee
        :rtype: float
        """
        res = [self.forest.predict(row) for index, row in self.test_dataset.iterrows()]
        accuracy = 1 - sum([int(value != self.test_labels[x]) for x, value in enumerate(res)]) / len(self.test_labels)

        return accuracy, len(self.test_dataset)

    def get_local_accuracy(self):
        """
        Calcule la précision de l'algo pour une distribution locale
        :return: Precision locale
        :rtype: float
        """
        # Entrainer un modele de randomForest (scikit-learn) et retourner l'accuracy
        dt = RandomForestClassifier()
        dt.fit(self.dataset.values, self.labels)
        res = dt.predict(self.test_dataset)
        accuracy = 1 - sum([int(value != self.test_labels[x]) for x, value in enumerate(res)]) / len(self.test_labels)
        return accuracy, len(self.test_dataset)

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
            col = dataset[f]
            minimum = col.min()
            maximum = col.max()
            if  np.isnan(minimum) or np.isnan(maximum):
                values.append(np.nan)
            else:
                values.append(np.random.default_rng().uniform(low=minimum, high=maximum))

        return values

    def get_features(self):
        """
        Recupere les features (colonnes) du dataset du client sous forme de liste

        :return: liste des features du client
        :rtype: list
        """
        return list(self.dataset.columns)

    def set_dataset(self, dataset, labels):
        """
        Mise a jour du dataset (et des labels) du client

        :param dataset: dataset a modifier
        :param labels: liste des labels a modifier
        """
        dataset = dataset.reset_index(drop=True)
        self.labels = labels

        labels = pd.DataFrame(labels).reset_index(drop=True)

        train_idx = np.random.choice(
            len(dataset) - 1, replace=False, size=int(len(dataset) * 0.8))

        self.test_dataset = dataset.loc[~dataset.index.isin(
            train_idx)].copy()
        self.test_labels = labels.loc[~labels.index.isin(train_idx)].values.T[0]

        self.dataset = dataset.loc[train_idx].copy()
        self.labels = labels.loc[train_idx].values.T[0]

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
        """
        Permet de calculer le gain de Gini

        :param col: Nom de la colonne associé au seuil (threshold)
        :type col: str
        :param y: Liste des differents labels
        :type y: list
        :param thresholds: Seuil de séparation du noeud de l'arbre associe a une colonne
        :type thresholds: float
        :param total_gini:
        :type total_gini: float
        :return: Le gain de Gini
        :rtype: float
        """
        threshold = thresholds[col.name][0]
        i_l = np.where(col <= threshold)[0]
        i_r = np.where(col > threshold)[0]

        l_gini = Client.gini_impurity(y[i_l])
        r_gini = Client.gini_impurity(y[i_r])
        sum_gini = total_gini - (len(i_l) / len(y)) * l_gini - (len(i_r) / len(y)) * r_gini

        return sum_gini
