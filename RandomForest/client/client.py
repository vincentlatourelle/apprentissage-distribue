import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


class Client:
    """Classe qui gere les operations des clients dans la construction d'une random forest ferere
    """

    def __init__(self, dataset=None) -> None:
        self.dataset = dataset
        self.forest = None
        self.labels = None
        self.test_dataset = None
        self.test_labels = None

        self.current_dataset = None
        self.current_labels = None

    def __bootstrap(self, x):
        """Effectue un bootstap sur le dataset x

        Args:
            x (pd.Dataframe): le dataset utilise pour le bootstrap

        Returns:
            pd.Dataframe: Donnees selectionnees du dataset
        """

        idx = np.random.choice(len(x) - 1, replace=True, size=len(x))
        return x.iloc[idx]

    def get_best_threshold(self, features, splits, current_tree):
        """Obtient le couple feature, valeur de separation qui ameliore le plus
           l'indice de gini pour le dataset du client.

        Args:
            features (list): liste des features a evaluer
            splits (list): liste des valeurs de separation associees aux features
            current_tree (Node): Arbre actuellement developpe

        Returns:
            tuple: tuple contenant:
                str: nom de l'attribut selectionne
                int: nombre de donnees actuellement evalues
        """

        # Separer les donnees en fonctions de l'arbre courant
        labels = self.current_labels
        dataset = self.current_dataset

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
        ginis = ds_star.apply(lambda col: Client.gini_gain(
            col, labels, thresholds, total_gini), 0).values

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

        Args:
            current_tree (Node): arbre actuellement evalue

        Returns:
            list: labels (cibles)
        """
        """Obtient la distribution des classes pour un dataset 
           (possiblement juste la classe majoritaire si on decide d'utiliser un vote)

        :param current_tree: arbre actuellement evalue
        :type current_tree: Node
        """
        # Separer les donnees en fonctions de l'arbre courant
        labels = self.labels.copy()
        dataset = self.dataset.copy()
        if current_tree is not None:
            dataset, labels = current_tree.get_current_node_data(
                dataset, labels)

        # retourner le nombre de valeurs perturbees pour chaque classe dans le dataset courant
        return labels

    def get_leaf_vote(self, current_tree):
        """Obtient la classe majoritaire selon les cibles majoritaires chez les clients

        Args:
            current_tree (Node): arbre actuellement evalue

        Returns:
            tuple: tuple contenant les valeurs suivantes:
                label (cible) majoritaire
                nombre de labels au total
        """

        # Separer les donnees en fonctions de l'arbre courant
        labels = self.labels.copy()
        dataset = self.dataset.copy()
        if current_tree is not None:
            dataset, labels = current_tree.get_current_node_data(
                dataset, labels)

        # retourner le nombre de valeurs perturbees pour chaque classe dans le dataset courant
        if len(labels) > 0:
            result, count = np.unique(labels, return_counts=True)

            return result[np.argmax(count)], len(self.labels)

        return "", 0

    def set_new_forest(self, random_forest):
        """Modifie la randomForest du client

        Args:
            random_forest (Node): Nouvelle RandomForest
        """
        self.forest = random_forest

    def get_federated_accuracy(self):
        """Calcule la precision de l'arbre entraine de facon federe

        Returns:
            tuple: tuple contentant:
                float: justesse (accuracy)
                int: nombre de donnees dans l'ensemble de test
        """

        res = self.forest.predict(self.test_dataset)

        accuracy = 1 - sum([int(value != self.test_labels[x])
                           for x, value in enumerate(res)]) / len(self.test_labels)

        return accuracy, len(self.test_dataset)

    def get_local_accuracy(self):
        """Calcule la precision d'un arbre entraine localement et teste localement

        Returns:
            tuple: tuple contentant:
                float: justesse (accuracy)
                int: nombre de donnees dans l'ensemble de test
        """

        # Entrainer un modele de randomForest (scikit-learn) et retourner l'accuracy
        dt = RandomForestClassifier()
        dt.fit(self.dataset, self.labels)
        res = dt.predict(self.test_dataset)
        accuracy = 1 - sum([int(value != self.test_labels[x])
                           for x, value in enumerate(res)]) / len(self.test_labels)
        return accuracy, len(self.test_dataset)

    def get_local_model(self):
        """Retourne un model entraine localement

        Returns:
            RandomForestClassifier: Model scikit-learn entraine localement
        """
        dt = RandomForestClassifier()
        dt.fit(self.dataset, self.labels)
        return dt

    def get_thresholds(self, features, current_tree):
        """Pour chaque features, recupere le min et le max, puis definit le threshold qui
            est un valeur entre le min et le max

        Args:
            features (list): Liste des features selectionnes par le master
            current_tree (Node): Arbre actuel (pour la separation des donnees)

        Returns:
            list: Array contenant une separation pour chaque feature
        """

        labels = self.labels.copy()
        dataset = self.dataset.copy()
        if current_tree is not None:
            dataset, labels = current_tree.get_current_node_data(
                dataset, labels)

        self.current_dataset = dataset
        self.current_labels = labels

        values = []
        for f in features:
            col = dataset[f]
            minimum = col.min()
            maximum = col.max()
            if np.isnan(minimum) or np.isnan(maximum):
                values.append(np.nan)
            else:
                values.append(np.random.default_rng().uniform(
                    low=minimum, high=maximum))

        return values

    def get_features(self):
        """Recupere les features (colonnes) du dataset du client sous forme de liste

        Returns:
            list: Liste des features du client
        """

        return list(self.dataset.columns)

    def set_dataset(self, dataset, labels):
        """Mise a jour du dataset (et des labels) du client

        Args:
            dataset (pd.DataFrame): dataset a modifier
            labels (list): liste des labels a modifier
        """
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
        self.test_labels = labels.loc[~labels.index.isin(
            train_idx)].values.T[0]

        self.dataset = dataset.loc[train_idx].copy()
        self.labels = labels.loc[train_idx].values.T[0]

    @staticmethod
    def gini_impurity(y):
        """Calcul l'impureté de Gini

        Args:
            y (list): Liste des différents labels

        Returns:
            float: L'impureté de Gini (entre 0 et 1)
        """

        l, count = np.unique(y, return_counts=True)
        prob = count / len(y)
        return 1 - np.sum(np.power(prob, 2))

    @staticmethod
    def gini_gain(col, y, thresholds, total_gini):
        """_summary_

        Args:
            col (str): Nom de la colonne associé au seuil (threshold)
            y (list): Liste des differents labels
            thresholds (float): Seuil de séparation du noeud de l'arbre associe a une colonne
            total_gini (float): Gain de Gini total du noeud courant avant la separation en 2 noeuds enfants

        Returns:
            _type_: Le gain de Gini
        """

        threshold = thresholds[col.name][0]
        i_l = np.where(col <= threshold)[0]
        i_r = np.where(col > threshold)[0]

        l_gini = Client.gini_impurity(y[i_l])
        r_gini = Client.gini_impurity(y[i_r])
        sum_gini = total_gini - (len(i_l) / len(y)) * \
            l_gini - (len(i_r) / len(y)) * r_gini

        return sum_gini
