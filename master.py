from sklearn.ensemble import RandomForestClassifier
from RandomForest.master.FederatedRandomForest import FederatedRandomForest
import numpy as np


class Master:
    """Point d'entre du framework pour appeler les differents algorithme d'entrainement distribue
       (seulement RandomForest pour l'instant)
    """

    def __init__(self, server_manager) -> None:
        self.server_manager = server_manager
        self.frf = FederatedRandomForest(server_manager)
        self.rf = None
        self.local_rf = None

    def cross_validation(self, k, type, distribution, **kwargs):
        """Fait une validation croisee pour determiner les meilleurs hyperparametres a utiliser
           selon le modele utilise

        Args:
            k (int): Nombre de sous-ensemble de notre dataset et le nombre d'iterations a faire
            type (str): Modele a executer
            distribution (str): Type de distribution: centralise, localise ou federe
        """
        if type == "rf" and distribution == "federated":
            list_n = kwargs['n']
            list_depth = kwargs['depth']
            results = {}
            for n in list_n:
                for depth in list_depth:
                    mean_accuracy = 0
                    for i in range(k):
                        # Il faut tester et entrainer avec un ensemble de validation
                        self.server_manager.get(None,'rf/set-validation')
                        self.frf.train(n, depth)
                        res = self.test("rf","federated")
                        mean_accuracy += res['accuracy']
                    results[(n,depth)] = mean_accuracy/k
                        
            
            self.server_manager.get(None,'rf/unset-validation')
            return max(results,key=results.get)

    def train(self, type, distribution, **kwargs):
        """ Entraine un model d'apprentissage 

        Args:
            type (str): type d'algorithme d'apprentissage (rf,)
            distribution (str): type de distribution (federated, localised, centralised)
        """

        if type == "rf" and distribution == "federated":
            n = kwargs['n']
            depth = kwargs['depth']
            
            self.frf.train(n, depth)

        if type == "rf" and distribution == "localised":
            self.local_rf = self.server_manager.get_models(
                None, 'rf/local-model')

        elif type == "rf" and distribution == "centralised":
            self.rf = RandomForestClassifier(
                random_state=1234,
                n_estimators=kwargs['n'], max_depth=kwargs['depth'])
            self.rf.fit(kwargs['dataset'].values, kwargs['labels'])

    def test(self, type, distribution, test_dataset=None, test_labels=None):
        """test l'accuracy d'un model   

        Args:
            type (str): type de model (rf,)
            distribution (str): type de distribution (federated, localised, centralised)
            test_dataset (pd.DataFrame, optional): Dataset de test (dans le cas centralise). Defaults to None.
            test_labels (pd.Series, optional): Labels de test (dans le cas centralise). Defaults to None.

        Returns:
            dict: accuracy moyenne, min et max, ainsi que la variance
        """

        if type == "rf" and distribution == "federated":
            self.frf.send_forest()
            response = self.server_manager.get(None, 'rf/federated-accuracy')
            n = [x['n'] for x in response]
            acc = [x['accuracy'] for x in response]
            return {'accuracy': sum([n[x] * acc[x] for x in range(len(n))]) / sum(n),
                    'min': min(acc),
                    'max': max(acc),
                    'var': np.var(acc)}

        # Entrainement localise et test federe
        elif type == "rf" and distribution == "local-federated":
            accuracy = []
            mins = []
            maxs = []
            var = []
            
            # Pour chaque modele entrainer localement, tester chez tous les clients
            for local_model in self.local_rf:
                self.server_manager.post_model('rf/random-forest', local_model)

                response = self.server_manager.get(None, 'rf/federated-accuracy')
                n = [x['n'] for x in response]
                acc = [x['accuracy'] for x in response]

                accuracy.append(sum([n[x] * acc[x]
                                for x in range(len(n))]) / sum(n))
                mins.append(min(acc))
                maxs.append(max(acc))
                var.append(np.var(acc))

            return {'accuracy': np.mean(accuracy),
                    'min': np.mean(mins),
                    'max': np.mean(maxs),
                    'var': np.mean(var)}

        elif type == "rf" and distribution == "localised":
            response = self.server_manager.get(None, 'rf/local-accuracy')
            n = [x['n'] for x in response]
            acc = [x['accuracy'] for x in response]

            return {'accuracy': sum([n[x] * acc[x] for x in range(len(n))]) / sum(n),
                    'min': min(acc),
                    'max': max(acc),
                    'var': np.var(acc)}

        elif type == "rf" and distribution == "centralised":
            res = self.rf.predict(test_dataset.values)
            return {'accuracy': 1 - sum([int(value != test_labels[x]) for x, value in enumerate(res)]) / len(test_labels)}
