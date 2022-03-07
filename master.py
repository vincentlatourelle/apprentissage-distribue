from sklearn.ensemble import RandomForestClassifier
from RandomForest.master.FederatedRandomForest import FederatedRandomForest
import numpy as np


class Master:
    def __init__(self, server_manager) -> None:
        self.server_manager = server_manager
        self.frf = FederatedRandomForest(server_manager)
        self.rf = None

    def train(self, type, network, distribution, **kwargs):
        if type == "rf" and distribution == "federated":
            self.frf.train(kwargs['n'], kwargs['depth'])

        if type == "rf" and distribution == "localised":
            pass

        elif type == "rf" and distribution == "centralised":
            self.rf = RandomForestClassifier(n_estimators=kwargs['n'], max_depth=kwargs['depth'])
            self.rf.fit(kwargs['dataset'].values, kwargs['labels'])

    def test(self, type, network, distribution, test_dataset=None, test_labels=None):
        if type == "rf" and distribution == "federated":
            response = self.server_manager.get({}, 'federated-accuracy')
            n = [x['n'] for x in response]
            acc = [x['accuracy'] for x in response]
            return {'accuracy': sum([n[x] * acc[x] for x in range(len(n))]) / sum(n),
                    'min': min(acc),
                    'max': max(acc),
                    'var': np.var(acc)}

        elif type == "rf" and distribution == "localised":
            response = self.server_manager.get({}, 'local-accuracy')
            n = [x['n'] for x in response]
            acc = [x['accuracy'] for x in response]
            
            return {'accuracy': sum([n[x] * acc[x] for x in range(len(n))]) / sum(n),
                    'min': min(acc), 
                    'max': max(acc), 
                    'var': np.var(acc)}

        elif type == "rf" and distribution == "centralised":
            res = self.rf.predict(test_dataset)
            return {'accuracy': 1 - sum([int(value != test_labels[x]) for x, value in enumerate(res)]) / len(test_labels)}
