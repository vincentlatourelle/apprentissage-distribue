import io
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from RandomForest.master.FederatedRandomForest import FederatedRandomForest
import numpy as np


class Master:
    def __init__(self, server_manager) -> None:
        self.server_manager = server_manager
        self.frf = FederatedRandomForest(server_manager)
        self.rf = None
        self.local_rf = None

    def train(self, type, distribution, **kwargs):
        if type == "rf" and distribution == "federated":
            self.frf.train(kwargs['n'], kwargs['depth'])

        if type == "rf" and distribution == "localised":
            self.local_rf = self.server_manager.get_models({},'rf/local-model')
            
            
        elif type == "rf" and distribution == "centralised":
            self.rf = RandomForestClassifier(n_estimators=kwargs['n'], max_depth=kwargs['depth'])
            self.rf.fit(kwargs['dataset'].values, kwargs['labels'])

    def test(self, type, distribution, test_dataset=None, test_labels=None):
        if type == "rf" and distribution == "federated":
            self.frf.send_forest()
            response = self.server_manager.get({}, 'rf/federated-accuracy')
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
            for local_model in self.local_rf:
                self.server_manager.post_model('rf/random-forest', local_model)
                
                response = self.server_manager.get({}, 'rf/federated-accuracy')
                n = [x['n'] for x in response]
                acc = [x['accuracy'] for x in response]
                
                accuracy.append(sum([n[x] * acc[x] for x in range(len(n))]) / sum(n))
                mins.append(min(acc))
                maxs.append(max(acc))
                var.append(np.var(acc))
                
            return {'accuracy': np.mean(accuracy),
                        'min': np.mean(mins),
                        'max': np.mean(maxs),
                        'var': np.mean(var)}

        elif type == "rf" and distribution == "localised":
            response = self.server_manager.get({}, 'rf/local-accuracy')
            n = [x['n'] for x in response]
            acc = [x['accuracy'] for x in response]
            
            return {'accuracy': sum([n[x] * acc[x] for x in range(len(n))]) / sum(n),
                    'min': min(acc), 
                    'max': max(acc), 
                    'var': np.var(acc)}

        elif type == "rf" and distribution == "centralised":
            res = self.rf.predict(test_dataset.values)
            return {'accuracy': 1 - sum([int(value != test_labels[x]) for x, value in enumerate(res)]) / len(test_labels)}
