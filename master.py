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

    def k_fold_cross_validation(self,k,type,distribution, **kwargs):
        if type == "rf" and distribution == "federated":
            list_n = kwargs['n']
            list_depth = kwargs['depth']
            for n in list_n:
                for depth in list_depth:
                    for i in range(k):
                        ## Il faut tester et entrainer avec un ensemble de validation
                        # self.frf.train(n, depth)
                        pass
                        
                        

    def train(self, type, distribution, **kwargs):
        if type == "rf" and distribution == "federated":
            n = kwargs['n']
            depth = kwargs['depth']
            if isinstance(n, list) or isinstance(depth, list):
                if not isinstance(n,list): n=[n] 
                if not isinstance(depth,list): depth=[depth] 
                
                n, depth = self.k_fold_cross_validation(k=10,type=type,distribution=distribution,n=n,depth=depth)
                
            self.frf.train(n, depth)

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
