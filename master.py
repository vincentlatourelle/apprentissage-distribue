

from sklearn.ensemble import ExtraTreesClassifier
from RandomForest.master.federatedRandomForest import FederatedRandomForest


class Master:
    def __init__(self, server_manager) -> None:
        self.frf = FederatedRandomForest(server_manager)
        self.rf = ExtraTreesClassifier()
        
        
    def train(self,type,network,distribution,**kwargs):
        
        if type=="rf" and distribution=="federated":
            self.frf.train(kwargs['n'], kwargs['depth'])
        elif type=="rf":
            # self.rf.fit()
            pass

