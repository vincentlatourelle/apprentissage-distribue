

from RandomForest.master.federatedRandomForest import FederatedRandomForest


class Master:
    def __init__(self, server_manager) -> None:
        self.frf = FederatedRandomForest(server_manager)
        
        
    def train(type,network,distribution,**args):
        pass