class Client():
    def __init__(self, dataset = None) -> None:
        self.dataset = dataset
        self.forest = None
    
    def __bootstrap(self):
        pass
    
    def get_best_split(self,features, splits, current_tree):
        pass
    
    def get_leaf(self,current_tree):
        pass
    
    def set_new_forest(self,random_forest):
        pass
    
    def get_local_accuracy(self):
        pass