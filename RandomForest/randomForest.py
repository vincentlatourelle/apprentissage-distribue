import numpy as np


class RandomForest:
    def __init__(self) -> None:
        self.forest = []
        
    def add(self,node):
        self.forest.append(node)
        
    def predict(self,x):
        result = [tree.predict(x) for tree in self.forest]
        result, count = np.unique(result, return_counts=True) 
        return result[np.argmax(count)]