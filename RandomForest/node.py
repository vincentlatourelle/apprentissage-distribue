import json

class Node:
    def __init__(self, feature=None, threshold=None, lNode=None, rNode=None, value=None) -> None:
        self.feature = feature
        self.threshold = threshold
        self.lNode = lNode
        self.rNode = rNode
        self.value = value
        
    def predict(self,x):
        if self.value != None:
            return self.value
        if x[self.feature] <= self.threshold:
            return self.lNode.predict(x)
        else:
            return self.rNode.predict(x)
        
    def get_custom_dict(self):
        if self.value != None:
            custom_dict = {
                "value" : self.value,  
            }
        elif self.feature is None:
            return {}
        else:
            custom_dict = {
                "feature" : self.feature,
                "threshold" : self.threshold,
                "rNode" : self.rNode.get_custom_dict(),
                "lNode" : self.lNode.get_custom_dict()    
            }
        return custom_dict
    
    def serialize(self):
        return self.get_custom_dict()
    
    @staticmethod
    def deserialize(tree_dict):
        if len(tree_dict) == 0:
            return
        
        new_tree = Node()
        for f in tree_dict:
            if f == "lNode":
                new_tree.lNode = Node.deserialize(tree_dict[f])
            elif f == "rNode":
                new_tree.rNode = Node.deserialize(tree_dict[f])
            else:
                new_tree.__dict__[f] = tree_dict[f]
                
        return new_tree
            