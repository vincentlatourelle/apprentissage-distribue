import numpy as np

class Client():
    def __init__(self, dataset = None) -> None:
        self.dataset = dataset
        self.forest = None
        self.labels = None
    
    def __bootstrap(self,x):
        idx = np.random.choice(len(x)-1, replace=True ,size =len(x))
        return x.iloc[idx]
    
    def get_best_threshold(self,features, splits, current_tree):
        
        # Separer les donnees en fonctions de l'arbre courant 
        
        # calcul de gini pour chaque feature
        
        # retourne l'attribut permetant d'avoir le meilleur "gain de gini", ainsi que le nombre 
        # donnees dans le dataset courant
        
        pass
    
    def get_leaf(self,current_tree):
        
        # Separer les donnees en fonctions de l'arbre courant 
        
        # retourner le nombre de valeurs perturbees pour chaque classe dans le dataset courant
        
        pass
    
    def set_new_forest(self,random_forest):
        self.forest = random_forest
    
    def get_local_accuracy(self):
        
        # Entrainer un modele de randomForest (scikit-learn) et retourner l'accuracy
        
        pass
    
    def get_thresholds(self,features):
        
        values = []
        for f in features:
            col = self.dataset[f]
            min = col.min()
            max = col.max()
            values.append(np.random.default_rng().uniform(low=min,high=max))
            
        return values
    
    @staticmethod
    def gini_impurity(y):
        l, count = np.unique(y,return_counts=True)
        prob = count/len(y)
        return 1 - np.sum(np.power(prob,2))