import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

from master import Master
from serverManager import ServerManager


class NetworkCreator:
    def __init__(self, dataset, labels, server_manager: ServerManager) -> None:
        self.dataset = dataset
        self.labels = labels
        self.test_dataset = None
        self.test_labels = None
        self.server_manager = server_manager

    def split_dataset(self):
        """ Separe les donnees et les transmets au client
            (seulement utile dans un contexte de tests)
        """

        n_clients = len(self.server_manager.clients)

        # Selectionne 80% des donnees pour l'entrainement et 20% pour les tests
        train_idx = np.random.choice(
            len(self.dataset) - 1, replace=False, size=int(len(self.dataset) * 0.8))

        self.test_dataset = self.dataset.loc[~self.dataset.index.isin(
            train_idx)]
        self.test_labels = self.labels.loc[~self.labels.index.isin(train_idx)]

        self.dataset = self.dataset.loc[train_idx]
        self.labels = self.labels.loc[train_idx]

        # Separe les donnees equitablement entre les n_clients
        # La separation se fait en ordre (les x premieres donnees vont
        # au premier clients, les x prochaines au deuxieme, etc.)
        step = int(len(self.labels) / n_clients)
        stop = len(self.labels)

        train_datasets = np.split(
            self.dataset, [x for x in range(step, stop, step)])
        train_labels = np.split(
            self.labels, [x for x in range(step, stop, step)])

        self.server_manager.post([{'dataset':train_datasets[i].to_dict(), 'labels':train_labels[i].to_dict()} for i in range(n_clients)],'dataset')

    # def get_federated_accuracy(self,forest):
    #     res = [forest.predict(row) for index, row in self.test_dataset.iterrows()]
    #     return (1 - sum([int(value != self.test_labels.values[x]) for x, value in enumerate(res)]) / len(self.test_labels))

    # def get_centralised_accuracy(self):
    #     dt = ExtraTreesClassifier()
    #     dt.fit(self.dataset.values, self.labels.values)
    #     res = dt.predict(self.test_dataset)
    #     return (1 - sum([int(value != self.test_labels.values[x]) for x, value in enumerate(res)]) / len(self.test_labels))

    # def get_local_accuracy(self):
    #     return (1 - np.mean(self.server_manager.get_clients_local_accuracy(self.test_dataset, self.test_labels)))
        
        
def main():
    df = pd.read_csv("./BCWdata.csv")

    labels = df['diagnosis']
    df.drop(["id", 'diagnosis', 'Unnamed: 32'], axis=1, inplace=True)

    server_manager = ServerManager(
        ["http://localhost:5001", "http://localhost:5002", "http://localhost:5003", "http://localhost:5004"])
    network_creator = NetworkCreator(df, labels, server_manager)
    network_creator.split_dataset()

    # A valider
    master = Master(server_manager)
    master.train(type="rf",network=None,distribution="federated",n=5,depth=3)
    
    # print(network_creator.get_local_accuracy())
    # print(network_creator.get_centralised_accuracy())
    # print(network_creator.get_federated_accuracy(master.forest))

if __name__ == '__main__':
    main()
