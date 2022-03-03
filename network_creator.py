import sys
import numpy as np
import pandas as pd

from master import Master
from serverManager import ServerManager


class NetworkCreator:
    def __init__(self, dataset, labels) -> None:
        self.dataset = dataset
        self.labels = labels

    def split_dataset(self, server_manager, repartition=None):
        """ Separe les donnees et les transmets au client
            (seulement utile dans un contexte de tests)
        """

        n_clients = len(server_manager.clients)

        if repartition and n_clients == 2:
            idx = np.random.choice(len(self.dataset) - 1, replace=False, size=int(len(self.dataset) * repartition))
            train_dataset1 = self.dataset.loc[idx]
            train_labels1 = self.labels.loc[idx]
            train_dataset2 = self.dataset.loc[~self.dataset.index.isin(idx)]
            train_labels2 = self.labels.loc[~self.labels.index.isin(idx)]

            server_manager.post([{'dataset': train_dataset1.to_dict(), 'labels': train_labels1.to_dict()},
                                 {'dataset': train_dataset2.to_dict(), 'labels': train_labels2.to_dict()}], 'dataset')

        else:
            # Separe les donnees equitablement entre les n_clients
            # La separation se fait en ordre (les x premieres donnees vont
            # au premier clients, les x prochaines au deuxieme, etc.)
            step = int(len(self.labels) / n_clients)
            stop = len(self.labels)

            train_datasets = np.split(
                self.dataset, [x for x in range(step, stop, step)])
            train_labels = np.split(
                self.labels, [x for x in range(step, stop, step)])

            server_manager.post([{'dataset': train_datasets[i].to_dict(), 'labels': train_labels[i].to_dict()} for i in
                                 range(n_clients)], 'dataset')


def split(dataset, labels):
    train_idx = np.random.choice(
        len(dataset) - 1, replace=False, size=int(len(dataset) * 0.8))

    test_dataset = dataset.loc[~dataset.index.isin(
        train_idx)]
    test_labels = labels.loc[~labels.index.isin(train_idx)]

    dataset = dataset.loc[train_idx]
    labels = labels.loc[train_idx]

    return dataset, labels, test_dataset, test_labels


def main():
    if len(sys.argv) < 5:
        print("Usage: python network_creator.py file_path n_clients repartition labels_column\n")
        print("\t file_path: Chemin du fichier csv a lire")
        print("\t n_clients: nombre de clients a utiliser (maximum de 10)")
        print("\t repartition: repartition des donnees entre les clients, si on utilise deux clients (0.1 a 0.9)")
        print("\t labels_column: Colonne cible")
        print(" exemple: python3 .\\network_creator.py .\\BCWdata.csv 2 0.5 diagnosis\n")
        return

    file_path = sys.argv[1]
    n_clients = int(sys.argv[2])
    repartition = float(sys.argv[3])
    labels_column = sys.argv[4]

    df = pd.read_csv(file_path)

    labels = df[labels_column]
    df.drop([labels_column], axis=1, inplace=True)
    network_creator = NetworkCreator(df, labels)

    server_manager = ServerManager(['http://localhost:50{}'.format(str(x).zfill(2)) for x in range(1, n_clients + 1)])
    network_creator.split_dataset(server_manager, repartition)

    # A valider
    # centralise
    master = Master(server_manager)
    dataset,labels,test_dataset,test_labels = split(df,labels)
    master.train(type="rf",network=None,distribution="centralised",n=100,depth=300,dataset=dataset,labels=labels)
    
    print(master.test(type="rf",network=None,distribution="centralised",test_dataset=test_dataset,test_labels=test_labels.values))
    
    master.train(type="rf",network=None,distribution="federated",n=10,depth=15)
    
    print("Centralise")
    print(master.test(type="rf", network=None, distribution="centralised", test_dataset=test_dataset,
                      test_labels=test_labels.values))

    print("localise")
    print(master.test(type="rf", network=None, distribution="localised"))

    print("federe")
    print(master.test(type="rf", network=None, distribution="federated"))


if __name__ == '__main__':
    main()
