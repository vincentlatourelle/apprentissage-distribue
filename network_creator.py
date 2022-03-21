import sys
import numpy as np
import pandas as pd

from master import Master
from serverManager import ServerManager
import time


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
            train_dataset = self.dataset.copy()
            train_dataset['target'] = self.labels

            train_dataset1 = train_dataset.groupby('target', group_keys=False).apply(
                lambda x: x.sample(frac=repartition))
            train_dataset2 = train_dataset.loc[~train_dataset.index.isin(
                train_dataset1.index)]

            train_labels1 = train_dataset1['target']
            train_labels2 = train_dataset2['target']
            train_dataset1 = train_dataset1.drop(['target'], axis=1)
            train_dataset2 = train_dataset2.drop(['target'], axis=1)

            server_manager.post([{'dataset': train_dataset1.to_dict(), 'labels': train_labels1.to_dict()},
                                 {'dataset': train_dataset2.to_dict(), 'labels': train_labels2.to_dict()}], 'dataset')

        else:
            train_datasets = []
            train_labels = []

            train_dataset = self.dataset.copy()
            train_dataset['target'] = self.labels

            for i in range(n_clients):
                train_dataset1 = train_dataset.groupby('target', group_keys=False).apply(
                    lambda x: x.sample(frac=1 / (n_clients - i)))

                train_labels.append(train_dataset1['target'])
                train_datasets.append(train_dataset1.drop(['target'], axis=1))

                train_dataset = train_dataset.loc[~train_dataset.index.isin(
                    train_dataset1.index)]

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
        print("\t n_clients: nombre de clients a utiliser (maximum de 10, 0 pour l'ensemble des tests)")
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

    if n_clients == 0:
        run_all_tests(df, labels, network_creator)
    else:
        run_test(n_clients, repartition, df, labels, network_creator)


def run_test(n_clients, repartition, df, labels, network_creator, df_results=None):
    server_manager = ServerManager(
        ['http://localhost:50{}'.format(str(x).zfill(2)) for x in range(1, n_clients + 1)])

    centralise = []
    federated = []
    localised = []

    start_time = time.time()

    for k in range(10):
        network_creator.split_dataset(server_manager, repartition)

        master = Master(server_manager)
        master.train(type="rf",
                     distribution="localised")

        print("Entrainement local")
        print(master.test(type="rf",
                          distribution="local-federated"))

        dataset, n_labels, test_dataset, test_labels = split(df, labels)
        master.train(type="rf", distribution="centralised",
                     n=100, depth=300, dataset=dataset, labels=n_labels)

        master.train(type="rf",
                     distribution="federated", n=10, depth=15)

        print("Centralise")
        res_centralise = master.test(type="rf", distribution="centralised", test_dataset=test_dataset,
                                     test_labels=test_labels.values)
        print(res_centralise)
        centralise.append(res_centralise['accuracy'])

        print("localise")
        res_local = master.test(type="rf",
                                distribution="localised")
        print(res_local)
        localised.append(res_local)

        print("federe")
        res_feder = master.test(type="rf",
                                distribution="federated")
        print(res_feder)
        federated.append(res_feder)

        print("####################################")

    if df_results is not None:
        df_results = add_result_to_df(df_results, centralise, federated, localised, n_clients, repartition,
                                      time.time() - start_time)
    print_results(centralise, federated, localised, time.time() - start_time)

    return df_results


def run_all_tests(df, labels, network_creator):
    df_results = pd.DataFrame(
        columns=["n_clients", "repartition", "federated_accuracy", "federated_min", "federated_max",
                 "federated_var", "local_accuracy", "local_min", "local_max", "local_var", "execution_time"])

    n_clients = [2, 4, 6, 8, 10]
    repartitions = [0.9, 0.8, 0.7, 0.8, 0.5]

    for repartition in repartitions:
        df_results = run_test(2, repartition, df, labels,
                              network_creator, df_results)

    for n in n_clients:
        df_results = run_test(n, 0.5, df, labels, network_creator, df_results)

    df_results.to_csv('./res.csv', float_format='%f')


def add_result_to_df(df, centralise, federated, localised, n_client, repartition, t_execution):
    df2 = {
        "n_clients": n_client,
        "repartition": repartition,
        "federated_accuracy": np.mean([config['accuracy'] for config in federated]),
        "federated_min": np.mean([config['min'] for config in federated]),
        "federated_max": np.mean([config['max'] for config in federated]),
        "federated_var": np.mean([config['var'] for config in federated]),
        "local_accuracy": np.mean([config['accuracy'] for config in localised]),
        "local_min": np.mean([config['min'] for config in localised]),
        "local_max": np.mean([config['max'] for config in localised]),
        "local_var": np.mean([config['var'] for config in localised]),
        "centralise_accuracy": np.mean(centralise),
        "execution_time": t_execution
    }
    return df.append(df2, ignore_index=True)


def print_results(centralise, federated, localised, t_execution):
    print("Accuracy:")
    print(f'Centralise: {np.mean(centralise)}')
    acc_local = np.mean([config['accuracy'] for config in localised])
    print(f'Locale: {acc_local}')
    acc_feder = np.mean([config['accuracy'] for config in federated])
    print(f'federe: {acc_feder}')

    print("Min:")
    stat_local = np.mean([config['min'] for config in localised])
    print(f'Locale: {stat_local}')
    stat_feder = np.mean([config['min'] for config in federated])
    print(f'federe: {stat_feder}')

    print("Max:")
    stat_local = np.mean([config['max'] for config in localised])
    print(f'Locale: {stat_local}')
    stat_feder = np.mean([config['max'] for config in federated])
    print(f'federe: {stat_feder}')

    print("Variance:")
    stat_local = np.mean([config['var'] for config in localised])
    print(f'Locale: {stat_local}')
    stat_feder = np.mean([config['var'] for config in federated])
    print(f'federe: {stat_feder}')

    print(f'Temps d\'execution (s): {t_execution}')


if __name__ == '__main__':
    main()
