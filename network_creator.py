import sys
import numpy as np
import pandas as pd

from master import Master
from serverManager import ServerManager
import time

import matplotlib.pyplot as plt

class NetworkCreator:
    def __init__(self, dataset, labels) -> None:
        self.dataset = dataset
        self.labels = labels

    def split_dataset(self, server_manager, data_repartition=None, label_repartition = None, unequal_rep=False):
        """Separe le dataset entre les clients. S'il y a 2 clients, il y a possibilite 
        de preciser le pourcentage de donnees chez chaque client et la distribution des donnees chez le premier client.

        Args:
            server_manager (ServerManager): Le ServerManager pour envoyer les requêtes aux clients
            data_repartition (float, optional): Pourcentage des donnees allant chez le premier client. Defaults to None.
            label_repartition (dict, optional): Pourcentage des donnees de chacun des labels chez le client 1. Doit sommer a 1. Defaults to None.
        """        

        n_clients = len(server_manager.clients)
        
        if unequal_rep:
            train_datasets = []
            train_labels = []

            train_dataset = self.dataset.copy()
            train_dataset['target'] = self.labels

            
            for i in range(n_clients):

                frac = 1 if i == n_clients -1 else 0.5
                train_dataset1 = train_dataset.groupby('target', group_keys=False).apply(
                         lambda x: x.sample(frac=frac))

                train_labels.append(train_dataset1['target'])
                train_datasets.append(train_dataset1.drop(['target'], axis=1))

                train_dataset = train_dataset.loc[~train_dataset.index.isin(
                    train_dataset1.index)]
                
            for ds in train_datasets:
                print(len(ds))
                
            server_manager.post([{'dataset': train_datasets[i].to_dict(), 'labels': train_labels[i].to_dict()} 
                                    for i in range(n_clients)], 'dataset')  
            
            
        # Cas avec 2 clients (et repartition inegales)
        elif data_repartition and n_clients == 2:
            train_dataset = self.dataset.copy()
            train_dataset['target'] = self.labels

            # Regroupe par target et prend un echantillon de (repartition)% des donnees pour chaque target
            if label_repartition is not None:
                nb = len(train_dataset)*data_repartition
                nb_data_label = {}
                for k in label_repartition:
                    nb_data_label[k] = int(nb*label_repartition[k])
                
                
                train_dataset1 = pd.DataFrame()
                for i, label in enumerate(train_dataset['target'].unique()):
                    train_dataset1 = pd.concat([train_dataset1, train_dataset[train_dataset['target'] == label].sample(n=min(len(train_dataset[train_dataset['target'] == label]),nb_data_label[label]))])
            
            else:
                train_dataset1 = train_dataset.groupby('target', group_keys=False).apply(lambda x: x.sample(frac=data_repartition))
            
            
            # Prend les donnees qui n'ont pas ete utilisees dans le premier dataset
            train_dataset2 = train_dataset.loc[~train_dataset.index.isin(
                train_dataset1.index)]
            

            # Recupere le target des dataset et l'enleve de ceux-ci
            train_labels1 = train_dataset1['target']
            train_labels2 = train_dataset2['target']
            train_dataset1 = train_dataset1.drop(['target'], axis=1)
            train_dataset2 = train_dataset2.drop(['target'], axis=1)

            # Envoi les datasets au 2 clients
            server_manager.post([{'dataset': train_dataset1.to_dict(), 'labels': train_labels1.to_dict()},
                                 {'dataset': train_dataset2.to_dict(), 'labels': train_labels2.to_dict()}], 'dataset')

        # Cas avec plus de 2 clients (et repartitions egales)
        else:
            train_datasets = []
            train_labels = []

            train_dataset = self.dataset.copy()
            train_dataset['target'] = self.labels

            
            for i in range(n_clients):
                # Prend un echantillon (1/(nombre de clients restant))% par target
                train_dataset1 = train_dataset.groupby('target', group_keys=False).apply(
                    lambda x: x.sample(frac=1 / (n_clients - i)))

                train_labels.append(train_dataset1['target'])
                train_datasets.append(train_dataset1.drop(['target'], axis=1))

                train_dataset = train_dataset.loc[~train_dataset.index.isin(
                    train_dataset1.index)]

            server_manager.post([{'dataset': train_datasets[i].to_dict(), 'labels': train_labels[i].to_dict()} 
                                 for i in range(n_clients)], 'dataset')


def split(dataset, labels):
    """Separe les donnees/labels (cibles) en donnees/labels d'entrainement et de test

    Args:
        dataset (pd.DataFrame): Dataset des donnees
        labels (pd.DataFrame): Dataset des labels (cibles)

    Returns:
        tuple: tuple contenant les informations suivantes:
            dataset (pd.DataFrame): donnees d'entrainement
            labels (pd.DataFrame): labels d'entrainement
            test_dataset (pd.DataFrame): donnees de test
            test_labels (pd.DataFrame): labels de test

    """
    rng = np.random.default_rng(1234)    
    # Definit l'ensemble d'entrainement aleatoirement avec 80% des donnees du dataset
    train_idx = rng.choice(
        len(dataset) - 1, replace=False, size=int(len(dataset) * 0.8))

    # Definit l'ensemble de test et les cibles des 20% restants
    # (les donnees qui ne font pas partie de l'ensemble d'entrainement)
    test_dataset = dataset.loc[~dataset.index.isin(
        train_idx)]
    test_labels = labels.loc[~labels.index.isin(train_idx)]

    dataset = dataset.loc[train_idx]
    labels = labels.loc[train_idx]
    
    
    return dataset, labels, test_dataset, test_labels


def main():
    if len(sys.argv) < 6:
        print("Usage: python network_creator.py file_path n_clients repartition labels_column\n")
        print("\t file_path: Chemin du fichier csv a lire")
        print("\t n_clients: nombre de clients a utiliser (maximum de 20, 0 pour executer l'ensemble des tests)")
        print("\t repartition: repartition des donnees entre les clients, si on utilise deux clients (0.1 a 0.9)")
        print("\t labels_column: Colonne cible")
        print("\t cross_validation: 0=non ou 1=oui ")
        print(" exemple: python3 .\\network_creator.py .\\BCWdata.csv 2 0.5 diagnosis 0\n")
        return

    file_path = sys.argv[1]
    n_clients = int(sys.argv[2])
    repartition = float(sys.argv[3])
    labels_column = sys.argv[4]
    cross_validation = bool(int(sys.argv[5]))

    df = pd.read_csv(file_path)

    labels = df[labels_column]
    df.drop([labels_column], axis=1, inplace=True)
    network_creator = NetworkCreator(df, labels)

    if n_clients == 0:
        run_all_tests(df, labels, network_creator)
    else:
        run_test(n_clients, repartition, df, labels, network_creator, cross_validation=cross_validation)


def run_test(n_clients, repartition, df, labels, network_creator, df_results=None, label_repartition=None, unequal_rep=False, cross_validation=False):
    server_manager = ServerManager(
        ['http://localhost:5{}'.format(str(x).zfill(3)) for x in range(1, n_clients + 1)])

    centralise = []
    federated = []
    localised = []

    start_time = time.time()
    
    network_creator.split_dataset(server_manager, repartition, label_repartition,unequal_rep=unequal_rep)
    master = Master(server_manager)
    
    if cross_validation:
        list_n = [10,20,30]
        list_depth = [10,15,20,25]
        n, depth = master.cross_validation(6, "rf","federated",n=list_n,depth=list_depth)
    else:
        n=20
        depth=20
    
    for k in range(1):
        network_creator.split_dataset(server_manager, repartition, label_repartition,unequal_rep=unequal_rep)

        
        # master.train(type="rf",
        #              distribution="localised")

        # print("Entrainement local")
        # print(master.test(type="rf",
        #                   distribution="local-federated"))

        dataset, n_labels, test_dataset, test_labels = split(df, labels)
        master.train(type="rf", distribution="centralised",
                      n=100, depth=300, dataset=dataset, labels=n_labels)

        master.train(type="rf",
                     distribution="federated", n=n, depth=depth)

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
        df_results = add_result_to_df(df_results, centralise, federated, localised, n_clients, repartition,label_repartition, 
                                      time.time() - start_time)
    print_results(centralise, federated, localised, time.time() - start_time)

    return df_results


def run_all_tests(df, labels, network_creator):
    columns=["n_clients", "repartition","label_repartition", "federated_accuracy", "federated_min", "federated_max",
                 "federated_var", "local_accuracy", "local_min", "local_max", "local_var", "execution_time"]
    df_results = pd.DataFrame(columns=columns)

    n_clients = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]#,40,60,80,100, 120, 140]
    n_clients2 = [2,3,4,5]
    data_repartitions = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    labels_repartitions = [None, 
                           {'B':0.85, 'M':0.15},
                           {'B':0.75, 'M':0.25},
                           {'B':0.60, 'M':0.40},
                           {'B':0.50, 'M':0.50},
                           {'B':0.40, 'M':0.60},
                           {'B':0.25, 'M':0.75}, 
                           {'B':0.15, 'M':0.85}, 
                           ]
    
    # Roule les tests pour chacune des repartions entre 2 clients (0.9-0.1, 0.8-0.2, ...)
    df_results_data_repartition = pd.DataFrame(columns=columns)
    # for data_repartition in data_repartitions:
    #     df_results_data_repartition = run_test(2, data_repartition, df, labels,
    #                         network_creator, df_results_data_repartition)
        
    df_results_labels_repartition = pd.DataFrame(columns=columns)
    # for label_repartition in labels_repartitions:
    #     df_results_labels_repartition = run_test(2, 0.5, df, labels,
    #                         network_creator, df_results_labels_repartition, label_repartition)

    # Roule les tests pour chaque nombre de clients de n_clients
    df_results_nb_clients = pd.DataFrame(columns=columns)
    df_results_unequal = pd.DataFrame(columns=columns)
    for n in n_clients:
        df_results_nb_clients = run_test(n, 0.5, df, labels, network_creator, df_results_nb_clients)
        #df_results_unequal = run_test(n, 0.5, df, labels, network_creator, df_results_unequal, unequal_rep=True)
        
        
    
    #print_results_scatter(df_results_unequal, 'n_clients')

    print_results_scatter(df_results_nb_clients, 'n_clients')
    # # print_results_scatter(df_results_labels_repartition, 'label_repartition')
    # print_results_scatter(df_results_data_repartition, 'repartition')
    
    df_results = pd.concat([df_results, df_results_nb_clients, df_results_data_repartition, df_results_labels_repartition])
    df_results.to_csv('./res2.csv', float_format='%f')


def add_result_to_df(df, centralise, federated, localised, n_client, repartition,label_repartition, t_execution):
    df2 = {
        "n_clients": n_client,
        "repartition": repartition,
        "label_repartition":label_repartition,
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

def print_results_scatter(df_results,x_columns):
    plt.plot(df_results[x_columns].values, df_results['federated_accuracy'].values, label='federated', color='b')
    plt.plot(df_results[x_columns].values, df_results['federated_min'].values, label='federated_min', color='b', linestyle=":")
    plt.plot(df_results[x_columns].values, df_results['federated_max'].values, label='federated_max', color='b', linestyle="-.")
    
    plt.plot(df_results[x_columns].values, df_results['local_accuracy'].values, label='local', color='r')
    plt.plot(df_results[x_columns].values, df_results['local_min'].values, label='local_min', color='r', linestyle=":")
    plt.plot(df_results[x_columns].values, df_results['local_max'].values, label='local_max', color='r', linestyle="-.")
    
    
    plt.plot(df_results[x_columns].values, df_results['centralise_accuracy'].values, label='centralised', color='g')
    plt.xlabel(x_columns)
    plt.ylabel('Accuracy')
    # plt.xticks(range(len(df_results[x_columns].values)), df_results[x_columns].values)
    plt.legend()
    plt.show()


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