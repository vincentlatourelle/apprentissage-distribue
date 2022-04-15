import pandas as pd
from master import Master
from network_creator import NetworkCreator, print_results, split
from serverManager import ServerManager


def run_test(n_clients, repartition, df, labels, network_creator):
    server_manager = ServerManager(
        ['http://localhost:5{}'.format(str(x).zfill(3)) for x in range(1, n_clients + 1)])

    centralise = []
    federated = []
    localised = []

    
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
                    distribution="federated", n=10, depth=5)

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

def main():


    file_path = "BCWdata.csv"
    n_clients = 2
    labels_column = "diagnosis"

    df = pd.read_csv(file_path)

    labels = df[labels_column]
    df.drop([labels_column], axis=1, inplace=True)
    network_creator = NetworkCreator(df, labels)


    run_test(n_clients, 0.5, df, labels, network_creator)
        
if __name__ == "__main__":
    main()