from serverManager import ServerManager
from master import Master
import pandas as pd

def main():
    df = pd.read_csv("../../BCWdata.csv")

    labels = df['diagnosis']
    df.drop(["id",'diagnosis', 'Unnamed: 32'],axis=1, inplace=True)
    
    server_manager = ServerManager(["http://localhost:5001","http://localhost:5002"])
    master = Master(df, labels, server_manager)
    master.split_dataset()
    
    master.train()

if __name__ == "__main__":
    main()