from master import Master
import pandas as pd

def main():
    df = pd.read_csv("../../BCWdata.csv")

    labels = df['diagnosis']
    df.drop(["id",'diagnosis', 'Unnamed: 32'],axis=1, inplace=True)

    master = Master(df, labels)
    master.split_dataset(2)

if __name__ == "__main__":
    main()