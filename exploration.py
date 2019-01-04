import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def explore_dataset(df):
    print("\nExploring the Dataset..\n")

    # Dataframe shape
    print(df.shape[1], " columns.")
    print(df.shape[0], " observations.\n")
    # Dataframe datatypes
    datatypes = df.dtypes
    #print(datatypes)

    #print(df['Soil_Type15'].sum())
    # Supprimer une colonne non pertinente :
    #df_train = df_train.drop(['Soil_Type7', 'Soil_Type15'], axis=1)


def plot_correlation(df):
    correlation_matrix = df.corr()
    f, ax = plt.subplots(figsize=(10, 8))
    sns_plot = sns.heatmap(correlation_matrix, vmax=0.8, square=True)
    fig = sns_plot.get_figure()
    fig.savefig("correlation_matrix.png")
    plt.show()


def main():
    print("Reading data")

    train_df = pd.read_csv("all-data/train-set.csv")
    explore_dataset(train_df)


    test_df = pd.read_csv("all-data/test-set.csv")
    explore_dataset(test_df)

    # Remove ID column
    train_df = train_df.iloc[:, 1:]
    #df_test = test_df.iloc[:, 1:]

    # Correlation matrix for 10 first columns

    # Plotting
    #plot_correlation(train_df.iloc[:, :10])


if __name__ == '__main__':
    main()