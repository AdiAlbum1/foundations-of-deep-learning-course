import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # Read data from CSV
    train_dataset_complete = pd.read_csv(r"./dataset/train.csv")
    test_dataset_complete = pd.read_csv(r"./dataset/test.csv")

    # Read 'open' column only
    train_dataset_processed = train_dataset_complete.iloc[:, 1:2].values
    test_dataset_processed = test_dataset_complete.iloc[:, 1:2].values

    # train_dataset_processed = np.reshape(train_dataset_processed, len(train_dataset_processed))

    x_1 = list(range(1, len(train_dataset_processed)+1))
    x_2 = list(range(len(train_dataset_processed)+1, len(train_dataset_processed) + len(test_dataset_processed) + 1))
    plt.title("Dataset")
    plt.plot(x_1, train_dataset_processed)
    plt.plot(x_2, test_dataset_processed)
    plt.legend(["train_data", "test_data"])
    plt.show()