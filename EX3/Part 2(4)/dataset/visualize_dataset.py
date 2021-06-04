import matplotlib.pyplot as plt
import numpy as np

DATASET_PATH = "dataset/data.npy"

if __name__ == "__main__":
    data = np.load(DATASET_PATH)
    data_x, data_y = data

    plt.title("Our Data")
    plt.plot(data_x, data_y, 'o')
    plt.show()
