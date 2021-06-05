import matplotlib.pyplot as plt
import numpy as np

import dataset.config as config

if __name__ == "__main__":
    data = np.load(config.DATASET_PATH)
    data_x, data_y = data

    plt.title("Our Data")
    plt.plot(data_x, data_y, 'o')
    plt.show()
