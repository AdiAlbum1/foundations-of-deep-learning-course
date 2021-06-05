import numpy as np
import math
import dataset.config as config


if __name__ == "__main__":
    input_data = np.random.uniform(low=-1.2, high=1.2, size=config.DATASET_SIZE)
    output_noise = np.random.normal(loc=0.0, scale=0.2, size=config.DATASET_SIZE)
    output_without_noise = np.array([math.tan(xi) for xi in input_data])
    output_data = np.add(output_without_noise, output_noise)

    data = np.vstack((input_data, output_data))
    np.save(config.DATASET_PATH.split(".")[0], data)
