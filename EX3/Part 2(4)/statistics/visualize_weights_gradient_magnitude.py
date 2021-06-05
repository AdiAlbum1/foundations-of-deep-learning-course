import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    weights_gradient_magnitude_N_2 = np.load("statistics/results/depth_2/train_weights_gradient_magnitude.npy")
    weights_gradient_magnitude_N_3 = np.load("statistics/results/depth_3/train_weights_gradient_magnitude.npy")
    weights_gradient_magnitude_N_4 = np.load("statistics/results/depth_4/train_weights_gradient_magnitude.npy")

    x_axis = list(range(len(weights_gradient_magnitude_N_2)))
    fig, ax = plt.subplots()
    ax.plot(x_axis, weights_gradient_magnitude_N_2)
    ax.plot(x_axis, weights_gradient_magnitude_N_3)
    ax.plot(x_axis, weights_gradient_magnitude_N_4)
    ax.set(xlabel='epochs', ylabel='magnitude', title='Weights\' Gradient Magnitudes')
    ax.grid()
    plt.legend(["N=2", "N=3", "N=4"])

    fig.savefig("statistics/results/weights_gradient_magnitude.png")
    plt.close(fig)