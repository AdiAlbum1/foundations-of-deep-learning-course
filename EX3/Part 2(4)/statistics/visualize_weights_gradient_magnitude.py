import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    weights_gradient_magnitude = np.load("statistics/results/train_weights_gradient_magnitude.npy")

    x_axis = list(range(len(weights_gradient_magnitude)))
    fig, ax = plt.subplots()
    ax.plot(x_axis, weights_gradient_magnitude)
    ax.set(xlabel='epochs', ylabel='magnitude', title='weights gradient magnitude')
    ax.grid()

    fig.savefig("statistics/results/weights_gradient_magnitude.png")
    plt.close(fig)