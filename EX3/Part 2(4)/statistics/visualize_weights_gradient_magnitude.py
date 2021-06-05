import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    neural_net_depths = [2, 3, 4]
    for N in neural_net_depths:
        weights_gradient_magnitude = np.load("statistics/results/depth_%d/train_weights_gradient_magnitude.npy" %N)

        x_axis = list(range(5, len(weights_gradient_magnitude)))
        fig, ax = plt.subplots()
        ax.plot(x_axis, weights_gradient_magnitude[5:])
        ax.set(xlabel='epochs', ylabel='magnitude', title='Weights\' Gradient Magnitudes\nDepth: %d'%N)
        ax.grid()

        fig.savefig("statistics/results/depth_%d/weights_gradient_magnitude.png" %N)
        plt.close(fig)