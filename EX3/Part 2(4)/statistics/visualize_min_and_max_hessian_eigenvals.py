import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    neural_net_depths = [2, 3, 4]
    for N in neural_net_depths:
        min_eigenvals = np.load("statistics/results/depth_%d/min_eigenval.npy" %N)
        max_eigenvals = np.load("statistics/results/depth_%d/max_eigenval.npy" %N)

        x_axis = list(range(5, len(min_eigenvals)))
        fig, ax = plt.subplots()
        ax.plot(x_axis, min_eigenvals[5:])
        ax.plot(x_axis, max_eigenvals[5:])
        ax.set(xlabel='epochs', ylabel='eigenvalue', title='Hessian\'s Eigenvalue\nDepth: %d'%N)
        plt.legend(["min_eigenval", "max_eigenval"])
        ax.grid()

        fig.savefig("statistics/results/depth_%d/min_and_max_hessian_eigenvals.png" %N)
        plt.close(fig)