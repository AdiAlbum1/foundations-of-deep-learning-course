import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    min_eigenvals = np.load("statistics/results/min_eigenval.npy")
    max_eigenvals = np.load("statistics/results/max_eigenval.npy")

    x_axis = list(range(len(min_eigenvals)))
    fig, ax = plt.subplots()
    ax.plot(x_axis, min_eigenvals)
    ax.plot(x_axis, max_eigenvals)
    ax.set(xlabel='epochs', ylabel='eigenvalue', title='Hessian\'s eigenvalue per epoch')
    plt.legend(["min_eigenval", "max_eigenval"])
    ax.grid()

    fig.savefig("statistics/results/min_and_max_hessian_eigenvals.png")
    plt.close(fig)