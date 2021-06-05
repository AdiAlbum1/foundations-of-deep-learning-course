import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    min_eigenvals_N_2 = np.load("statistics/results/depth_2/min_eigenval.npy")
    min_eigenvals_N_3 = np.load("statistics/results/depth_3/min_eigenval.npy")
    min_eigenvals_N_4 = np.load("statistics/results/depth_4/min_eigenval.npy")

    max_eigenvals_N_2 = np.load("statistics/results/depth_2/max_eigenval.npy")
    max_eigenvals_N_3 = np.load("statistics/results/depth_3/max_eigenval.npy")
    max_eigenvals_N_4 = np.load("statistics/results/depth_4/max_eigenval.npy")

    # plot min eigenvals
    x_axis = list(range(len(min_eigenvals_N_2)))
    fig, ax = plt.subplots()
    ax.plot(x_axis, min_eigenvals_N_2)
    ax.plot(x_axis, min_eigenvals_N_3)
    ax.plot(x_axis, min_eigenvals_N_4)
    ax.set(xlabel='epochs', ylabel='eigenvalue', title='Hessian\'s Minimum Eigenvalue')
    plt.legend(["N=2", "N=3", "N=4"])
    ax.grid()

    fig.savefig("statistics/results/min_eigenvals.png")
    plt.close(fig)

    # plot max eigenvals
    x_axis = list(range(len(max_eigenvals_N_2)))
    fig, ax = plt.subplots()
    ax.plot(x_axis, max_eigenvals_N_2)
    ax.plot(x_axis, max_eigenvals_N_3)
    ax.plot(x_axis, max_eigenvals_N_4)
    ax.set(xlabel='epochs', ylabel='eigenvalue', title='Hessian\'s Maximum Eigenvalue')
    plt.legend(["N=2", "N=3", "N=4"])
    ax.grid()

    fig.savefig("statistics/results/max_eigenvals.png")
    plt.close(fig)