import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    gd_end_to_end_matrices_N_2 = np.load("statistics/results/depth_2/GD_end_to_end_matricies.npy")
    gd_end_to_end_matrices_N_3 = np.load("statistics/results/depth_3/GD_end_to_end_matricies.npy")
    e2ed_end_to_end_matrices_N_2 = np.load("statistics/results/depth_2/e2ed_end_to_end_matricies.npy")
    e2ed_end_to_end_matrices_N_3 = np.load("statistics/results/depth_3/e2ed_end_to_end_matricies.npy")

    # matrices plot
    x_axis = list(range(len(gd_end_to_end_matrices_N_2)))
    fig, ax = plt.subplots()
    ax.plot(x_axis, gd_end_to_end_matrices_N_2)
    ax.plot(x_axis, gd_end_to_end_matrices_N_3)
    ax.plot(x_axis, e2ed_end_to_end_matrices_N_2)
    ax.plot(x_axis, e2ed_end_to_end_matrices_N_3)
    ax.set(xlabel='epochs', ylabel='value', title='end to end matrix per epoch')
    ax.grid()
    plt.legend(["GD: N=2", "GD: N=3", "E2ED: N=2", "E2ED: N=3"])

    fig.savefig("statistics/results/end_to_end_matrix.png")
    plt.close(fig)