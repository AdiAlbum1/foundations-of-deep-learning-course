import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    gd_train_stats_N_2 = np.load("statistics/results/depth_2/GD_train_stats.npy")
    gd_train_stats_N_3 = np.load("statistics/results/depth_3/GD_train_stats.npy")
    e2ed_train_stats_N_2 = np.load("statistics/results/depth_2/end_to_end_dynamics_train_stats.npy")
    e2ed_train_stats_N_3 = np.load("statistics/results/depth_3/end_to_end_dynamics_train_stats.npy")

    gd_train_loss_N_2_per_epoch, gd_test_loss_N_2_per_epoch = gd_train_stats_N_2
    gd_train_loss_N_3_per_epoch, gd_test_loss_N_3_per_epoch = gd_train_stats_N_3
    e2ed_train_loss_N_2_per_epoch, e2ed_test_loss_N_2_per_epoch = e2ed_train_stats_N_2
    e2ed_train_loss_N_3_per_epoch, e2ed_test_loss_N_3_per_epoch = e2ed_train_stats_N_3

    # Train loss plot
    x_axis = list(range(len(gd_train_loss_N_2_per_epoch)))
    fig, ax = plt.subplots()
    ax.plot(x_axis, gd_train_loss_N_2_per_epoch)
    ax.plot(x_axis, gd_train_loss_N_3_per_epoch)
    ax.plot(x_axis, e2ed_train_loss_N_2_per_epoch)
    ax.plot(x_axis, e2ed_train_loss_N_3_per_epoch)
    ax.set(xlabel='epochs', ylabel='loss', title='Train loss per epoch')
    ax.grid()
    plt.legend(["GD: N=2", "GD: N=3", "E2ED: N=2", "E2ED: N=3"])

    fig.savefig("statistics/results/train_loss.png")
    plt.close(fig)

    # Train test plot
    x_axis = list(range(len(gd_test_loss_N_2_per_epoch)))
    fig, ax = plt.subplots()
    ax.plot(x_axis, gd_test_loss_N_2_per_epoch)
    ax.plot(x_axis, gd_test_loss_N_3_per_epoch)
    ax.plot(x_axis, e2ed_test_loss_N_2_per_epoch)
    ax.plot(x_axis, e2ed_test_loss_N_3_per_epoch)
    ax.set(xlabel='epochs', ylabel='loss', title='Test loss per epoch')
    ax.grid()
    plt.legend(["GD: N=2", "GD: N=3", "E2ED: N=2", "E2ED: N=3"])

    fig.savefig("statistics/results/test_loss.png")
    plt.close(fig)