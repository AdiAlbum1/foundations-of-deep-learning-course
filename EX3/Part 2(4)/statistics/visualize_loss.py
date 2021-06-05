import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    train_stats_N_2 = np.load("statistics/results/depth_2/train_stats.npy")
    train_stats_N_3 = np.load("statistics/results/depth_3/train_stats.npy")
    train_stats_N_4 = np.load("statistics/results/depth_4/train_stats.npy")

    train_loss_N_2_per_epoch, test_loss_N_2_per_epoch = train_stats_N_2
    train_loss_N_3_per_epoch, test_loss_N_3_per_epoch = train_stats_N_3
    train_loss_N_4_per_epoch, test_loss_N_4_per_epoch = train_stats_N_4

    # Train loss plot
    x_axis = list(range(3, len(train_loss_N_2_per_epoch)))
    fig, ax = plt.subplots()
    ax.plot(x_axis, train_loss_N_2_per_epoch[3:])
    ax.plot(x_axis, train_loss_N_3_per_epoch[3:])
    ax.plot(x_axis, train_loss_N_4_per_epoch[3:])
    ax.set(xlabel='epochs', ylabel='loss', title='Train loss per epoch')
    ax.grid()
    plt.legend(["N=2", "N=3", "N=4"])

    fig.savefig("statistics/results/train_loss.png")
    plt.close(fig)

    # Train test plot
    x_axis = list(range(3, len(test_loss_N_2_per_epoch)))
    fig, ax = plt.subplots()
    ax.plot(x_axis, test_loss_N_2_per_epoch[3:])
    ax.plot(x_axis, test_loss_N_3_per_epoch[3:])
    ax.plot(x_axis, test_loss_N_4_per_epoch[3:])
    ax.set(xlabel='epochs', ylabel='loss', title='Test loss per epoch')
    ax.grid()
    plt.legend(["N=2", "N=3", "N=4"])

    fig.savefig("statistics/results/test_loss.png")
    plt.close(fig)