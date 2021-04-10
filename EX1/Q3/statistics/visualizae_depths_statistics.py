import matplotlib.pyplot as plt
import numpy as np

def visualize_loss(train_loss_0, test_loss_0, train_loss_1, test_loss_1, train_loss_2, test_loss_2, train_loss_3, test_loss_3, network_widths):
    x_axis = list(range(1, len(train_loss_1)+1))

    fig, ax = plt.subplots()
    ax.plot(x_axis, train_loss_0, color=(1.0, 0.0, 0.0))
    ax.plot(x_axis, train_loss_1, color=(0.0, 1.0, 0.0))
    ax.plot(x_axis, train_loss_2, color=(0.0, 0.0, 1.0))
    ax.plot(x_axis, train_loss_3, color=(0.5, 0.5, 0.5))
    ax.plot(x_axis, test_loss_0, color=(0.6, 0.2, 0.2))
    ax.plot(x_axis, test_loss_1, color=(0.2, 0.6, 0.2))
    ax.plot(x_axis, test_loss_2, color=(0.2, 0.2, 0.6))
    ax.plot(x_axis, test_loss_3, color=(0.5, 0.5, 0.5))
    ax.set(xlabel='epochs', ylabel='loss',
           title="Compare depths Loss")
    ax.grid()

    legend = ["train_loss - depth = " + str(width) for width in network_widths] +\
             ["test_loss - depth = " + str(width) for width in network_widths]
    plt.legend(legend)

    fig.savefig("statistics/results/depths/loss.png")
    plt.close(fig)

def visualize_acc(train_acc_0, test_acc_0, train_acc_1, test_acc_1, train_acc_2, test_acc_2, train_acc_3, test_acc_3, network_widths):
    x_axis = list(range(1, len(train_acc_1)+1))

    fig, ax = plt.subplots()
    ax.plot(x_axis, train_acc_0, color=(1.0, 0.0, 0.0))
    ax.plot(x_axis, train_acc_1, color=(0.0, 1.0, 0.0))
    ax.plot(x_axis, train_acc_2, color=(0.0, 0.0, 1.0))
    ax.plot(x_axis, train_acc_3, color=(0.5, 0.5, 0.5))
    ax.plot(x_axis, test_acc_0, color=(0.6, 0.2, 0.2))
    ax.plot(x_axis, test_acc_1, color=(0.2, 0.6, 0.2))
    ax.plot(x_axis, test_acc_2, color=(0.2, 0.2, 0.6))
    ax.plot(x_axis, test_acc_3, color=(0.5, 0.5, 0.5))
    ax.set(xlabel='epochs', ylabel='acc',
           title="Compare depths Accuracy")
    ax.grid()
    legend = ["train_acc - depth = " + str(width) for width in network_widths] + \
             ["test_acc - depth = " + str(width) for width in network_widths]
    plt.legend(legend)

    fig.savefig("statistics/results/depths/acc.png")
    plt.close(fig)

if __name__ == "__main__":
    network_widths = [2, 3, 4, 5]

    width_stats_0 = np.load("statistics/results/depths/results/depth_2_stats.npy", allow_pickle=True)
    width_stats_1 = np.load("statistics/results/depths/results/depth_3_stats.npy", allow_pickle=True)
    width_stats_2 = np.load("statistics/results/depths/results/depth_4_stats.npy", allow_pickle=True)
    width_stats_3 = np.load("statistics/results/depths/results/depth_5_stats.npy", allow_pickle=True)

    train_loss_0, test_loss_0, train_acc_0, test_acc_0 = width_stats_0
    train_loss_1, test_loss_1, train_acc_1, test_acc_1 = width_stats_1
    train_loss_2, test_loss_2, train_acc_2, test_acc_2 = width_stats_2
    train_loss_3, test_loss_3, train_acc_3, test_acc_3 = width_stats_3

    visualize_loss(train_loss_0, test_loss_0, train_loss_1, test_loss_1, train_loss_2, test_loss_2, train_loss_3, test_loss_3, network_widths)
    visualize_acc(train_acc_0, test_acc_0, train_acc_1, test_acc_1, train_acc_2, test_acc_2, train_acc_3, test_acc_3, network_widths)