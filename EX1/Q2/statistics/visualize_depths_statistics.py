import matplotlib.pyplot as plt
import numpy as np

def visualize_loss(train_loss_0, test_loss_0, train_loss_1, test_loss_1, train_loss_2, test_loss_2, network_depths):
    x_axis = list(range(1, len(train_loss_1)+1))

    fig, ax = plt.subplots()
    ax.plot(x_axis, train_loss_0, color=(1.0, 0.0, 0.0))
    ax.plot(x_axis, train_loss_1, color=(0.0, 1.0, 0.0))
    ax.plot(x_axis, train_loss_2, color=(0.0, 0.0, 1.0))
    ax.plot(x_axis, test_loss_0, color=(0.6, 0.2, 0.2))
    ax.plot(x_axis, test_loss_1, color=(0.2, 0.6, 0.2))
    ax.plot(x_axis, test_loss_2, color=(0.2, 0.2, 0.6))
    ax.set(xlabel='epochs', ylabel='loss',
           title="Compare depths Loss")
    ax.grid()

    legend = ["train_loss - depth = " + str(depth) for depth in network_depths] +\
             ["test_loss - depth = " + str(depth) for depth in network_depths]
    plt.legend(legend)

    fig.savefig("statistics/results/depths/loss.png")
    plt.close(fig)

def visualize_acc(train_acc_0, test_acc_0, train_acc_1, test_acc_1, train_acc_2, test_acc_2, network_depths):
    x_axis = list(range(1, len(train_acc_1)+1))

    fig, ax = plt.subplots()
    ax.plot(x_axis, train_acc_0, color=(1.0, 0.0, 0.0))
    ax.plot(x_axis, train_acc_1, color=(0.0, 1.0, 0.0))
    ax.plot(x_axis, train_acc_2, color=(0.0, 0.0, 1.0))
    ax.plot(x_axis, test_acc_0, color=(0.6, 0.2, 0.2))
    ax.plot(x_axis, test_acc_1, color=(0.2, 0.6, 0.2))
    ax.plot(x_axis, test_acc_2, color=(0.2, 0.2, 0.6))
    ax.set(xlabel='epochs', ylabel='acc',
           title="Compare depths Accuracy")
    ax.grid()
    legend = ["train_acc - depth = " + str(depth) for depth in network_depths] + \
             ["test_acc - depth = " + str(depth) for depth in network_depths]
    plt.legend(legend)

    fig.savefig("statistics/results/depths/acc.png")
    plt.close(fig)

if __name__ == "__main__":
    network_depths = [3, 4, 10]

    depth_stats_0 = np.load("statistics/results/depths/results/depth_3_stats.npy")
    depth_stats_1 = np.load("statistics/results/depths/results/depth_4_stats.npy")
    depth_stats_2 = np.load("statistics/results/depths/results/depth_10_stats.npy")

    train_loss_0, test_loss_0, train_acc_0, test_acc_0 = depth_stats_0
    train_loss_1, test_loss_1, train_acc_1, test_acc_1 = depth_stats_1
    train_loss_2, test_loss_2, train_acc_2, test_acc_2 = depth_stats_2

    visualize_loss(train_loss_0, test_loss_0, train_loss_1, test_loss_1, train_loss_2, test_loss_2, network_depths)
    visualize_acc(train_acc_0, test_acc_0, train_acc_1, test_acc_1, train_acc_2, test_acc_2, network_depths)