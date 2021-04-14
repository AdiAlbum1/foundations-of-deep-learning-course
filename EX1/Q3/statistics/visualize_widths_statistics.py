import matplotlib.pyplot as plt
import numpy as np

def visualize_loss(train_loss_0, test_loss_0, train_loss_1, test_loss_1, train_loss_2, test_loss_2, network_widths):
    x_axis = list(range(1, len(train_loss_1)+1))

    fig, ax = plt.subplots()
    ax.plot(x_axis, train_loss_0, color=(1.0, 0.0, 0.0))
    ax.plot(x_axis, train_loss_1, color=(0.0, 1.0, 0.0))
    ax.plot(x_axis, train_loss_2, color=(0.0, 0.0, 1.0))
    ax.plot(x_axis, test_loss_0, color=(0.6, 0.2, 0.2))
    ax.plot(x_axis, test_loss_1, color=(0.2, 0.6, 0.2))
    ax.plot(x_axis, test_loss_2, color=(0.2, 0.2, 0.6))
    ax.set(xlabel='epochs', ylabel='loss',
           title="Compare widths Loss")
    ax.grid()

    legend = ["train_loss - width = " + str(width) for width in network_widths] +\
             ["test_loss - width = " + str(width) for width in network_widths]
    plt.legend(legend)

    fig.savefig("statistics/results/widths/loss.png")
    plt.close(fig)

def visualize_acc(train_acc_0, test_acc_0, train_acc_1, test_acc_1, train_acc_2, test_acc_2, network_widths):
    x_axis = list(range(1, len(train_acc_1)+1))

    fig, ax = plt.subplots()
    ax.plot(x_axis, train_acc_0, color=(1.0, 0.0, 0.0))
    ax.plot(x_axis, train_acc_1, color=(0.0, 1.0, 0.0))
    ax.plot(x_axis, train_acc_2, color=(0.0, 0.0, 1.0))
    ax.plot(x_axis, test_acc_0, color=(0.6, 0.2, 0.2))
    ax.plot(x_axis, test_acc_1, color=(0.2, 0.6, 0.2))
    ax.plot(x_axis, test_acc_2, color=(0.2, 0.2, 0.6))
    ax.set(xlabel='epochs', ylabel='acc',
           title="Compare widths Accuracy")
    ax.grid()
    legend = ["train_acc - width = " + str(width) for width in network_widths] + \
             ["test_acc - width = " + str(width) for width in network_widths]
    plt.legend(legend)

    fig.savefig("statistics/results/widths/acc.png")
    plt.close(fig)

if __name__ == "__main__":
    network_widths = [(64, 16), (256, 64), (512, 256)]

    width_stats_0 = np.load("statistics/results/widths/results/width_(64, 16)_stats.npy", allow_pickle=True)
    width_stats_1 = np.load("statistics/results/widths/results/width_(256, 64)_stats.npy", allow_pickle=True)
    width_stats_2 = np.load("statistics/results/widths/results/width_(512, 256)_stats.npy", allow_pickle=True)

    train_loss_0, test_loss_0, train_acc_0, test_acc_0 = width_stats_0
    train_loss_1, test_loss_1, train_acc_1, test_acc_1 = width_stats_1
    train_loss_2, test_loss_2, train_acc_2, test_acc_2 = width_stats_2

    visualize_loss(train_loss_0, test_loss_0, train_loss_1, test_loss_1, train_loss_2, test_loss_2, network_widths)
    visualize_acc(train_acc_0, test_acc_0, train_acc_1, test_acc_1, train_acc_2, test_acc_2, network_widths)