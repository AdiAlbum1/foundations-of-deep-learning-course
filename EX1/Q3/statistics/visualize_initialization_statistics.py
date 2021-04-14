import matplotlib.pyplot as plt
import numpy as np

def visualize_loss(train_loss_1, test_loss_1, init_name_1, train_loss_2, test_loss_2, init_name_2):
    x_axis = list(range(1, len(train_loss_1)+1))

    fig, ax = plt.subplots()
    ax.plot(x_axis, train_loss_1, color=(1.0, 0.0, 0.0))
    ax.plot(x_axis, train_loss_2, color=(0.0, 1.0, 0.0))
    ax.plot(x_axis, test_loss_1, color=(0.0, 0.0, 1.0))
    ax.plot(x_axis, test_loss_2, color=(0.33, 0.33, 0.33))
    ax.set(xlabel='epochs', ylabel='loss',
           title="Compare Init Methods Loss")
    ax.grid()
    plt.legend([init_name_1 + " - train_loss", init_name_2 + " - train_loss", init_name_1 + " - test_loss", init_name_2 + " - test_loss"])

    fig.savefig("statistics/results/initialization/loss.png")
    plt.close(fig)

def visualize_acc(train_acc_1, test_acc_1, init_name_1, train_acc_2, test_acc_2, init_name_2):
    x_axis = list(range(1, len(train_acc_1)+1))

    fig, ax = plt.subplots()
    ax.plot(x_axis, train_acc_1, color=(1.0, 0.0, 0.0))
    ax.plot(x_axis, train_acc_2, color=(0.0, 1.0, 0.0))
    ax.plot(x_axis, test_acc_1, color=(0.0, 0.0, 1.0))
    ax.plot(x_axis, test_acc_2, color=(0.33, 0.33, 0.33))
    ax.set(xlabel='epochs', ylabel='acc',
           title="Compare Init Methods Accuracy")
    ax.grid()
    plt.legend([init_name_1 + " - train_acc", init_name_2 + " - train_acc", init_name_1 + " - test_acc", init_name_2 + " - test_acc"])

    fig.savefig("statistics/results/initialization/acc.png")
    plt.close(fig)

if __name__ == "__main__":
    normal_stats = np.load("statistics/results/initialization/results/random_init_stats.npy", allow_pickle=True)
    xavier_stats = np.load("statistics/results/initialization/results/xavier_init_stats.npy", allow_pickle=True)

    normal_train_loss, normal_test_loss, normal_train_acc, normal_test_acc = normal_stats
    xavier_train_loss, xavier_test_loss, xavier_train_acc, xavier_test_acc = xavier_stats

    visualize_loss(normal_train_loss, normal_test_loss, "Normal", xavier_train_loss, xavier_test_loss, "Xavier")
    visualize_acc(normal_train_acc, normal_test_acc, "Normal", xavier_train_acc, xavier_test_acc, "Xavier")

