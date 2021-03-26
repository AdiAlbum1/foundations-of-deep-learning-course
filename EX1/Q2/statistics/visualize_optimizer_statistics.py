import matplotlib.pyplot as plt
import numpy as np

def visualize_loss(train_loss_1, test_loss_1, optimizer_name_1, train_loss_2, test_loss_2, optimizer_name_2):
    x_axis = list(range(1, len(train_loss_1)+1))

    fig, ax = plt.subplots()
    ax.plot(x_axis, train_loss_1, color=(1.0, 0.0, 0.0))
    ax.plot(x_axis, train_loss_2, color=(0.0, 1.0, 0.0))
    ax.plot(x_axis, test_loss_1, color=(0.0, 0.0, 1.0))
    ax.plot(x_axis, test_loss_2, color=(0.33, 0.33, 0.33))
    ax.set(xlabel='epochs', ylabel='loss',
           title="Compare Optimizers Loss")
    ax.grid()
    plt.legend([optimizer_name_1 + " - train_loss", optimizer_name_2 + " - train_loss", optimizer_name_1 + " - test_loss", optimizer_name_2 + " - test_loss"])

    fig.savefig("statistics/results/optimization/loss.png")
    plt.close(fig)

def visualize_acc(train_acc_1, test_acc_1, optimizer_name_1, train_acc_2, test_acc_2, optimizer_name_2):
    x_axis = list(range(1, len(train_acc_1)+1))

    fig, ax = plt.subplots()
    ax.plot(x_axis, train_acc_1, color=(1.0, 0.0, 0.0))
    ax.plot(x_axis, train_acc_2, color=(0.0, 1.0, 0.0))
    ax.plot(x_axis, test_acc_1, color=(0.0, 0.0, 1.0))
    ax.plot(x_axis, test_acc_2, color=(0.33, 0.33, 0.33))
    ax.set(xlabel='epochs', ylabel='acc',
           title="Compare Optimizers Accuracy")
    ax.grid()
    plt.legend([optimizer_name_1 + " - train_acc", optimizer_name_2 + " - train_acc", optimizer_name_1 + " - test_acc", optimizer_name_2 + " - test_acc"])

    fig.savefig("statistics/results/optimization/acc.png")
    plt.close(fig)

if __name__ == "__main__":
    sgd_stats = np.load("statistics/results/optimization/results/optim_sgd_stats.npy")
    adam_stats = np.load("statistics/results/optimization/results/optim_adam_stats.npy")

    sdg_train_loss, sgd_test_loss, sgd_train_acc, sgd_test_acc = sgd_stats
    adam_train_loss, adam_test_loss, adam_train_acc, adam_test_acc = adam_stats

    visualize_loss(sdg_train_loss, sgd_test_loss, "SGD", adam_train_loss, adam_test_loss, "Adam")
    visualize_acc(sgd_train_acc, sgd_test_acc, "SGD", adam_train_acc, adam_test_acc, "Adam")

