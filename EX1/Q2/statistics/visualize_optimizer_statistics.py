import matplotlib.pyplot as plt
import numpy as np

def visualize_loss(train_loss, test_loss, optimizer_name):
    x_axis = list(range(1, len(train_loss)+1))

    fig, ax = plt.subplots()
    ax.plot(x_axis, train_loss)
    ax.plot(x_axis, test_loss)
    ax.set(xlabel='epochs', ylabel='loss',
           title="Loss " + optimizer_name)
    ax.grid()
    plt.legend(["train_loss", "test_loss"])

    fig.savefig("statistics/results/optimization/loss/" + optimizer_name+ ".png")
    plt.close(fig)

def visualize_acc(train_acc, test_acc, optimizer_name):
    x_axis = list(range(1, len(train_acc)+1))

    fig, ax = plt.subplots()
    ax.plot(x_axis, train_acc)
    ax.plot(x_axis, test_acc)
    ax.set(xlabel='epochs', ylabel='acc',
           title="Accuracy " + optimizer_name)
    ax.grid()
    plt.legend(["train_acc", "test_acc"])

    fig.savefig("statistics/results/optimization/acc/" + optimizer_name + ".png")
    plt.close(fig)

if __name__ == "__main__":
    sgd_stats = np.load("statistics/results/optimization/results/optim_sgd_stats.npy")
    adam_stats = np.load("statistics/results/optimization/results/optim_adam_stats.npy")

    sdg_train_loss, sgd_test_loss, sgd_train_acc, sgd_test_acc = sgd_stats
    adam_train_loss, adam_test_loss, adam_train_acc, adam_test_acc = adam_stats

    visualize_loss(sdg_train_loss, sgd_test_loss, "SGD")
    visualize_acc(sgd_train_acc, sgd_test_acc, "SGD")
    visualize_loss(adam_train_loss, adam_test_loss, "Adam")
    visualize_acc(adam_train_acc, adam_test_acc, "Adam")

