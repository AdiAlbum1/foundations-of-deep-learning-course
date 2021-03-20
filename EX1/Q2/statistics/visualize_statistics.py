import matplotlib.pyplot as plt
import numpy as np

def visualize_loss(train_loss, test_loss):
    epochs = len(train_loss)
    x_axis = list(range(1, epochs+1))

    fig, ax = plt.subplots()
    ax.plot(x_axis, train_loss)
    ax.plot(x_axis, test_loss)
    ax.set(xlabel='epochs', ylabel='loss',
           title="loss monitor")
    ax.grid()
    plt.legend(["train_loss", "test_loss"])

    fig.savefig("statistics/loss.png")
    plt.show()

def visualize_acc(train_acc, test_acc):
    epochs = len(train_acc)
    x_axis = list(range(1, epochs+1))

    fig, ax = plt.subplots()
    ax.plot(x_axis, train_acc)
    ax.plot(x_axis, test_acc)
    ax.set(xlabel='epochs', ylabel='acc',
           title="accuracy monitor")
    ax.grid()
    plt.legend(["train_acc", "test_acc"])

    fig.savefig("statistics/acc.png")
    plt.show()

if __name__ == "__main__":
    train_stats = np.load("statistics/curr_stats.npy")

    train_loss_per_epoch, test_loss_per_epoch, train_acc_per_epoch, test_acc_per_epoch = train_stats

    visualize_loss(train_loss_per_epoch, test_loss_per_epoch)
    visualize_acc(train_acc_per_epoch, test_acc_per_epoch)

