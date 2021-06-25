import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    train_stats = np.load("statistics/results/train_statistics.npy")

    train_loss, train_accuracy, test_loss, test_accuracy = train_stats

    # loss plot
    x_axis = list(range(len(train_loss)))
    fig, ax = plt.subplots()
    ax.plot(x_axis, train_loss)
    ax.plot(x_axis, test_loss)
    ax.set(xlabel='epochs', ylabel='loss', title='Loss per epoch')
    ax.grid()
    plt.legend(["train loss", "test loss"])

    fig.savefig("statistics/results/loss.png")
    plt.close(fig)

    # acc plot
    x_axis = list(range(len(train_accuracy)))
    fig, ax = plt.subplots()
    ax.plot(x_axis, train_accuracy)
    ax.plot(x_axis, test_accuracy)
    ax.set(xlabel='epochs', ylabel='acc', title='Accuracy per epoch')
    ax.grid()
    plt.legend(["train accuracy", "test accuracy"])

    fig.savefig("statistics/results/accuracy.png")
    plt.close(fig)