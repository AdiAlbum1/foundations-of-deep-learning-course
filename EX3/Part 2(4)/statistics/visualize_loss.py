import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    train_stats = np.load("statistics/results/train_stats.npy")

    train_loss_per_epoch, test_loss_per_epoch = train_stats

    x_axis = list(range(len(train_loss_per_epoch)))
    fig, ax = plt.subplots()
    ax.plot(x_axis, train_loss_per_epoch)
    ax.plot(x_axis, test_loss_per_epoch)
    ax.set(xlabel='epochs', ylabel='loss', title='training procedure')
    ax.grid()
    plt.legend(["train_loss", "test_loss"])

    fig.savefig("statistics/results/loss.png")
    plt.close(fig)