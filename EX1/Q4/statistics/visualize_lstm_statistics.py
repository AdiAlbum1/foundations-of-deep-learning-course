import matplotlib.pyplot as plt
import numpy as np

def visualize_loss(train_loss, test_loss):
    x_axis = list(range(1, len(train_loss)+1))

    fig, ax = plt.subplots()
    ax.plot(x_axis, train_loss)
    ax.plot(x_axis, test_loss)
    ax.set(xlabel='epochs', ylabel='loss',
           title="LSTM loss")
    ax.grid()
    plt.legend(["train_loss", "test_loss"])

    fig.savefig("statistics/results/lstm/loss.png")
    plt.close(fig)

if __name__ == "__main__":
    stats = np.load("statistics/results/lstm/stats.npy")
    train_loss, test_loss = stats

    visualize_loss(train_loss, test_loss)

