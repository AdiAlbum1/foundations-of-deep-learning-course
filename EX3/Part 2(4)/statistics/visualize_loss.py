import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    neural_net_depths = [2, 3, 4]
    for N in neural_net_depths:
        train_stats = np.load("statistics/results/depth_%d/train_stats.npy" %N)

        train_loss_per_epoch, test_loss_per_epoch = train_stats

        x_axis = list(range(len(train_loss_per_epoch)))
        fig, ax = plt.subplots()
        ax.plot(x_axis, train_loss_per_epoch)
        ax.plot(x_axis, test_loss_per_epoch)
        ax.set(xlabel='epochs', ylabel='loss', title='training procedure')
        ax.grid()
        plt.legend(["train_loss", "test_loss"])
        plt.title("Loss\nDepth: %d"%N)

        fig.savefig("statistics/results/depth_%d/loss.png" %N)
        plt.close(fig)