import matplotlib.pyplot as plt
import numpy as np

def visualize_loss(train_loss_1, test_loss_1, train_loss_2, test_loss_2):
    x_axis = list(range(1, len(train_loss_1)+1))

    fig, ax = plt.subplots()
    ax.plot(x_axis, train_loss_1, color=(1.0, 0.0, 0.0))
    ax.plot(x_axis, train_loss_2, color=(0.0, 1.0, 0.0))
    ax.plot(x_axis, test_loss_1, color=(0.6, 0.2, 0.2))
    ax.plot(x_axis, test_loss_2, color=(0.2, 0.6, 0.2))
    ax.set(xlabel='epochs', ylabel='loss',
           title="Compare preprocessing Loss")
    ax.grid()

    legend = ["train_loss - no preprocessing", "train_loss - whitening",\
              "test_loss - no preprocessing", "test_loss - whitening"]
    plt.legend(legend)

    fig.savefig("statistics/results/preprocessing/loss.png")
    plt.close(fig)

def visualize_acc(train_acc_1, test_acc_1, train_acc_2, test_acc_2):
    x_axis = list(range(1, len(train_acc_1)+1))

    fig, ax = plt.subplots()
    ax.plot(x_axis, train_acc_1, color=(1.0, 0.0, 0.0))
    ax.plot(x_axis, train_acc_2, color=(0.0, 1.0, 0.0))
    ax.plot(x_axis, test_acc_1, color=(0.6, 0.2, 0.2))
    ax.plot(x_axis, test_acc_2, color=(0.2, 0.6, 0.2))
    ax.set(xlabel='epochs', ylabel='acc',
           title="Compare preprocessing Accuracy")
    ax.grid()
    legend = ["train_acc - no preprocessing", "train_acc - whitening",\
              "test_acc - no preprocessing", "test_acc - whitening"]
    plt.legend(legend)

    fig.savefig("statistics/results/preprocessing/acc.png")
    plt.close(fig)

if __name__ == "__main__":
    no_preprocess_stats = np.load("statistics/results/preprocessing/results/no_preprocessing_stats.npy")
    whitened_stats = np.load("statistics/results/preprocessing/results/whitening_stats.npy")

    train_loss_0, test_loss_0, train_acc_0, test_acc_0 = no_preprocess_stats
    train_loss_1, test_loss_1, train_acc_1, test_acc_1 = whitened_stats

    visualize_loss(train_loss_0, test_loss_0, train_loss_1, test_loss_1)
    visualize_acc(train_acc_0, test_acc_0, train_acc_1, test_acc_1)