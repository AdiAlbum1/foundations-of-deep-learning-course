import matplotlib.pyplot as plt
import numpy as np

def visualize_loss(train_loss, test_loss, i, j, k, std, learning_rate, momentum):
    epochs = len(train_loss)
    x_axis = list(range(1, epochs+1))

    fig, ax = plt.subplots()
    ax.plot(x_axis, train_loss)
    ax.plot(x_axis, test_loss)
    ax.set(xlabel='epochs', ylabel='loss',
           title="loss\nstd=%.2f, learning_rate=%.2f, momentum=%.2f" %(std, learning_rate, momentum))
    ax.grid()
    plt.legend(["train_loss", "test_loss"])

    fig.savefig("statistics/results/loss_%d_%d_%d.png" %(i, j, k))
    # plt.show()

def visualize_acc(train_acc, test_acc, i, j, k, std, learning_rate, momentum):
    epochs = len(train_acc)
    x_axis = list(range(1, epochs+1))

    fig, ax = plt.subplots()
    ax.plot(x_axis, train_acc)
    ax.plot(x_axis, test_acc)
    ax.set(xlabel='epochs', ylabel='acc',
           title="accuracy\nstd=%.2f, learning_rate=%.2f, momentum=%.2f" %(std, learning_rate, momentum))
    ax.grid()
    plt.legend(["train_acc", "test_acc"])

    fig.savefig("statistics/results/acc_%d_%d_%d.png" %(i, j, k))
    # plt.show()

if __name__ == "__main__":
    stds = [0.7, 1.0, 1.3]
    learning_rates = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    momentums = [0.5, 0.9, 0.98]

    for i, std in enumerate(stds):
        for j, learning_rate in enumerate(learning_rates):
            for k, momentum in enumerate(momentums):
                train_stats = np.load("statistics/curr_stats_%d_%d_%d.npy" % (i, j, k))

                train_loss_per_epoch, test_loss_per_epoch, train_acc_per_epoch, test_acc_per_epoch = train_stats

                visualize_loss(train_loss_per_epoch, test_loss_per_epoch, i, j, k, std, learning_rate, momentum)
                visualize_acc(train_acc_per_epoch, test_acc_per_epoch, i, j, k, std, learning_rate, momentum)

