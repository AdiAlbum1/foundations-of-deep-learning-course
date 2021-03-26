import matplotlib.pyplot as plt
import numpy as np

vis_num_epochs = 100

def visualize_loss(train_loss, test_loss, i, j, k, std, learning_rate, momentum):
    x_axis = list(range(3, vis_num_epochs+3))

    fig, ax = plt.subplots()
    ax.plot(x_axis, train_loss[:vis_num_epochs])
    ax.plot(x_axis, test_loss[:vis_num_epochs])
    ax.set(xlabel='epochs', ylabel='loss',
           title="loss\nstd=%.1f, learning_rate=%.3f, momentum=%.1f" %(std, learning_rate, momentum))
    ax.grid()
    plt.legend(["train_loss", "test_loss"])

    fig.savefig("statistics/results/baseline/loss/loss_%d_%d_%d.png" %(i, j, k))
    plt.close(fig)

def visualize_acc(train_acc, test_acc, i, j, k, std, learning_rate, momentum):
    x_axis = list(range(3, vis_num_epochs+3))

    fig, ax = plt.subplots()
    ax.plot(x_axis, train_acc[:vis_num_epochs])
    ax.plot(x_axis, test_acc[:vis_num_epochs])
    ax.set(xlabel='epochs', ylabel='acc',
           title="accuracy\nstd=%.1f, learning_rate=%.3f, momentum=%.1f" %(std, learning_rate, momentum))
    ax.grid()
    plt.legend(["train_acc", "test_acc"])

    fig.savefig("statistics/results/baseline/acc/acc_%d_%d_%d.png" %(i, j, k))
    plt.close(fig)

if __name__ == "__main__":
    stds = [0.05, 0.1, 0.3]
    learning_rates = [1e-3, 1e-2, 1e-1]
    momentums = [0, 0.5, 0.9]

    for i, std in enumerate(stds):
        for j, learning_rate in enumerate(learning_rates):
            for k, momentum in enumerate(momentums):
                train_stats = np.load("statistics/results/baseline/results/curr_stats_%d_%d_%d.npy" % (i, j, k))

                train_loss_per_epoch, test_loss_per_epoch, train_acc_per_epoch, test_acc_per_epoch = train_stats
                train_loss_per_epoch = train_loss_per_epoch[2:]
                test_loss_per_epoch = test_loss_per_epoch[2:]
                train_acc_per_epoch = train_acc_per_epoch[2:]
                test_acc_per_epoch = test_acc_per_epoch[2:]

                visualize_loss(train_loss_per_epoch, test_loss_per_epoch, i, j, k, std, learning_rate, momentum)
                visualize_acc(train_acc_per_epoch, test_acc_per_epoch, i, j, k, std, learning_rate, momentum)

