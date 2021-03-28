import matplotlib.pyplot as plt
import numpy as np

def visualize_loss(train_loss_1, test_loss_1, train_loss_2, test_loss_2, train_loss_3, test_loss_3, reg_type, values):
    x_axis = list(range(1, len(train_loss_1)+1))

    fig, ax = plt.subplots()
    ax.plot(x_axis, train_loss_1, color=(1.0, 0.0, 0.0))
    ax.plot(x_axis, train_loss_2, color=(0.0, 1.0, 0.0))
    ax.plot(x_axis, train_loss_3, color=(0.0, 0.0, 1.0))
    ax.plot(x_axis, test_loss_1, color=(0.6, 0.2, 0.2))
    ax.plot(x_axis, test_loss_2, color=(0.2, 0.6, 0.2))
    ax.plot(x_axis, test_loss_3, color=(0.2, 0.2, 0.6))
    ax.set(xlabel='epochs', ylabel='loss',
           title="Compare " + reg_type + " Loss")
    ax.grid()
    legend = ["train_loss - " + reg_type + " = "+str(val) for val in values] +\
             ["test_loss - " + reg_type + " = "+str(val) for val in values]
    plt.legend(legend)

    fig.savefig("statistics/results/regularization/loss_" + reg_type + ".png")
    plt.close(fig)

def visualize_acc(train_acc_1, test_acc_1, train_acc_2, test_acc_2, train_acc_3, test_acc_3, reg_type, values):
    x_axis = list(range(1, len(train_acc_1)+1))

    fig, ax = plt.subplots()
    ax.plot(x_axis, train_acc_1, color=(1.0, 0.0, 0.0))
    ax.plot(x_axis, train_acc_2, color=(0.0, 1.0, 0.0))
    ax.plot(x_axis, train_acc_3, color=(0.0, 0.0, 1.0))
    ax.plot(x_axis, test_acc_1, color=(0.6, 0.2, 0.2))
    ax.plot(x_axis, test_acc_2, color=(0.2, 0.6, 0.2))
    ax.plot(x_axis, test_acc_3, color=(0.2, 0.2, 0.6))
    ax.set(xlabel='epochs', ylabel='acc',
           title="Compare " + reg_type + " Accuracy")
    ax.grid()
    legend = ["train_acc - " + reg_type + " = " + str(val) for val in values] +\
             ["test_acc - " + reg_type + " = " + str(val) for val in values]
    plt.legend(legend)

    fig.savefig("statistics/results/regularization/acc_" + reg_type + ".png")
    plt.close(fig)

if __name__ == "__main__":
    weight_decay_vals = [0, 0.01, 0.05]
    p_dropouts = [0, 0.2, 0.35]


    weight_decay_stats_0 = np.load("statistics/results/regularization/results/regularization_weight_decay_stats_0.npy")
    weight_decay_stats_1 = np.load("statistics/results/regularization/results/regularization_weight_decay_stats_1.npy")
    weight_decay_stats_2 = np.load("statistics/results/regularization/results/regularization_weight_decay_stats_2.npy")
    dropout_stats_0 = np.load("statistics/results/regularization/results/regularization_dropout_stats_0.npy")
    dropout_stats_1 = np.load("statistics/results/regularization/results/regularization_dropout_stats_1.npy")
    dropout_stats_2 = np.load("statistics/results/regularization/results/regularization_dropout_stats_2.npy")

    train_loss_0, test_loss_0, train_acc_0, test_acc_0 = weight_decay_stats_0
    train_loss_1, test_loss_1, train_acc_1, test_acc_1 = weight_decay_stats_1
    train_loss_2, test_loss_2, train_acc_2, test_acc_2 = weight_decay_stats_2
    train_loss_3, test_loss_3, train_acc_3, test_acc_3 = dropout_stats_0
    train_loss_4, test_loss_4, train_acc_4, test_acc_4 = dropout_stats_1
    train_loss_5, test_loss_5, train_acc_5, test_acc_5 = dropout_stats_1

    visualize_loss(train_loss_0, test_loss_0, train_loss_1, test_loss_1, train_loss_2, test_loss_2, "weight_decay", weight_decay_vals)
    visualize_acc(train_acc_0, test_acc_0, train_acc_1, test_acc_1, train_acc_2, test_acc_2, "weight_decay", weight_decay_vals)
    visualize_loss(train_loss_3, test_loss_3, train_loss_4, test_loss_4, train_loss_5, test_loss_5, "dropout_prob", p_dropouts)
    visualize_acc(train_acc_3, test_acc_3, train_acc_4, test_acc_4, train_acc_5, test_acc_5, "dropout_prob", p_dropouts)