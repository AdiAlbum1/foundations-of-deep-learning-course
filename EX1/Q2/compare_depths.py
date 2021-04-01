from dataset_extractor import load_dataset
from baseline_model import Baseline_Network

import torch
import numpy as np

from statistics.calc_statistics import calc_dataset_acc, calc_dataset_loss


if __name__ == "__main__":
    # D_in is input dimension; H is hidden dimension; D_out is output dimension.
    epochs = 80
    batch_size = 32
    # selected optimization parameters
    std, learning_rate, momentum = 0.1, 1e-3, 0.9
    net_width = 64

    # load dataset
    train_dataloader, test_dataloader = load_dataset(batch_size)

    # Define Loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    network_depths = [3, 4, 10]

    for network_depth in network_depths:
        # Define NN model
        net = Baseline_Network(net_width=net_width, n_hidden_layers=network_depth)

        # random initialize net
        net.normal_random_init(std=std)

        # Define Optimizer
        optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

        train_loss_per_epoch, test_loss_per_epoch, train_acc_per_epoch, test_acc_per_epoch = [], [], [], []

        # Train Network
        for epoch in range(epochs):
            for sample_batched in train_dataloader:
                images, labels = sample_batched

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(images.float())
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

            # calculate train & test loss
            epoch_train_loss = calc_dataset_loss(train_dataloader, net, loss_fn)
            epoch_test_loss = calc_dataset_loss(test_dataloader, net, loss_fn)
            # calculate train & test accuracy
            epoch_train_acc = calc_dataset_acc(train_dataloader, net)
            epoch_test_acc = calc_dataset_acc(test_dataloader, net)

            print('[epoch %d]\ttrain_loss: %.3f\t test_loss: %.3f\ttrain_acc: %.2f%%\ttest_acc: %.2f%%' %
                  (epoch + 1, epoch_train_loss, epoch_test_loss, 100*epoch_train_acc, 100*epoch_test_acc))

            train_loss_per_epoch.append(epoch_train_loss)
            test_loss_per_epoch.append(epoch_test_loss)
            train_acc_per_epoch.append(epoch_train_acc)
            test_acc_per_epoch.append(epoch_test_acc)

        np.save("statistics/results/depths/results/depth_%d_stats" % network_depth, np.array([train_loss_per_epoch, test_loss_per_epoch, train_acc_per_epoch, test_acc_per_epoch]))
        print()