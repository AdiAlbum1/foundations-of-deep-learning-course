from dataset_extractor import load_dataset
from baseline_model import Baseline_Network

import torch
import numpy as np

from statistics.calc_statistics import calc_dataset_acc, calc_dataset_loss

if __name__ == "__main__":
    # D_in is input dimension; H is hidden dimension; D_out is output dimension.
    epochs = 150
    batch_size, D_in, H, D_out = 32, 3072, 256, 10
    # selected SGD optimization parameters
    std, learning_rate, momentum = 0.1, 1e-3, 0.9

    # load dataset
    train_dataloader, test_dataloader = load_dataset(batch_size)

    # Define NN model
    net = Baseline_Network(D_in, H, D_out)

    # Define Loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # --- train SGD model ---
    # random initialize net
    net.random_init(std=std)

    # Define Optimizer
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

    train_loss_per_epoch, test_loss_per_epoch, train_acc_per_epoch, test_acc_per_epoch = [], [], [], []

    # Train Network
    for epoch in range(epochs):

        running_loss = 0.0

        for i_batch, sample_batched in enumerate(train_dataloader):
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

    np.save("statistics/results/optimization/results/optim_sgd_stats", np.array([train_loss_per_epoch, test_loss_per_epoch, train_acc_per_epoch, test_acc_per_epoch]))
    print()

    # --- train Adam model ---
    # random initialize net
    net.random_init(std=std)

    # Define Optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    train_loss_per_epoch, test_loss_per_epoch, train_acc_per_epoch, test_acc_per_epoch = [], [], [], []

    # Train Network
    for epoch in range(epochs):

        running_loss = 0.0

        for i_batch, sample_batched in enumerate(train_dataloader):
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

    np.save("statistics/results/optimization/results/optim_adam_stats", np.array([train_loss_per_epoch, test_loss_per_epoch, train_acc_per_epoch, test_acc_per_epoch]))
