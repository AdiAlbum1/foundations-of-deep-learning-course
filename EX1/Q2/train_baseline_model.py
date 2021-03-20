from dataset_extractor import load_dataset
from baseline_model import Baseline_Network

import torch
import numpy as np

from statistics.calc_statistics import calc_dataset_acc, calc_dataset_loss

if __name__ == "__main__":
    # D_in is input dimension; H is hidden dimension; D_out is output dimension.
    epochs = 300
    batch_size, D_in, H, D_out = 64, 3072, 256, 10
    learning_rate, momentum = 1e-4, 0.9

    # load dataset
    train_dataloader, test_dataloader = load_dataset(batch_size)

    # Define NN model
    net = Baseline_Network(D_in, H, D_out)

    # Define Loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # preform grid search
    stds = [0.7, 1.0, 1.3]
    learning_rates = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    momentums = [0.5, 0.9, 0.98]

    for i, std in enumerate(stds):
        for j, learning_rate in enumerate(learning_rates):
            for k, momentum in enumerate(momentums):
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

                    # if epoch % 10 == 0:
                    print('[%d_%d_%d]:\t[epoch %d]\ttrain_loss: %.3f\t test_loss: %.3f\ttrain_acc: %.3f\ttest_acc: %.3f' %
                          (i, j, k, epoch + 1, epoch_train_loss, epoch_test_loss, epoch_train_acc, epoch_test_acc))

                    train_loss_per_epoch.append(epoch_train_loss)
                    test_loss_per_epoch.append(epoch_test_loss)
                    train_acc_per_epoch.append(epoch_train_acc)
                    test_acc_per_epoch.append(epoch_test_acc)

                np.save("statistics/curr_stats_%d_%d_%d" % (i, j, k), np.array([train_loss_per_epoch, test_loss_per_epoch, train_acc_per_epoch, test_acc_per_epoch]))
            print()