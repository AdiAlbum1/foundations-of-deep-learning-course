from dataset_extractor import load_dataset
from conv_model import Conv_Network

import torch
import numpy as np

from statistics.calc_statistics import calc_dataset_acc, calc_dataset_loss

if __name__ == "__main__":
    epochs = 200
    batch_size = 32

    # load dataset
    train_dataloader, test_dataloader = load_dataset(batch_size)

    # Define NN model
    conv_net = Conv_Network()

    # Define Loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # perform grid search
    stds = [0.05, 0.1, 0.2]
    learning_rates = [1e-3, 1e-2, 1e-1]
    momentums = [0, 0.5, 0.9]

    for i, std in enumerate(stds):
        for j, learning_rate in enumerate(learning_rates):
            for k, momentum in enumerate(momentums):
                # random initialize net
                conv_net.random_init(std=std)

                # Define Optimizer
                optimizer = torch.optim.SGD(conv_net.parameters(), lr=learning_rate, momentum=momentum)

                train_loss_per_epoch, test_loss_per_epoch, train_acc_per_epoch, test_acc_per_epoch = [], [], [], []

                # Train Network
                for epoch in range(epochs):

                    running_loss = 0.0

                    for i_batch, sample_batched in enumerate(train_dataloader):
                        images, labels = sample_batched

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward + backward + optimize
                        outputs = conv_net(images.float())
                        loss = loss_fn(outputs, labels)
                        loss.backward()
                        optimizer.step()

                    # calculate train & test loss
                    epoch_train_loss = calc_dataset_loss(train_dataloader, conv_net, loss_fn)
                    epoch_test_loss = calc_dataset_loss(test_dataloader, conv_net, loss_fn)
                    # calculate train & test accuracy
                    epoch_train_acc = calc_dataset_acc(train_dataloader, conv_net)
                    epoch_test_acc = calc_dataset_acc(test_dataloader, conv_net)

                    print('[%d_%d_%d]:\t[epoch %d]\ttrain_loss: %.3f\t test_loss: %.3f\ttrain_acc: %.2f%%\ttest_acc: %.2f%%' %
                          (i, j, k, epoch + 1, epoch_train_loss, epoch_test_loss, 100*epoch_train_acc, 100*epoch_test_acc))

                    train_loss_per_epoch.append(epoch_train_loss)
                    test_loss_per_epoch.append(epoch_test_loss)
                    train_acc_per_epoch.append(epoch_train_acc)
                    test_acc_per_epoch.append(epoch_test_acc)

                np.save("statistics/results/baseline/results/curr_stats_%d_%d_%d" % (i, j, k), np.array([train_loss_per_epoch, test_loss_per_epoch, train_acc_per_epoch, test_acc_per_epoch]))
                print()