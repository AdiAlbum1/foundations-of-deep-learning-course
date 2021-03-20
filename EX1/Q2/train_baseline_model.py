from dataset_extractor import load_dataset
from baseline_model import Baseline_Network

import torch
import numpy as np

from statistics import calc_dataset_acc, calc_dataset_loss

if __name__ == "__main__":

    # batch_size is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    epochs = 5
    batch_size, D_in, H, D_out = 64, 3072, 256, 10
    learning_rate, momentum = 1e-4, 0.9

    # load dataset
    train_images, train_labels, test_images, test_labels = load_dataset()

    tensor_train_x, tensor_train_y = torch.tensor(train_images), torch.tensor(train_labels)
    tensor_train = torch.utils.data.TensorDataset(tensor_train_x, tensor_train_y)
    train_dataloader = torch.utils.data.DataLoader(tensor_train, batch_size=batch_size, shuffle=True)

    tensor_test_x, tensor_test_y = torch.tensor(test_images), torch.tensor(test_labels)
    tensor_test = torch.utils.data.TensorDataset(tensor_test_x, tensor_test_y)
    test_dataloader = torch.utils.data.DataLoader(tensor_test)

    # Define NN model
    net = Baseline_Network(D_in, H, D_out)
    net.random_init()

    # Define Loss function
    loss_fn = torch.nn.CrossEntropyLoss()

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

        print('[%d]\ttrain_loss: %.3f\t test_loss: %.3f\ttrain_acc: %.3f\ttest_acc: %.3f' %
              (epoch + 1, epoch_train_loss, epoch_test_loss, epoch_train_acc, epoch_test_acc))

        train_loss_per_epoch.append(epoch_train_loss)
        test_loss_per_epoch.append(epoch_test_loss)
        train_acc_per_epoch.append(epoch_train_acc)
        test_acc_per_epoch.append(epoch_test_acc)

    np.save("curr_stats", np.array([train_loss_per_epoch, test_loss_per_epoch, train_acc_per_epoch, test_acc_per_epoch]))