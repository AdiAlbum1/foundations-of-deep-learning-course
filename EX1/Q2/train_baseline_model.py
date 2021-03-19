from dataset_extractor import load_dataset
from baseline_model import Baseline_Network

import torch
import numpy as np

if __name__ == "__main__":

    # batch_size is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    epochs = 30
    batch_size, D_in, H, D_out = 64, 3072, 256, 10

    # load dataset
    train_images, train_labels, test_images, test_labels = load_dataset()

    tensor_train_x, tensor_train_y = torch.tensor(train_images), torch.tensor(train_labels)
    tensor_train = torch.utils.data.TensorDataset(tensor_train_x, tensor_train_y)

    train_dataloader = torch.utils.data.DataLoader(tensor_train, batch_size=batch_size)

    tensor_test_x, tensor_test_y = torch.tensor(test_images), torch.tensor(test_labels)
    tensor_test = torch.utils.data.TensorDataset(tensor_test_x, tensor_test_y)

    test_dataloader = torch.utils.data.DataLoader(tensor_test)

    # Create random Tensors to hold inputs and outputs
    x = torch.randn(batch_size, D_in)
    y = torch.randn(batch_size, D_out)

    # Define NN model
    net = Baseline_Network(D_in, H, D_out)

    # Define Loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # Define Optimizer
    learning_rate = 1e-4
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

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

        # print epoch statistics
        num_train_batches = 0
        running_loss = 0.0
        for i_batch, sample_batched in enumerate(train_dataloader): # TODO add validation
            images, labels = sample_batched
            num_train_batches += 1

            outputs = net(images.float())
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()

        epoch_loss = running_loss / num_train_batches
        print('[%d] train_loss: %.3f' %
              (epoch + 1, epoch_loss))