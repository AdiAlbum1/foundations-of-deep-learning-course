import torch
import numpy as np

from dataset.dataset_extractor import load_dataset
import dataset.config as config
from linear_nueral_network_model import Linear_Network
from statistics.calc_dataset_loss import calc_dataset_loss


if __name__ == "__main__":
    epochs = 200
    N = 3                                   # Net depth
    input_dim, output_dim = config.INPUT_DIM, config.OUTPUT_DIM
    hidden_dims = 1
    weight_init_std = 1.7
    learning_rate = 1e-2

    # load dataset
    train_dataloader, test_dataloader = load_dataset()

    # define nn
    linear_net = Linear_Network(input_dim, hidden_dims, output_dim, N)

    # define Loss function
    loss_fn = torch.nn.MSELoss()

    # randomly initialize network weights
    linear_net.normal_random_init(std=weight_init_std)
    # linear_net.normal_random_init()

    # define optimizer
    optimizer = torch.optim.SGD(linear_net.parameters(), lr=learning_rate)

    # train network
    train_loss_per_epoch, test_loss_per_epoch, train_acc_per_epoch, test_acc_per_epoch = [], [], [], []

    print(linear_net)

    for epoch in range(epochs):
        for sample_batched in train_dataloader:
            batch_x, batch_y = sample_batched

            batch_x = torch.reshape(batch_x, (len(batch_x), 1))
            batch_y = torch.reshape(batch_y, (len(batch_y), 1))

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = linear_net(batch_x.float())
            loss = loss_fn(outputs, batch_y.float())
            loss.backward()
            optimizer.step()

            # print(list(linear_net.parameters()))

        # calculate train & test loss
        epoch_train_loss = calc_dataset_loss(train_dataloader, linear_net, loss_fn)
        epoch_test_loss = calc_dataset_loss(test_dataloader, linear_net, loss_fn)
        print('[epoch %d]\ttrain_loss: %.3f\t test_loss: %.3f' %
              (epoch + 1, epoch_train_loss, epoch_test_loss))

        train_loss_per_epoch.append(epoch_train_loss)
        test_loss_per_epoch.append(epoch_test_loss)

    np.save("statistics/results/linear_nn_train_stats", np.array([train_loss_per_epoch, test_loss_per_epoch]))
