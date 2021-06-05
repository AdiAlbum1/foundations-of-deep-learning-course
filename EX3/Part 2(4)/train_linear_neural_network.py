import torch
import numpy as np

from dataset.dataset_extractor import load_dataset
import dataset.config as config

from linear_nueral_network_model import Linear_Network

from statistics.calc_dataset_loss import calc_dataset_loss
from statistics.calc_weights_gradients_magnitude import calc_weights_gradients_magnitude
from statistics.calc_min_and_max_eigenval_of_hessian import calc_min_and_max_eigenval_of_hessian


if __name__ == "__main__":
    epochs = 100
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
    weights_gradient_magnitude_per_epoch = []
    min_eigenvalue_per_epoch = []
    max_eigenvalue_per_epoch = []

    for epoch in range(epochs):
        for sample_batched in train_dataloader:
            batch_x, batch_y = sample_batched

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = linear_net(batch_x)
            batch_y = torch.reshape(batch_y, (len(batch_y), 1))
            outputs = torch.reshape(outputs, (len(outputs), 1))
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()

        # calculate train & test loss
        epoch_train_loss = calc_dataset_loss(train_dataloader, linear_net, loss_fn)
        epoch_test_loss = calc_dataset_loss(test_dataloader, linear_net, loss_fn)

        train_loss_per_epoch.append(epoch_train_loss)
        test_loss_per_epoch.append(epoch_test_loss)

        # calculate weights gradient magnitude
        curr_weights_gradient_magnitude = calc_weights_gradients_magnitude(linear_net)
        weights_gradient_magnitude_per_epoch.append(curr_weights_gradient_magnitude)

        # calculate min and max eigenvalue of hessian
        min_eigenval, max_eigenval = calc_min_and_max_eigenval_of_hessian(linear_net, loss_fn, train_dataloader, N)
        min_eigenvalue_per_epoch.append(min_eigenval)
        max_eigenvalue_per_epoch.append(max_eigenval)

        print('[epoch %d]\ttrain_loss: %.3f\t test_loss: %.3f' %
              (epoch + 1, epoch_train_loss, epoch_test_loss))

    np.save("statistics/results/train_stats", np.array([train_loss_per_epoch, test_loss_per_epoch]))
    np.save("statistics/results/train_weights_gradient_magnitude", np.array(weights_gradient_magnitude_per_epoch))
    np.save("statistics/results/min_eigenval", np.array(min_eigenvalue_per_epoch))
    np.save("statistics/results/max_eigenval", np.array(max_eigenvalue_per_epoch))
