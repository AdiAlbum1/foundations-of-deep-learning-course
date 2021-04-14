from dataset_extractor import load_dataset
from conv_model import Conv_Network

import torch
import numpy as np

from statistics.calc_statistics import calc_dataset_acc, calc_dataset_loss

def train_network_with_optimizer(net, optimizer, loss_func, epochs, train_data, test_data, reg_kind, iter_num):
  train_loss_per_epoch, test_loss_per_epoch, train_acc_per_epoch, test_acc_per_epoch = [], [], [], []
  for epoch in range(epochs):
      running_loss = 0.0
      for i_batch, sample_batched in enumerate(train_data):
        images, labels = sample_batched
        images, labels = images.to(torch.device("cuda:0")), labels.to(torch.device("cuda:0"))
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(images.float())
        loss = loss_func(outputs, labels).cuda()
        loss.backward()
        optimizer.step()

      # calculate train & test loss
      epoch_train_loss = calc_dataset_loss(train_data, net, loss_func)
      epoch_test_loss = calc_dataset_loss(test_data, net, loss_func)
      # calculate train & test accuracy
      epoch_train_acc = calc_dataset_acc(train_data, net)
      epoch_test_acc = calc_dataset_acc(test_data, net)

      print('[epoch %d]\ttrain_loss: %.3f\t test_loss: %.3f\ttrain_acc: %.2f%%\ttest_acc: %.2f%%' %
            (epoch + 1, epoch_train_loss, epoch_test_loss, 100*epoch_train_acc, 100*epoch_test_acc))

      train_loss_per_epoch.append(epoch_train_loss)
      test_loss_per_epoch.append(epoch_test_loss)
      train_acc_per_epoch.append(epoch_train_acc)
      test_acc_per_epoch.append(epoch_test_acc)
  np.save(f"statistics/results/regularization/results/regularization_{reg_kind}_stats_{iter_num}", np.array([train_loss_per_epoch, test_loss_per_epoch, train_acc_per_epoch, test_acc_per_epoch]))
  print()


if __name__ == "__main__":
  
    epochs = 100
    # selected optimization parameters
    std, learning_rate, momentum = 0.1, 1e-3, 0.9

    batch_size = 32
    train_data, test_data = load_dataset(batch_size)

    # Define CNN model
    net = Conv_Network()

    # Define Loss function
    loss_func = torch.nn.CrossEntropyLoss()

    # --- train SGD with weight decay ---
    weight_decay_vals = [0, 0.01, 0.03]
    for i, weight_decay in enumerate(weight_decay_vals):
      net.random_init(std=std)  # init the params randomly
      sgd_optimizer = torch.optim.SGD(net.parameters(),
                                     lr=learning_rate,
                                     momentum=momentum,
                                     weight_decay=weight_decay)

      train_network_with_optimizer(net, sgd_optimizer, loss_func, epochs,
                                   train_data, test_data, "weight_decay", i)


    # # --- train new with dropout ---
    
    p_dropouts = [0, 0.3, 0.5]

    for i, p in enumerate(p_dropouts):
      net = Conv_Network(p_dropout=p)
      net.random_init(std=std)  # init the params randomly
      sgd_optimizer = torch.optim.SGD(net.parameters(),
                                     lr=learning_rate,
                                     momentum=momentum)

      train_network_with_optimizer(net, sgd_optimizer, loss_func, epochs,
                                   train_data, test_data, "dropout", i)
