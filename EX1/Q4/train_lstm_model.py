from dataset_extractor import load_dataset
from lstm_model import LSTM
from statistics.calc_statistics import calc_dataset_loss

import torch
import numpy as np

if __name__ == "__main__":
    epochs = 50
    batch_size = 1
    history = 30
    learning_rate = 1e-3

    # load dataset
    train_dataloader, test_dataloader, train_scaler, test_scaler = load_dataset(batch_size, history)

    # define NN model
    lstm = LSTM()
    lstm = lstm.double()

    # define loss function
    loss_fn = torch.nn.MSELoss()

    # define optimizer
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

    # training procedure
    train_loss_per_epoch = []
    test_loss_per_epoch = []
    for i in range(epochs):
        for i_batch, sample_batched in enumerate(train_dataloader):
            samples, labels = sample_batched
            samples = np.reshape(samples, (history, batch_size, 1))

            # zero the paramaters gradients, and hidden states
            optimizer.zero_grad()
            lstm.hidden_cell = (torch.zeros(1, 1, lstm.hidden_layer_size).double(),
                                torch.zeros(1, 1, lstm.hidden_layer_size).double())

            # forward + backward + optimize
            outputs = lstm(samples.double())
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

        epoch_train_loss = calc_dataset_loss(train_dataloader, lstm, loss_fn, history, batch_size, train_scaler)
        epoch_test_loss = calc_dataset_loss(test_dataloader, lstm, loss_fn, history, batch_size, test_scaler)

        print('[epoch: %d]\ttrain_loss: %.3f\ttest_loss: %.3f' %
              (i+1, epoch_train_loss, epoch_test_loss))

        train_loss_per_epoch.append(epoch_train_loss)
        test_loss_per_epoch.append(epoch_test_loss)

    np.save("statistics/results/lstm/stats", np.array([train_loss_per_epoch, test_loss_per_epoch]))
