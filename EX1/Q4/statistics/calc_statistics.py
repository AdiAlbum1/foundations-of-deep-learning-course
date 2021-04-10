import torch
import numpy as np

def calc_dataset_loss(dataset_loader, net, loss_fn, history, batch_size):
    running_loss = 0.0
    for i_batch, sample_batched in enumerate(dataset_loader):
        samples, labels = sample_batched
        samples = np.reshape(samples, (history, batch_size, 1))

        net.hidden_cell = (torch.zeros(1, 1, net.hidden_layer_size).double(),
                            torch.zeros(1, 1, net.hidden_layer_size).double())
        outputs = net(samples.double())
        loss = loss_fn(outputs, labels)
        running_loss += loss.item()

    dataset_loss = running_loss / len(dataset_loader)
    return dataset_loss