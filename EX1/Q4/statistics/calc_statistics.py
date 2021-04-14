import torch
import numpy as np

def calc_dataset_loss(dataset_loader, net, loss_fn, history, batch_size, scaler):
    running_loss = 0.0
    for i_batch, sample_batched in enumerate(dataset_loader):
        samples, labels = sample_batched
        samples = np.reshape(samples, (history, batch_size, 1))

        net.hidden_cell = (torch.zeros(1, 1, net.hidden_layer_size).double(),
                            torch.zeros(1, 1, net.hidden_layer_size).double())
        outputs = net(samples.double())

        outputs = torch.tensor(scaler.inverse_transform(outputs.detach().numpy().reshape(-1, 1))[0])
        labels = torch.tensor(scaler.inverse_transform(labels.detach().numpy().reshape(-1, 1))[0])

        loss = loss_fn(outputs, labels)
        running_loss += loss.item()

    dataset_loss = running_loss / len(dataset_loader)
    return dataset_loss