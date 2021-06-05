import torch

def calc_dataset_loss(dataset_loader, net, loss_fn):
    running_loss = 0.0
    for sample_batched in dataset_loader:
        batch_x, batch_y = sample_batched

        outputs = net(batch_x)
        loss = loss_fn(outputs, batch_y)
        running_loss += loss.item()

    dataset_loss = running_loss / len(dataset_loader)
    return dataset_loss