import torch

def calc_dataset_loss(dataset_loader, net, loss_fn):
    running_loss = 0.0
    for sample_batched in dataset_loader:
        batch_x, batch_y = sample_batched

        batch_x = torch.reshape(batch_x, (len(batch_x), 1))
        batch_y = torch.reshape(batch_y, (len(batch_y), 1))

        outputs = net(batch_x.float())
        loss = loss_fn(outputs, batch_y.float())
        running_loss += loss.item()

    dataset_loss = running_loss / len(dataset_loader)
    return dataset_loss