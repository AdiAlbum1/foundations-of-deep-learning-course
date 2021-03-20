import torch

def calc_dataset_acc(dataset_loader, net):
    # calculate train loss
    dataset_size = 0
    num_of_success_predictions = 0
    for i_batch, sample_batched in enumerate(dataset_loader):
        images, labels = sample_batched
        outputs = net(images.float())
        prediction = torch.argmax(outputs, dim=1)
        diff_tensor = prediction - labels
        num_of_success_predictions_in_batch = len(labels) - torch.count_nonzero(diff_tensor)

        dataset_size += len(labels)
        num_of_success_predictions += num_of_success_predictions_in_batch

    acc = num_of_success_predictions / dataset_size
    return acc

def calc_dataset_loss(dataset_loader, net, loss_fn):
    # calculate train loss
    running_loss = 0.0
    for i_batch, sample_batched in enumerate(dataset_loader):
        images, labels = sample_batched
        outputs = net(images.float())
        loss = loss_fn(outputs, labels)
        running_loss += loss.item()

    dataset_loss = running_loss / len(dataset_loader)
    return dataset_loss