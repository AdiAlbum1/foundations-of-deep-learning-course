def calc_dataset_loss(dataset_loader, net, loss_fn):
    running_loss = 0.0
    for i_batch, sample_batched in enumerate(dataset_loader):
        images, labels = sample_batched
        outputs = net(images.float())
        loss = loss_fn(outputs, labels)
        running_loss += loss.item()

    dataset_loss = running_loss / len(dataset_loader)
    return dataset_loss