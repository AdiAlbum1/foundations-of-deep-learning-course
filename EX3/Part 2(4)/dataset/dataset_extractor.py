import torch
import numpy as np
import random

import dataset.config as config

def split_train_test(data_x, data_y):
    combined = list(zip(data_x, data_y))
    random.shuffle(combined)
    data_x, data_y = zip(*combined)

    train_x, train_y = data_x[:int(config.TRAIN_RATIO * len(data_x))], data_y[:int(config.TRAIN_RATIO * len(data_y))]
    test_x, test_y = data_x[int(config.TRAIN_RATIO * len(data_x)):], data_y[int(config.TRAIN_RATIO * len(data_y)):]

    return (train_x, train_y), (test_x, test_y)

def normalize_dataset(data_x):
    data_x = data_x / max(data_x)
    return data_x

def read_dataset(path):
    data = np.load(path)
    data_x, data_y = data
    return data_x, data_y

def load_dataset():
    # read dataset from file
    data_x, data_y = read_dataset(config.DATASET_PATH)

    # normalize dataset
    data_x = normalize_dataset(data_x)

    # split to train and test
    (train_x, train_y), (test_x, test_y) = split_train_test(data_x, data_y)

    # Organize data in PyTorch DataLoader
    tensor_train_x, tensor_train_y = torch.tensor(train_x), torch.tensor(train_y)
    tensor_train = torch.utils.data.TensorDataset(tensor_train_x, tensor_train_y)
    train_dataloader = torch.utils.data.DataLoader(tensor_train, batch_size=len(tensor_train))

    tensor_test_x, tensor_test_y = torch.tensor(test_x), torch.tensor(test_y)
    tensor_test = torch.utils.data.TensorDataset(tensor_test_x, tensor_test_y)
    test_dataloader = torch.utils.data.DataLoader(tensor_test, batch_size=len(tensor_test))

    return train_dataloader, test_dataloader