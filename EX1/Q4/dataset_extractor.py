import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def scale_dataset(train_dataset, test_dataset):
    scaler = MinMaxScaler(feature_range=(0,1))

    train_dataset_scaled = scaler.fit_transform(train_dataset)
    test_dataset_scaled = scaler.transform(test_dataset)

    return train_dataset_scaled, test_dataset_scaled

def load_dataset(batch_size, history_length):
    # Read data from CSV
    train_dataset_complete = pd.read_csv(r"./dataset/train.csv")
    test_dataset_complete = pd.read_csv(r"./dataset/test.csv")

    # Read 'open' column only
    train_dataset_processed = train_dataset_complete.iloc[:, 1:2].values
    test_dataset_processed = test_dataset_complete.iloc[:, 1:2].values

    # Scale dataset to (0,1) range
    train_dataset_scaled, test_dataset_scaled = scale_dataset(train_dataset_processed, test_dataset_processed)

    # Reorganize data to right shape
    train_samples, train_labels = [], []
    for i in range(history_length, len(train_dataset_scaled)):
        train_samples.append(train_dataset_scaled[i-history_length:i, 0])
        train_labels.append(train_dataset_scaled[i, 0])

    test_samples, test_labels = [], []
    for i in range(history_length, len(test_dataset_scaled)):
        test_samples.append(test_dataset_scaled[i-history_length:i, 0])
        test_labels.append(test_dataset_scaled[i, 0])

    # Organize data in PyTorch DataLoader
    tensor_train_x, tensor_train_y = torch.tensor(train_samples), torch.tensor(train_labels)
    tensor_train = torch.utils.data.TensorDataset(tensor_train_x, tensor_train_y)
    train_dataloader = torch.utils.data.DataLoader(tensor_train, batch_size=batch_size, shuffle=True)

    tensor_test_x, tensor_test_y = torch.tensor(test_samples), torch.tensor(test_labels)
    tensor_test = torch.utils.data.TensorDataset(tensor_test_x, tensor_test_y)
    test_dataloader = torch.utils.data.DataLoader(tensor_test, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader


if __name__ == "__main__":
    train_dataloader, test_dataloader = load_dataset(4, 4)

    print(len(train_dataloader))
    print(len(test_dataloader))