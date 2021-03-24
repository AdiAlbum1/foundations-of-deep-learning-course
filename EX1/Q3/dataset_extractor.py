import numpy as np
import pickle
import random
import torch

from PIL import Image

DATASET_BATCHES = 5
DATASET_DIR = "dataset/cifar-10-batches-py/"
TRAIN_DATASET_PREFIX = "data_batch_"
TEST_DATASET = "test_batch"

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def unpickle_and_merge_train_dataset():
    all_images = []
    all_labels = []
    for i in range(1, DATASET_BATCHES + 1):
        curr_dataset_batch = unpickle(DATASET_DIR + TRAIN_DATASET_PREFIX + str(i))
        all_images.extend(curr_dataset_batch[b'data'])
        all_labels.extend(curr_dataset_batch[b'labels'])

    return all_images, all_labels

def unpickle_test_dataset():
    test_dataset = unpickle(DATASET_DIR + TEST_DATASET)
    test_images = test_dataset[b'data']
    test_labels = test_dataset[b'labels']

    return test_images, test_labels


def subsample_dataset(images, labels):
    # Simultaneously shuffle images and labels
    combined = list(zip(images, labels))
    random.shuffle(combined)
    images, labels = zip(*combined)

    # Subsample 10%
    new_size = len(images) // 10

    subsampled_images = images[:new_size]
    subsampled_labels = labels[:new_size]

    return subsampled_images, subsampled_labels

def normalize_dataset(images):
    images = images / 255.0
    # images = [image / 255.0 for image in images]
    return images

def reorganize_images_shape(images):
    # TO RGB format
    reorganized_images = []
    for image in images:
        reorganized_img = []
        for i in range(32*32):
            reorganized_img.append(image[i])
            reorganized_img.append(image[i + 1024])
            reorganized_img.append(image[i + 2*1024])

        img_in_shape = Image.frombytes("RGB", (32,32), bytes(reorganized_img), "raw")
        reorganized_images.append(np.array(img_in_shape))
    reorganized_images = np.array(reorganized_images)

    return reorganized_images

def load_dataset(batch_size):
    # Unpickle and merge whole CIFAR-10 dataset
    all_train_images, all_train_labels = unpickle_and_merge_train_dataset()

    # Unpickle test dataset
    all_test_images, all_test_labels = unpickle_test_dataset()

    # Subsample 10% of CIFAR-10 dataset
    train_images, train_labels = subsample_dataset(all_train_images, all_train_labels)
    test_images, test_labels = subsample_dataset(all_test_images, all_test_labels)

    # Reorganize images to RGB format
    train_images = reorganize_images_shape(train_images)
    test_images = reorganize_images_shape(test_images)

    # Normalize images to [0,1] range by dividing by 255.0
    train_images = normalize_dataset(train_images)
    test_images = normalize_dataset(test_images)

    # Organize data in PyTorch DataLoader
    tensor_train_x, tensor_train_y = torch.tensor(train_images), torch.tensor(train_labels)
    tensor_train = torch.utils.data.TensorDataset(tensor_train_x, tensor_train_y)
    train_dataloader = torch.utils.data.DataLoader(tensor_train, batch_size=batch_size, shuffle=True)

    tensor_test_x, tensor_test_y = torch.tensor(test_images), torch.tensor(test_labels)
    tensor_test = torch.utils.data.TensorDataset(tensor_test_x, tensor_test_y)
    test_dataloader = torch.utils.data.DataLoader(tensor_test, batch_size=batch_size)

    return train_dataloader, test_dataloader