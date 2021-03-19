import numpy as np
import pickle
import random

DATASET_BATCHES = 5
DATASET_DIR = "dataset/cifar-10-batches-py/"
DATASET_PREFIX = "data_batch_"

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def unpickle_and_merge_dataset():
    all_images = []
    all_labels = []
    for i in range(1, DATASET_BATCHES + 1):
        curr_dataset_batch = unpickle(DATASET_DIR + DATASET_PREFIX + str(i))
        all_images.extend(curr_dataset_batch[b'data'])
        all_labels.extend(curr_dataset_batch[b'labels'])

    return all_images, all_labels

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
    normalized_images = [image / 255.0 for image in images]
    return normalized_images

def load_dataset():
    # Unpickle and merge whole CIFAR-10 dataset
    all_images, all_labels = unpickle_and_merge_dataset()

    # Subsample 10% of CIFAR-10 dataset
    images, labels = subsample_dataset(all_images, all_labels)

    # Normalize images to [0,1] range by dividing by 255.0
    images = normalize_dataset(images)

    return images, labels