import pickle
import random

DATASET_BATCHES = 5
DATASET_DIR = "dataset/cifar-10-batches-py/"
DATASET_PREFIX = "data_batch_"

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def extract_dataset():
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


if __name__ == "__main__":
    # Load whole CIFAR-10 dataset
    all_images, all_labels = extract_dataset()

    # Subsample 10% of CIFAR-10 dataset
    images, labels = subsample_dataset(all_images, all_labels)

    print(len(images))
    print(len(labels))