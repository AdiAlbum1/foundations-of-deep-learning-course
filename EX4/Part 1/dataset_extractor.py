import numpy as np
import pickle
import random
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

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

def normalize_dataset(images):
    images = images / 255.0
    return images


def reorganize_images_shape(images):
    # TO RGB format
    reorganized_images = []
    for image in images:
        reorganized_img = []
        for i in range(32 * 32):
            reorganized_img.append(image[i])
            reorganized_img.append(image[i + 1024])
            reorganized_img.append(image[i + 2 * 1024])

        try:
            img_in_shape = Image.frombytes("RGB", (32, 32), bytes(reorganized_img), "raw")
        except:
            img_in_shape = np.reshape(np.array(reorganized_img), (32, 32, 3))
        reorganized_images.append(np.array(img_in_shape))
    reorganized_images = np.array(reorganized_images)

    return reorganized_images

def adverserial_permutation(label):
    # 0 --> 1, 1 --> 2, 2 --> 3, ... , 8 --> 9, 9 --> 0
    if label < 9:
        new_label = label + 1
    else:
        new_label = 0
    return new_label

def adverserial_labeler(labels):
    adverserial_labels = [0] * len(labels)
    for i, label in enumerate(labels):
        adverserial_labels[i] = adverserial_permutation(label)
    return adverserial_labels


def load_cifar_dataset(batch_size):
    # Unpickle and merge whole CIFAR-10 dataset
    train_images, train_labels = unpickle_and_merge_train_dataset()

    # Unpickle test dataset
    test_images, test_labels = unpickle_test_dataset()

    train_images, test_images = np.array(train_images), np.array(test_images)

    # Reorganize images to RGB format
    train_images = reorganize_images_shape(train_images)
    test_images = reorganize_images_shape(test_images)

    # Normalize images to [0,1] range by dividing by 255.0
    train_images = normalize_dataset(train_images)
    test_images = normalize_dataset(test_images)

    # transform labels to one-hot encoding
    train_labels = to_categorical(train_labels, 10)
    test_labels = to_categorical(test_labels, 10)

    # Organize data in PyTorch DataLoader
    gen = ImageDataGenerator()
    train_generator = gen.flow(train_images, train_labels, batch_size=batch_size, shuffle=True)
    test_generator = gen.flow(test_images, test_labels, batch_size=batch_size, shuffle=True)

    return train_generator, test_generator

def load_random_dataset(batch_size):
    train_images = tf.random.uniform(shape=(50000,32, 32, 3))
    test_images = tf.random.uniform(shape=(10000, 32, 32, 3))

    train_labels = tf.random.uniform(shape=(50000, 1), minval=0, maxval=9, dtype=tf.int32)
    test_labels = tf.random.uniform(shape=(10000, 1), minval=0, maxval=9, dtype=tf.int32)

    # transform labels to one-hot encoding
    train_labels = to_categorical(train_labels, 10)
    test_labels = to_categorical(test_labels, 10)

    # Organize data in PyTorch DataLoader
    gen = ImageDataGenerator()
    train_generator = gen.flow(train_images, train_labels, batch_size=batch_size, shuffle=True)
    test_generator = gen.flow(test_images, test_labels, batch_size=batch_size, shuffle=True)

    return train_generator, test_generator

def load_half_cifar_half_random_dataset(batch_size):
    # Unpickle and merge whole CIFAR-10 dataset
    cifar_train_images, cifar_train_labels = unpickle_and_merge_train_dataset()

    # Unpickle test dataset
    cifar_test_images, cifar_test_labels = unpickle_test_dataset()

    cifar_train_images, cifar_test_images = np.array(cifar_train_images), np.array(cifar_test_images)

    # Reorganize images to RGB format
    cifar_train_images = reorganize_images_shape(cifar_train_images)
    cifar_test_images = reorganize_images_shape(cifar_test_images)

    # Normalize images to [0,1] range by dividing by 255.0
    cifar_train_images = normalize_dataset(cifar_train_images)
    cifar_test_images = normalize_dataset(cifar_test_images)

    # transform labels to one-hot encoding
    cifar_train_labels = to_categorical(cifar_train_labels, 10)
    cifar_test_labels = to_categorical(cifar_test_labels, 10)

    # half dataset
    half_cifar_train_images, half_cifar_train_labels = cifar_train_images[:len(cifar_train_images)//2], cifar_train_labels[:len(cifar_train_labels)//2]
    half_cifar_test_images, half_cifar_test_labels = cifar_test_images[:len(cifar_test_images)//2], cifar_test_labels[:len(cifar_test_labels)//2]

    random_train_images = tf.random.uniform(shape=(25000,32, 32, 3))
    random_test_images = tf.random.uniform(shape=(5000, 32, 32, 3))

    random_train_labels = tf.random.uniform(shape=(25000, 1), minval=0, maxval=9, dtype=tf.int32)
    random_test_labels = tf.random.uniform(shape=(5000, 1), minval=0, maxval=9, dtype=tf.int32)

    # transform labels to one-hot encoding
    random_train_labels = to_categorical(random_train_labels, 10)
    random_test_labels = to_categorical(random_test_labels, 10)

    # merge cifar and random data
    train_images = np.concatenate((half_cifar_train_images, random_train_images))
    train_labels = np.concatenate((half_cifar_train_labels, random_train_labels))
    test_images = np.concatenate((half_cifar_test_images, random_test_images))
    test_labels = np.concatenate((half_cifar_test_labels, random_test_labels))

    # Organize data in PyTorch DataLoader
    gen = ImageDataGenerator()
    train_generator = gen.flow(train_images, train_labels, batch_size=batch_size, shuffle=True)
    test_generator = gen.flow(test_images, test_labels, batch_size=batch_size, shuffle=True)

    return train_generator, test_generator

def load_half_cifar_half_adverserial_dataset(batch_size):
    # Unpickle and merge whole CIFAR-10 dataset
    cifar_train_images, cifar_train_labels = unpickle_and_merge_train_dataset()

    # Unpickle test dataset
    cifar_test_images, cifar_test_labels = unpickle_test_dataset()

    cifar_train_images, cifar_test_images = np.array(cifar_train_images), np.array(cifar_test_images)

    # Reorganize images to RGB format
    cifar_train_images = reorganize_images_shape(cifar_train_images)
    cifar_test_images = reorganize_images_shape(cifar_test_images)

    # Normalize images to [0,1] range by dividing by 255.0
    cifar_train_images = normalize_dataset(cifar_train_images)
    cifar_test_images = normalize_dataset(cifar_test_images)

    # half dataset
    half_cifar_train_images, half_cifar_train_labels = cifar_train_images[:len(cifar_train_images)//2], cifar_train_labels[:len(cifar_train_labels)//2]
    half_cifar_test_images, half_cifar_test_labels = cifar_test_images[:len(cifar_test_images)//2], cifar_test_labels[:len(cifar_test_labels)//2]

    other_half_cifar_train_images, other_half_cifar_train_labels = cifar_train_images[len(cifar_train_images)//2:], cifar_train_labels[len(cifar_train_labels)//2:]
    other_half_cifar_test_images, other_half_cifar_test_labels = cifar_test_images[len(cifar_test_images)//2:], cifar_test_labels[len(cifar_test_labels)//2:]

    # adverserialy label second half of dataset
    other_half_cifar_train_labels = adverserial_labeler(other_half_cifar_train_labels)
    other_half_cifar_test_labels = adverserial_labeler(other_half_cifar_test_labels)

    # merge cifar and adverserial data
    train_images = np.concatenate((half_cifar_train_images, other_half_cifar_train_images))
    train_labels = np.concatenate((half_cifar_train_labels, other_half_cifar_train_labels))
    test_images = np.concatenate((half_cifar_test_images, other_half_cifar_test_images))
    test_labels = np.concatenate((half_cifar_test_labels, other_half_cifar_test_labels))

    # transform labels to one-hot encoding
    train_labels = to_categorical(train_labels, 10)
    test_labels = to_categorical(test_labels, 10)

    # Organize data in PyTorch DataLoader
    gen = ImageDataGenerator()
    train_generator = gen.flow(train_images, train_labels, batch_size=batch_size, shuffle=True)
    test_generator = gen.flow(test_images, test_labels, batch_size=batch_size, shuffle=True)

    return train_generator, test_generator


