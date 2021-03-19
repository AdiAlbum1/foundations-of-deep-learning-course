# Exercise 1 - Q1

## Table of Contents

- [Task](#task)
- [Installation](#installation)
- [Extract Dataset](#extract_dataset)
- [Usage](#usage)
- [Results](#results)

## Task

In this exercise you will experiment with different architectures for image classification. We will
use the CIFAR-10 dataset, which consists of 60000 32x32 color images from 10 classes (6000 images
per class). There are 50000 training images and 10000 test images. In order to reduce computation
time, sample at random a subset of 10% of the original data (i.e. you should have 5000 images
for training and 1000 for test).1 Throughout the rest of the assignment you need only use the
subsampled version.
1. Download and extract CIFAR-10.2 Subsample 10% of the original data as explained above
and normalize the inputs to span the range [0,1].
2. As a baseline, use the sklearn python package to implement an SVM classifier. For both the
linear and RBF kernels, report the train and test accuracies obtained.

## Installation
```sh
git clone https://github.com/AdiAlbum1/foundations-of-deep-learning-course/
cd foundations-of-deep-learning-course/EX1/Q1
pip install -r requirments.txt
```

## Extract Dataset

1. [Download dataset](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)
2. Untar dataset and place it in [dataset folder](./dataset) with following structure:
```
./dataset/cifar-10-batches-py/...
```

## Usage
```sh
python classify.py
```

## Results
| Method        | Test Accuracy           |
| :-------------: |:-------------:|
| Linear SVM      | 29.8% |
| RBF SVM      | 44.6%      |