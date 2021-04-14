# Exercise 1 - Q4

## Table of Contents

- [Task](#task)
- [Installation](#installation)
- [Extract Dataset](#extract_dataset)
- [Usage](#usage)
- [Results](#results)

## Task

In this part you are free to use any dataset of your choice. Compare two recurrent architectures,
RNN and LSTM, with hidden state size 100. Demonstrate the vanishing/exploding gradient
problem, explaining your results along with empirical evidence supporting your claims.
We chose a time-series prediction problem of the value of Gold everyday.

## Installation
```sh
git clone https://github.com/AdiAlbum1/foundations-of-deep-learning-course/
cd foundations-of-deep-learning-course/EX1/Q4
pip install -r requirments.txt
```

## Extract Dataset

1. Extract dataset from [Yahoo Finance](https://finance.yahoo.com/quote/GC%3DF?p=GC%3DF). For training set we will use January 1st 2012 to December 31st 2018, and for test set we will use January 1st 2019 to December 31st 2020
2. Name the train dataset's CSV ```train.csv``` and place it as follows:
    ```
    ./dataset/train.csv
    ```
3. Name the test dataset's CSV ```test.csv``` and place it as follows:
    ```
    ./dataset/test.csv
    ```



## Usage
1. Training RNN model:
    ```sh
    python train_rnn_model.py
    ```
2. Training LSTM model:
    ```sh
    python train_lstm_model.py
    ```

## Results
```
TBD
```