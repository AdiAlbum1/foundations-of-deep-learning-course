# Exercise 3 - Part 3 - Question 4

## Table of Contents

- [Task](#task)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Results](#results)

## Task

On a scalar regression dataset of your choice, train a depth
N linear neural network (with hidden width no smaller than the minimum between input
dimension and output dimension) by minimizing l2 loss via (full batch) gradient descent with
small learning rate and initialization close to zero. Compare the trajectory taken by the endto-end matrix to that obtained by directly applying the (discrete version of the) end-to-end
dynamics to a linear model:
<br/>
<img src="equation.png" width="400" height="200">
<br/>

## Installation
```sh
git clone https://github.com/AdiAlbum1/foundations-of-deep-learning-course/
cd foundations-of-deep-learning-course/EX1/Part 3(4)
pip install -r requirments.txt
```

## Dataset

Dataset is a noisy sample of trigonometric tan(x)
<br/>
<img src="dataset/dataset.png" width="800" height="300">
<br/>

## Usage
1. Training linear neural network:
    ```sh
    python train_linear_neural_network.py
    ```
2. For visualizing training procedure's train and test loss, run:
    ```sh
    python statistics/visualize_loss.py
    ```
3. For visualizing the weights' gradient magnitude throughout training, run:
    ```sh
    python statistics/visualize_weights_gradient_magnitude.py
    ```
4. For visualizing the weights' gradient Hessian min and max eigenvalue, run:
    ```sh
    python statistics/visualize_min_and_max_hessian_eigenvals.py
    ```

## Results
1. Training procedure - Train and Test Loss:<br/>
    <br/>
    <img src="statistics/results/train_loss.png" width="400" height="400">
    <img src="statistics/results/test_loss.png" width="400" height="400">
    <br/>

2. Weights' gradient magnitude<br/>
    <br/>
    <img src="statistics/results/weights_gradient_magnitude.png" width="400" height="400">
    <br/>

3. Weights' gradient Hessian min and max eigenvalue<br/>
    <br/>
    <img src="statistics/results/min_eigenvals.png" width="400" height="400">
    <img src="statistics/results/max_eigenvals.png" width="400" height="400">
    <br/>