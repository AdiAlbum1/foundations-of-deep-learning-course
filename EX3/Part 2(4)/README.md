# Exercise 3 - Part 2 - Question 4

## Table of Contents

- [Task](#task)
- [Installation](#installation)
- [Extract Dataset](#extract_dataset)
- [Usage](#usage)
- [Results](#results)

## Task

On a regression dataset of your choice, train a depth N linear
neural network (with hidden widths no smaller than the minimum between input dimension
and output dimension) by minimizing l2 loss via (full batch) gradient descent with small
learning rate and initialization. Plot the objective value, the magnitude of its gradient, and
the maximal and minimal eigenvalues of its Hessian throughout the run. Explain your results.
Repeat the experiment with depths N=2,3,4


## Installation
```sh
git clone https://github.com/AdiAlbum1/foundations-of-deep-learning-course/
cd foundations-of-deep-learning-course/EX1/Part 2(4)
pip install -r requirments.txt
```

## Dataset

Dataset is a noisy sample of trigonometric tan(x)
<br/>
<img src="dataset/dataset.png" width="400" height="400">
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
    python statistics/visualize_weights_gradient_magnitude.py.py
    ```
4. For visualizing weights' gradients Hessian min and max eigenvalue, run:
    ```sh
    python statistics/visualize_min_and_max_hessian_eigenvals.py
    ```

## Results
1. Training procedure - Train and Test Loss:<br/>
    <br/>
    <img src="statistics/results/train_loss.png" width="400" height="400">
    <img src="statistics/results/test_loss.png" width="400" height="400">
    <br/>

2. Optimizer Comparison<br/>
    For optimizer comparison we compared SGD with above hyperparameters and Adam optimizer.
    This gave us the following results:
    <br/>
    <img src="statistics/results/optimization/loss.png" width="400" height="400">
    <img src="statistics/results/optimization/acc.png" width="400" height="400">
    <br/>

3. Initialization Technique Comparison<br/>
    For initialization technique comparison we compared random normal initialization with Xavier initialization.
    This gave us the following results:
    <br/>
    <img src="statistics/results/initialization/loss.png" width="400" height="400">
    <img src="statistics/results/initialization/acc.png" width="400" height="400">
    <br/>

4. Regularizer Comparison<br/>
    For comparison of different regularization techniques: weight_decay and dropout.
    This gave us the following results: <br/>
    weight_decay:
    <br/>
    <img src="statistics/results/regularization/loss_weight_decay.png" width="400" height="400">
    <img src="statistics/results/regularization/acc_weight_decay.png" width="400" height="400">
    <br/>
    dropout:
    <br/>
    <img src="statistics/results/regularization/loss_dropout_prob.png" width="400" height="400">
    <img src="statistics/results/regularization/acc_dropout_prob.png" width="400" height="400">
    <br/>

5. Preprocessing Comparison<br/>
    For comparison of different preprocessing techniques: no-preprocessing vs whitening.
    This gave us the following results:
    <br/>
    <img src="statistics/results/preprocessing/loss.png" width="400" height="400">
    <img src="statistics/results/preprocessing/acc.png" width="400" height="400">
    <br/>

6. Network Widths Comparison<br/>
    For comparison of affect network width has on training.
    This gave us the following results:
    <br/>
    <img src="statistics/results/widths/loss.png" width="400" height="400">
    <img src="statistics/results/widths/acc.png" width="400" height="400">
    <br/>

7. Network Depths Comparison<br/>
    For comparison of affect network depth has on training.
    This gave us the following results:
    <br/>
    <img src="statistics/results/depths/loss.png" width="400" height="400">
    <img src="statistics/results/depths/acc.png" width="400" height="400">
    <br/>