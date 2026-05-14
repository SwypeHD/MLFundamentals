from random import random
import numpy as np
import statistics
import math
from numpy import array

# This neural network is structured to identify digits (the classic example) but does not implement many optimizations. True training would likely take hours, and would only be possible with the future usage of stochastic gradient descent to accelerate training. In the meantime It'll be adapted to be a function guesser.
activation_size_1 = 28
activation_size_2 = 16
activation_size_3 = 10
L1_activations = array([random() for _ in range(activation_size_1**2)])  # 728 (27^2)
L2_activations = array([random() for _ in range(16)])  # 16
L3_activations = array([random() for _ in range(16)])  # 16
L4_activations = array([random() for _ in range(10)])  # 10

W1to2 = array(
    [[random() for _ in range(activation_size_1**2)] for _ in range(activation_size_2)]
)
b1to2 = array([random() for _ in range(activation_size_2)])

W2to3 = array(
    [[random() for _ in range(activation_size_2)] for _ in range(activation_size_2)]
)
b2to3 = array([random() for _ in range(activation_size_2)])
W3to4 = array(
    [[random() for _ in range(activation_size_2)] for _ in range(activation_size_3)]
)
b3to4 = array([random() for _ in range(activation_size_3)])


# get the pre-activation value
def get_z_val(weights, previous_activations, biases):
    return weights @ previous_activations + biases


# calculate the sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# returns the derivative of a sigmoid
# utilizes that s' = s(1-s)
def sigmoid_derivative(z):
    a = get_activation(z)
    return a * (1 - a)


# maps sigmoid onto vector
def get_activation(z):
    return np.vectorize(sigmoid)(z)


# calculates the error signal (delta) for respective layers
def calculate_error_signal(weights, comparison_val, z_val):
    sigz = sigmoid_derivative(z_val)
    return (weights.T @ comparison_val) * sigz


# Calculates MSE
def mean_square_error(vals, biases):
    return 1 / len(vals) * sum((vals - biases) ** 2)


test_activation = get_activation(get_z_val(W3to4, L3_activations, b3to4))
print(test_activation)
label_2 = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
print(mean_square_error(test_activation, label_2))
print(len(W3to4))
z = get_z_val(W3to4, L3_activations, b3to4)
# print("Cost MSE", mean_square_error())
learning_rate = 0.01
itr = 0
while mean_square_error(L4_activations, label_2) > 0.001:
    itr += 1
    if itr % 100000 == 0:
        print(f"Iteration {itr}, MSE: {mean_square_error(L4_activations, label_2):.6f}")
    z2 = get_z_val(W1to2, L1_activations, b1to2)
    L2_activations = get_activation(z2)
    z1 = get_z_val(W2to3, L2_activations, b2to3)
    L3_activations = get_activation(z1)
    z = get_z_val(W3to4, L3_activations, b3to4)
    L4_activations = get_activation(z)
    scalar = 1 / len(L4_activations)
    delta_4 = (L4_activations - label_2) * sigmoid_derivative(z)  # shape (10,)
    delta_3 = calculate_error_signal(W3to4, delta_4, z1)  # shape (16,)
    delta_2 = calculate_error_signal(W2to3, delta_3, z2)  # shape (16,)

    # Use each delta with outer product to get weight gradients
    dW3to4 = np.outer(delta_4, L3_activations)
    dW2to3 = np.outer(delta_3, L2_activations)
    dW1to2 = np.outer(delta_2, L1_activations)

    W3to4 -= learning_rate * dW3to4
    W2to3 -= learning_rate * dW2to3
    W1to2 -= learning_rate * dW1to2

    b3to4 -= learning_rate * delta_4
    b2to3 -= learning_rate * delta_3
    b1to2 -= learning_rate * delta_2

print(f"DONE, Took {itr} iterations")
print(W1to2)
