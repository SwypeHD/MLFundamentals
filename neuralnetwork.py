from random import random
import numpy as np
import statistics
import math
from numpy import array

activation_size_1 = 27
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
# activation_size_3 = 10
# W4 = array([[random() for _ in range(activation_size_3)] for _ in range(activation_size_3)])
# b4 = array([[random() for _ in range(activation_size_3)] for _ in range(activation_size_3)])


# get the pre-activation value
def get_z_val(weights, previous_activations, biases):
    return weights @ previous_activations + biases


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    a = get_activation(z)
    return a * (1 - a)


def get_activation(z):
    return np.vectorize(sigmoid)(z)


def cost_function_derivative(activations, bias, z, last_activations):
    sigz = sigmoid_derivative(z)
    print(
        "Activations \n",
        activations,
        "Bias \n",
        bias,
        "sigz \n",
        sigz,
        "last_activations \n",
        last_activations,
    )
    return (
        2 / len(activations) * np.outer((activations - bias) * sigz, last_activations)
    )


def mean_square_error(vals, biases):
    return 1 / len(vals) * sum((vals - biases) ** 2)


test_activation = get_activation(get_z_val(W3to4, L3_activations, b3to4))
print(test_activation)
label_2 = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
print(mean_square_error(test_activation, label_2))
print(len(W3to4))
z = get_z_val(W3to4, L3_activations, b3to4)
cost_derivative = cost_function_derivative(L4_activations, b3to4, z, L3_activations)
print(cost_derivative)
# print("Cost MSE", mean_square_error())
learning_rate = 0.01
while mean_square_error(L4_activations, label_2) > 0.001:
    z = get_z_val(W3to4, L3_activations, b3to4)
    z1 = get_z_val(W2to3, L2_activations, b2to3)
    z2 = get_z_val(W1to2, L1_activations, b1to2)
    L2_activations = get_activation(z2)
    L3_activations = get_activation(z1)
    L4_activations = get_activation(z)
    W3to4_cost_prime = cost_function_derivative(
        L4_activations, b3to4, z, L3_activations
    )
    W2to3_cost_prime = cost_function_derivative(
        L3_activations, b2to3, z1, L2_activations
    )
    W1to2_cost_prime = cost_function_derivative(
        L2_activations, b2to3, z2, L1_activations
    )
    W3to4 -= learning_rate * W3to4_cost_prime
    W2to3 -= learning_rate * W2to3_cost_prime
    W1to2 -= learning_rate * W1to2_cost_prime
