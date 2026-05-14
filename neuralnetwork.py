from random import random
import numpy as np
import statistics
import math
from numpy import array

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


def cost_function_derivative(activations, target, z, last_activations):
    sigz = sigmoid_derivative(z)
    # print("--- Cost Function Derivative Debug ---")
    # print(f"Activations shape: {activations.shape}, values: {activations}")
    # print(f"Bias (target label) shape: {np.array(bias).shape}, values: {bias}")
    # print(f"Sigmoid derivative (sigz) shape: {sigz.shape}")
    # print(f"Last activations shape: {last_activations.shape}")
    # print("-------------------------------------")
    return (
        1 / len(activations) * np.outer((activations - target) * sigz, last_activations)
    )


def calculate_error_signal(weights, comparison_val, z_val):
    sigz = sigmoid_derivative(z_val)
    return (weights.T @ comparison_val) * sigz


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

    delta_4 = (L4_activations - label_2) * sigmoid_derivative(z)  # shape (10,)
    delta_3 = calculate_error_signal(W3to4, delta_4, z1)  # shape (16,)
    delta_2 = calculate_error_signal(W2to3, delta_3, z2)  # shape (16,)

    # Use each delta with outer product to get weight gradients
    # dW3to4 = np.outer(delta_4, L3_activations)
    dW2to3 = np.outer(delta_3, L2_activations)
    dW1to2 = np.outer(delta_2, L1_activations)

    W3to4_cost_prime = cost_function_derivative(
        L4_activations, label_2, z, L3_activations
    )
    W3to4 -= learning_rate * W3to4_cost_prime
    W2to3 -= learning_rate * dW2to3
    W1to2 -= learning_rate * dW1to2

    b3to4 -= learning_rate * delta_4
    b2to3 -= learning_rate * delta_3
    b1to2 -= learning_rate * delta_2

print("DONE")
print(W1to2)
