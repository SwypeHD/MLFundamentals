# Solving polynomial regression with gradient descent
from random import random
import numpy as np
import statistics
from numpy import array
import math

bases = [lambda x: x**3, lambda x: x**2, lambda x: x, lambda x: 1]
a, b, c, d = tuple(
    np.vectorize(lambda x: np.round(x, 2))([random() * 1000 for i in range(4)])
)

# Defines the cubic and standardizes input vals
true_func = lambda x: a * x**3 + b * x**2 + c * x + d
# true_func = lambda x: math.sin(x)
print(f"The true function is {a}x^3 + {b}x^2 + {c}x + {d}")
# x_vals = [6, 5, 4, 3, 2, 1, -6, -2, -4, -8, -10, 3]
x_vals = [3, 2, 1, 0, -1, -2, -3]
x_vals = [float(val) for val in x_vals]
x_vals_mean = statistics.mean(x_vals)
x_vals_std = statistics.stdev(x_vals)
x_vals = array(
    [
        (val - x_vals_mean) / x_vals_std if val - x_vals_mean != 0 else val / x_vals_std
        for val in x_vals
    ]
)
# set random weights
weights = array([random() for i in range(len(bases))])
# designate feature map based off basis
feature_map = array([[basis(val) for val in x_vals] for basis in bases]).T


# We note that I (feature map) times w (the weight vector) - y (bias) should be minimize
# We measure this with mean square error
def mean_square_error(feature_map, weights, true_vals):
    num_entries = len(x_vals)
    result = feature_map @ weights.T - true_vals
    total = 0
    for val in result:
        total += val**2
    return 1 / num_entries * (total)


# we use gradient descent to minimize the weights
# min MSE = 1/n * 2 (Iw - y) I(x)
def grad_func(feature_map, weights, true_vals):
    result = feature_map @ weights.T - true_vals
    num_entries = len(x_vals)
    sum = array([0.0 for i in range(len(bases))])
    for i in range(len(feature_map)):
        sum += result[i] * feature_map[i]
    return 2 / num_entries * sum


true_vals = [true_func(val) for val in x_vals]
eta = 0.0001
counter = 0
error_threshhold = 0.00001
while mean_square_error(feature_map, weights, true_vals) > error_threshhold:
    # way to track regressions with over 100,000 runs
    counter += 1
    if counter % 100000 == 0:
        print(weights)
    # adjust the weights based off eta * grad
    weights -= eta * grad_func(feature_map, weights, true_vals)

print(f"The solution is {np.vectorize(lambda x: np.round(x,2))(weights)}")
# Compare this to the closed form:
# Note Iw = y,
# I^T I w = I^T y,
# w = (I^T I)^-1 I^T y
print(
    f"The closed form solution (for comparison) is: {np.vectorize(lambda x: np.round(x,2))(np.linalg.inv((feature_map.T @ feature_map)) @ feature_map.T @ true_vals)}"
)
