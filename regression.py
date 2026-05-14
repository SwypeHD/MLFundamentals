# Solving polynomial regression with gradient descent
from random import random
import numpy as np
from numpy import array

bases = [lambda x: x * x * x, lambda x: x * x, lambda x: x, lambda x: 1]
true_func = lambda x: 3 * x**2 + 2 * x + 1
x_vals = [6, 5, 4, 3, 2, 1]
x_vals = [float(val) for val in x_vals]
weights = array([random() for i in range(len(bases))])
# weights = array([1, 2, 3])
print(weights)
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
eta = 0.001
print("grad", eta * grad_func(feature_map, weights, true_vals))
while mean_square_error(feature_map, weights, true_vals) > 0.000001:
    weights -= eta * grad_func(feature_map, weights, true_vals)

print(f"The solution is {weights}")
# Compare this to the closed form:
# Note Iw = y,
# I^T I w = I^T y,
# w = (I^T I)^-1 I^T y
print(
    f"The closed form solution (for comparison) is: {np.linalg.inv((feature_map.T @ feature_map)) @ feature_map.T @ true_vals}"
)
