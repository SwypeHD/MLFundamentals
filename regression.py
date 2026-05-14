# Solving polynomial regression with gradient descent
from random import random
import numpy as np
import statistics
from numpy import array

# lambda x: x * x * x
bases = [lambda x: x**3, lambda x: x**2, lambda x: x, lambda x: 1]
a, b, c, d = tuple(
    np.vectorize(lambda x: np.round(x, 2))([random() * 100 for i in range(4)])
)

true_func = lambda x: 10 * x**2 + 10 * x + 10
true_func = lambda x: a * x**3 + b * x**2 + c * x + d
print(f"The true function is {a}x^3 + {b}x^2 + {c}x + {d}")
x_vals = [6, 5, 4, 3, 2, 1]
x_vals = [float(val) for val in x_vals]
x_vals_mean = statistics.mean(x_vals)
x_vals_std = statistics.stdev(x_vals)
x_vals = array(
    [
        (val - x_vals_mean) / x_vals_std if val - x_vals_mean != 0 else val / x_vals_std
        for val in x_vals
    ]
)
weights = array([random() for i in range(len(bases))])
# print(weights)
feature_map = array([[basis(val) for val in x_vals] for basis in bases]).T
# print("feature map \n", feature_map)
# scaling calculation
# scaling_factors = [
#     statistics.stdev(col) if statistics.stdev(col) != 0 else 1 for col in feature_map.T
# ]
# means = [sum(col) / len(col) for col in feature_map.T]
# scaled_map = feature_map.T
# Shift values by the mean
# scaled_map = array(
#     [
#         (
#             ((scaled_map[i] - means[i]) / scaling_factors[i])
#             if ((scaled_map[i] - means[i]).all() != 0)
#             else scaled_map[i] / scaling_factors[i]
#         )
#         for i in range(len(scaled_map))
#     ]
# ).T


# print(scaled_map)


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
# scaled_vals = array([true_vals[i] / scaling_factors[i] for i in range(len(true_vals))])
# feature_map = scaled_map
eta = 0.001
# print("grad", eta * grad_func(feature_map, weights, true_vals))
while mean_square_error(feature_map, weights, true_vals) > 0.000001:
    weights -= eta * grad_func(feature_map, weights, true_vals)

print(f"The solution is {np.vectorize(lambda x: np.round(x,2))(weights)}")
# Compare this to the closed form:
# Note Iw = y,
# I^T I w = I^T y,
# w = (I^T I)^-1 I^T y
print(
    f"The closed form solution (for comparison) is: {np.vectorize(lambda x: np.round(x,2))(np.linalg.inv((feature_map.T @ feature_map)) @ feature_map.T @ true_vals)}"
)
