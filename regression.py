from random import random
import numpy as np
from numpy import array

bases = [lambda x: x * x, lambda x: x, lambda x: 1]
x_vals = [6, 5, 4, 3, 2, 1]
weights = [random() * len(bases)]
# print(bases[1](2))
feature_map = array([[basis(val) for val in x_vals] for basis in bases])
print(feature_map.T)
# Iw = y

while 
