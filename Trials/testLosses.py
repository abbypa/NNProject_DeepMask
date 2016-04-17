import numpy as np
from Losses import *

ones = np.array([[1]])
zeros = np.array([[0]])

a = binary_regression_error(ones, ones)
b = binary_regression_error(ones, zeros)

print K.eval(a)  # expecting ~0.3
print K.eval(b)  # expecting ~0.69

mask_ones = np.array([[[[1,1],[1,1]]]])

c = binary_regression_error(mask_ones[0], mask_ones[0])
print K.eval(c)  # expecting an array of scores of ~0.3

d = mask_binary_regression_error(mask_ones[0], mask_ones[0])
print K.eval(d)  # expecting 0- first item is 0

mask_zeros = np.array([[[[0,0],[0,0]]]])
e = mask_binary_regression_error(mask_zeros[0], mask_ones[0])
print K.eval(e)  # expecting ~0.69
