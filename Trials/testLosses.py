import numpy as np
from Losses import *
from ImageUtils import *

ones = np.array([[1]])
zeros = np.array([[0]])

print 'Score loss:'

a = binary_regression_error(ones, ones)
b = binary_regression_error(ones, zeros)

print 'Expecting ~%s - got %s' % (0.3 * score_output_lambda,  K.eval(a))
print 'Expecting ~%s - got %s' % (0.69 * score_output_lambda,  K.eval(b))

mask_ones = np.array([[[1,1],[1,1]]])

c = binary_regression_error(mask_ones, mask_ones)
print 'Expecting ~%s - got %s' % (0.3 * score_output_lambda,  K.eval(c))

print ''
print 'Mask loss:'
d = mask_binary_regression_error(mask_ones, mask_ones)
print 'Expecting 0 - got %s' % K.eval(d)

mask_neg_ones = np.array([[[-1,-1],[-1,-1]]])
e = mask_binary_regression_error(mask_neg_ones, mask_ones)
print 'Expecting ~%s - got %s' % (1.3 * seg_output_lambda,  K.eval(e))

f = mask_binary_regression_error(mask_neg_ones, mask_neg_ones)
print 'Expecting ~%s - got %s' % (0.3 * seg_output_lambda,  K.eval(f))
print ''

expected_mask = np.array([prepare_expected_mask('../Resources/LossTest/expected_mask.png')])
good_mask = np.array([prepare_expected_mask('../Resources/LossTest/good_mask.png')])
bad_mask = np.array([prepare_expected_mask('../Resources/LossTest/bad_mask.png')])
very_bad_mask = np.array([prepare_expected_mask('../Resources/LossTest/very_bad_mask.png')])
no_mask = np.array([prepare_expected_mask('../Resources/LossTest/no_mask.png')])

# expected low
print 'low losses:'
print K.eval(mask_binary_regression_error(expected_mask, expected_mask))
print K.eval(mask_binary_regression_error(expected_mask, good_mask))
# expected high
print 'high losses:'
print K.eval(mask_binary_regression_error(expected_mask, bad_mask))
print K.eval(mask_binary_regression_error(expected_mask, very_bad_mask))
# expected zero
print 'zero losses:'
print K.eval(mask_binary_regression_error(no_mask, good_mask))
print K.eval(mask_binary_regression_error(no_mask, bad_mask))
