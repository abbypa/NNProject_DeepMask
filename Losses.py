from keras import backend as K
from Constants import score_output_lambda, seg_output_lambda


def binary_regression_error(y_true, y_pred):
    return score_output_lambda * K.log(1 + K.exp(-y_true*y_pred))


def mask_binary_regression_error(y_true, y_pred):
    # upper left is -1 (background- legal centered mask)- multiply by 1
    # upper left is 1 (illegal centered mask)- return 0
    return seg_output_lambda * 0.5 * (1 - y_true[0][0][0]) * K.mean(K.log(1 + K.exp(-y_true*y_pred)))
