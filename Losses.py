from keras import backend as K


def binary_regression_error(y_true, y_pred):
    return K.log(1 + K.exp(-y_true*y_pred))


def mask_binary_regression_error(y_true, y_pred):
    # Previous trials:
    #
    # y_true_score = y_true[0]
    # y_true_mask = y_true[1]
    # return 0.5 * (1+y_true_score) * K.mean(binary_regression_error(y_true_mask, y_pred))
    # return 0.5 * (1+y_true[0]) * K.mean(K.log(1 + K.exp(-y_true[1]*y_pred)))
    # return K.mean(binary_regression_error(y_true[1], y_pred))
    # return K.mean(K.log(1 + K.exp(-y_true*y_pred))) #OK
    # return K.mean(binary_regression_error(y_true, y_pred)) # ??
    # return 0.5 * (1+y_true[0][0][0]) * K.mean(K.log(1 + K.exp(-y_true*y_pred)))

    # expecting first cell to have 0 for a centered mask (no corner pixel), and 1 otherwise
    return (1-y_true[0][0][0]) * K.mean(K.log(1 + K.exp(-y_true*y_pred)))
