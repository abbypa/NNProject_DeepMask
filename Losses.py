from keras import backend as K


def binary_regression_error(y_true, y_pred):
    return K.log(1 + K.exp(-y_true*y_pred))


def mask_binary_regression_error(y_true, y_pred):
    return (1-y_true[0][0][0]) * K.mean(K.log(1 + K.exp(-y_true*y_pred)))
