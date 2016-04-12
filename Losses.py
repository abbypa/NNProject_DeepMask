from keras import backend as K


def binary_regression_error(y_true, y_pred):
    return K.log(1 + K.exp(-y_true*y_pred))

# def mean_squared_error(y_true, y_pred):
#     return K.mean(K.square(y_pred - y_true), axis=-1)