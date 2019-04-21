import numpy as np


def relu(x):
    return np.maximum(x, 0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def prelu(x, a=None):

    if a is not None:

        y = x.copy()
        index = x < 0
        y[index] = a * x[index]

        return y

    else:

        raise NotImplementedError


def elu(x, a=None):

    if a is not None:

        y = x.copy()
        index = x < 0
        y[index] = a * (np.exp(x[index]) - 1)

        return y

    else:

        raise NotImplementedError
