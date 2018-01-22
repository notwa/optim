import numpy as np


def rolling(a, window):
    # http://stackoverflow.com/a/4924433
    shape = (a.size - window + 1, window)
    strides = (a.itemsize, a.itemsize)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def rolling_batch(a, window):
    # same as rolling, but acts on each batch (axis 0).
    shape = (a.shape[0], a.shape[-1] - window + 1, window)
    strides = (np.prod(a.shape[1:]) * a.itemsize, a.itemsize, a.itemsize)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
