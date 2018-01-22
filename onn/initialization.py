import numpy as np

# note: these are currently only implemented for 2D shapes.


def init_zeros(size, ins=None, outs=None):
    return np.zeros(size)


def init_ones(size, ins=None, outs=None):
    return np.ones(size)


def init_he_normal(size, ins, outs):
    s = np.sqrt(2 / ins)
    return np.random.normal(0, s, size=size)


def init_he_uniform(size, ins, outs):
    s = np.sqrt(6 / ins)
    return np.random.uniform(-s, s, size=size)


def init_glorot_normal(size, ins, outs):
    s = np.sqrt(2 / (ins + outs))
    return np.random.normal(0, s, size=size)


def init_glorot_uniform(size, ins, outs):
    s = np.sqrt(6 / (ins + outs))
    return np.random.uniform(-s, s, size=size)


# more

def init_gaussian_unit(size, ins, outs):
    s = np.sqrt(1 / ins)
    return np.random.normal(0, s, size=size)
