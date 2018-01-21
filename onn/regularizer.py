import numpy as np

from .floats import *

class Regularizer:
    pass

class L1L2(Regularizer):
    def __init__(self, l1=0.0, l2=0.0):
        self.l1 = _f(l1)
        self.l2 = _f(l2)

    def forward(self, X):
        f = _0
        if self.l1:
            f += np.sum(self.l1 * np.abs(X))
        if self.l2:
            f += np.sum(self.l2 * np.square(X))
        return f

    def backward(self, X):
        df = np.zeros_like(X)
        if self.l1:
            df += self.l1 * np.sign(X)
        if self.l2:
            df += self.l2 * 2 * X
        return df

# more

class SaturateRelu(Regularizer):
    # paper: https://arxiv.org/abs/1703.09202
    # TODO: test this (and ActivityRegularizer) more thoroughly.
    #       i've looked at the histogram of the resulting weights.
    #       it seems like only the layers after this are affected
    #       the way they should be.

    def __init__(self, lamb=0.0):
        self.lamb = _f(lamb)

    def forward(self, X):
        return self.lamb * np.where(X >= 0, X, 0)

    def backward(self, X):
        return self.lamb * np.where(X >= 0, 1, 0)
