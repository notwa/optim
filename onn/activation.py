import numpy as np

# just for speed, not strictly essential:
from scipy.special import expit as sigmoid

from .float import *
from .layer_base import *


class Identity(Layer):
    def forward(self, X):
        return X

    def backward(self, dY):
        return dY


class Sigmoid(Layer):  # aka Logistic, Expit (inverse of Logit)
    def forward(self, X):
        self.sig = sigmoid(X)
        return self.sig

    def backward(self, dY):
        return dY * self.sig * (1 - self.sig)


class Softplus(Layer):
    # integral of Sigmoid.

    def forward(self, X):
        self.X = X
        return np.log(1 + np.exp(X))

    def backward(self, dY):
        return dY * sigmoid(self.X)


class Tanh(Layer):
    def forward(self, X):
        self.sig = np.tanh(X)
        return self.sig

    def backward(self, dY):
        return dY * (1 - self.sig * self.sig)


class LeCunTanh(Layer):
    # paper: http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
    # paper: http://yann.lecun.com/exdb/publis/pdf/lecun-89.pdf
    # scaled such that f([-1, 1]) = [-1, 1].
    # helps preserve an input variance of 1.
    # second derivative peaks around an input of ±1.

    def forward(self, X):
        self.sig = np.tanh(2 / 3 * X)
        return 1.7159 * self.sig

    def backward(self, dY):
        return dY * (2 / 3 * 1.7159) * (1 - self.sig * self.sig)


class Relu(Layer):
    def forward(self, X):
        self.cond = X >= 0
        return np.where(self.cond, X, 0)

    def backward(self, dY):
        return np.where(self.cond, dY, 0)


class Elu(Layer):
    # paper: https://arxiv.org/abs/1511.07289

    def __init__(self, alpha=1):
        super().__init__()
        self.alpha = _f(alpha)  # FIXME: unused

    def forward(self, X):
        self.cond = X >= 0
        self.neg = np.exp(X) - 1
        return np.where(self.cond, X, self.neg)

    def backward(self, dY):
        return dY * np.where(self.cond, 1, self.neg + 1)


class GeluApprox(Layer):
    # paper: https://arxiv.org/abs/1606.08415
    #  plot: https://www.desmos.com/calculator/ydzgtccsld

    def forward(self, X):
        self.a = 1.704 * X
        self.sig = sigmoid(self.a)
        return X * self.sig

    def backward(self, dY):
        return dY * self.sig * (1 + self.a * (1 - self.sig))


class Softmax(Layer):
    def forward(self, X):
        alpha = np.max(X, axis=-1, keepdims=True)
        num = np.exp(X - alpha)
        den = np.sum(num, axis=-1, keepdims=True)
        self.sm = num / den
        return self.sm

    def backward(self, dY):
        return (dY - np.sum(dY * self.sm, axis=-1, keepdims=True)) * self.sm


class LogSoftmax(Softmax):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = _f(eps)

    def forward(self, X):
        return np.log(super().forward(X) + self.eps)

    def backward(self, dY):
        return dY - np.sum(dY, axis=-1, keepdims=True) * self.sm


class Cos(Layer):
    # performs well on MNIST for some strange reason.

    def forward(self, X):
        self.X = X
        return np.cos(X)

    def backward(self, dY):
        return dY * -np.sin(self.X)


class Selu(Layer):
    # paper: https://arxiv.org/abs/1706.02515

    def __init__(self, alpha=1.67326324, lamb=1.05070099):
        super().__init__()
        self.alpha = _f(alpha)
        self.lamb = _f(lamb)

    def forward(self, X):
        self.cond = X >= 0
        self.neg = self.alpha * np.exp(X)
        return self.lamb * np.where(self.cond, X, self.neg - self.alpha)

    def backward(self, dY):
        return dY * self.lamb * np.where(self.cond, 1, self.neg)


# more

class TanhTest(Layer):
    def forward(self, X):
        self.sig = np.tanh(1 / 2 * X)
        return 2.4004 * self.sig

    def backward(self, dY):
        return dY * (1 / 2 * 2.4004) * (1 - self.sig * self.sig)


class ExpGB(Layer):
    # an output layer for one-hot classification problems.
    # use with MSE (SquaredHalved), not CategoricalCrossentropy!
    # paper: https://arxiv.org/abs/1707.04199

    def __init__(self, alpha=0.1, beta=0.0):
        super().__init__()
        self.alpha = _f(alpha)
        self.beta = _f(beta)

    def forward(self, X):
        return self.alpha * np.exp(X) + self.beta

    def backward(self, dY):
        # this gradient is intentionally incorrect.
        return dY


class CubicGB(Layer):
    # an output layer for one-hot classification problems.
    # use with MSE (SquaredHalved), not CategoricalCrossentropy!
    # paper: https://arxiv.org/abs/1707.04199
    # note: in the paper, it's called pow3GB, which is ugly.

    def __init__(self, alpha=0.1, beta=0.0):
        # note: the paper suggests defaults of 0.001 and 0.0,
        # but these didn't seem to work as well in my limited testing.
        super().__init__()
        self.alpha = _f(alpha)
        self.beta = _f(beta)

    def forward(self, X):
        return self.alpha * X**3 + self.beta

    def backward(self, dY):
        # this gradient is intentionally incorrect.
        return dY
