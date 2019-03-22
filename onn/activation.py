import numpy as np

# just for speed, not strictly essential:
from scipy.special import expit as sigmoid

# needed for GELU:
from scipy.special import erf

from .float import _f, _1, _inv2, _invsqrt2, _invsqrt2pi
from .layer_base import *


class Activation(Layer):
    pass


class Identity(Activation):
    def forward(self, X):
        return X

    def backward(self, dY):
        return dY


class Sigmoid(Activation):  # aka Logistic, Expit (inverse of Logit)
    def forward(self, X):
        self.sig = sigmoid(X)
        return self.sig

    def backward(self, dY):
        return dY * self.sig * (1 - self.sig)


class Softplus(Activation):
    # integral of Sigmoid.

    def forward(self, X):
        self.X = X
        return np.log(1 + np.exp(X))

    def backward(self, dY):
        return dY * sigmoid(self.X)


class Tanh(Activation):
    def forward(self, X):
        self.sig = np.tanh(X)
        return self.sig

    def backward(self, dY):
        return dY * (1 - self.sig * self.sig)


class LeCunTanh(Activation):
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


class Relu(Activation):
    def forward(self, X):
        self.cond = X >= 0
        return np.where(self.cond, X, 0)

    def backward(self, dY):
        return np.where(self.cond, dY, 0)


class Elu(Activation):
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


class Swish(Activation):
    # paper: https://arxiv.org/abs/1710.05941
    # the beta parameter here is constant instead of trainable.
    # note that Swish generalizes both SiLU and an approximation of GELU.

    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = _f(scale)

    def forward(self, X):
        self.a = self.scale * X
        self.sig = sigmoid(self.a)
        return X * self.sig

    def backward(self, dY):
        return dY * self.sig * (1 + self.a * (1 - self.sig))


class Silu(Swish):
    # paper: https://arxiv.org/abs/1702.03118
    def __init__(self):
        super().__init__(_1)


class GeluApprox(Swish):
    # paper: https://arxiv.org/abs/1606.08415
    #  plot: https://www.desmos.com/calculator/ydzgtccsld

    def __init__(self):
        super().__init__(_f(1.702))


class Gelu(Activation):
    def forward(self, X):
        self.X = X
        self.cdf = _inv2 * (_1 + erf(X * _invsqrt2))
        return X * self.cdf

    def backward(self, dY):
        return dY * (self.cdf
                     + np.exp(-_inv2 * np.square(self.X))
                     * self.X * _invsqrt2pi)


class Softmax(Activation):
    def forward(self, X):
        # the alpha term is just for numerical stability.
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


class Cos(Activation):
    # performs well on MNIST for some strange reason.

    def forward(self, X):
        self.X = X
        return np.cos(X)

    def backward(self, dY):
        return dY * -np.sin(self.X)


class Selu(Activation):
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

class TanhTest(Activation):
    """preserves the variance of inputs drawn from the standard normal distribution."""
    def forward(self, X):
        self.sig = np.tanh(1 / 2 * X)
        return 2.4004 * self.sig

    def backward(self, dY):
        return dY * (1 / 2 * 2.4004) * (1 - self.sig * self.sig)


class ExpGB(Activation):
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


class CubicGB(Activation):
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


class Arcsinh(Activation):
    def forward(self, X):
        self.X = X
        return np.arcsinh(X)

    def backward(self, dY):
        return dY / np.sqrt(self.X * self.X + 1)


class HardClip(Activation):  # aka HardTanh when at default settings
    def __init__(self, lower=-1.0, upper=1.0):
        super().__init__()
        self.lower = _f(lower)
        self.upper = _f(upper)

    def forward(self, X):
        self.X = X
        return np.clip(X, self.lower, self.upper)

    def backward(self, dY):
        return dY * ((self.X >= self.lower) & (self.X <= self.upper))


class PolyFeat(Layer):
    # i haven't yet decided if this counts as an Activation subclass
    # due to the increased output size, so i'm opting not to inherit it.

    # an incomplete re-implementation of the following, but with gradients:
    # http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
    # i would not recommend using it with input sizes greater than 50.

    def __init__(self, ones_term=True):
        super().__init__()
        self.ones_term = bool(ones_term)

    def make_shape(self, parent):
        shape = parent.output_shape
        assert len(shape) == 1, shape
        self.input_shape = shape
        self.dim = shape[0] + shape[0] * (shape[0] + 1) // 2
        if self.ones_term:
            self.dim += 1
        self.output_shape = (self.dim,)

    def forward(self, X):
        self.X = X
        ones = [np.ones((X.shape[0], 1))] if self.ones_term else []
        return np.concatenate(ones + [X] + [X[:, i][:, None] * X[:, i:]
                              for i in range(X.shape[1])], axis=1)

    def backward(self, dY):
        bp = self.input_shape[0]
        if self.ones_term:
            dY = dY[:, 1:]
        dX = dY[:, :bp].copy()
        rem = dY[:, bp:]

        # TODO: optimize.
        temp = np.zeros((dY.shape[0], bp, bp))
        for i in range(bp):
            temp[:, i, i:] = rem[:, :bp - i]
            rem = rem[:, bp - i:]

        dX += ((temp + temp.transpose(0, 2, 1)) * self.X[:, :, None]).sum(1)
        return dX
