import numpy as np

from .float import _f


class Loss:
    def forward(self, p, y):
        raise NotImplementedError("unimplemented", self)

    def backward(self, p, y):
        raise NotImplementedError("unimplemented", self)


class NLL(Loss):  # Negative Log Likelihood
    def forward(self, p, y):
        correct = p * y
        return np.mean(-correct)

    def backward(self, p, y):
        return -y / len(p)


class CategoricalCrossentropy(Loss):
    # lifted from theano

    def __init__(self, eps=1e-6):
        self.eps = _f(eps)

    def forward(self, p, y):
        p = np.clip(p, self.eps, 1 - self.eps)
        f = np.sum(-y * np.log(p) - (1 - y) * np.log(1 - p), axis=-1)
        return np.mean(f)

    def backward(self, p, y):
        p = np.clip(p, self.eps, 1 - self.eps)
        df = (p - y) / (p * (1 - p))
        return df / len(y)


class Accuracy(Loss):
    # returns percentage of categories correctly predicted.
    # utilizes argmax(), so it cannot be used for gradient descent.
    # use CategoricalCrossentropy or NLL for that instead.

    def forward(self, p, y):
        correct = np.argmax(p, axis=-1) == np.argmax(y, axis=-1)
        return np.mean(correct)

    def backward(self, p, y):
        raise NotImplementedError("cannot take the gradient of Accuracy")


class ResidualLoss(Loss):
    def forward(self, p, y):
        return np.mean(self.f(p - y))

    def backward(self, p, y):
        ret = self.df(p - y) / len(y)
        return ret


class SquaredHalved(ResidualLoss):
    def f(self, r):
        return np.square(r) / 2

    def df(self, r):
        return r


class Squared(ResidualLoss):
    def f(self, r):
        return np.square(r)

    def df(self, r):
        return 2 * r


class Absolute(ResidualLoss):
    def f(self, r):
        return np.abs(r)

    def df(self, r):
        return np.sign(r)


class Huber(ResidualLoss):
    def __init__(self, delta=1.0):
        self.delta = _f(delta)

    def f(self, r):
        return np.where(r <= self.delta,
                        np.square(r) / 2,
                        self.delta * (np.abs(r) - self.delta / 2))

    def df(self, r):
        return np.where(r <= self.delta,
                        r,
                        self.delta * np.sign(r))


# more

class SomethingElse(ResidualLoss):
    # generalizes Absolute and SquaredHalved.
    # plot: https://www.desmos.com/calculator/fagjg9vuz7
    def __init__(self, a=4/3):
        assert 1 <= a <= 2, "parameter out of range"
        self.a = _f(a / 2)
        self.b = _f(2 / a)
        self.c = _f(2 / a - 1)

    def f(self, r):
        return self.a * np.abs(r)**self.b

    def df(self, r):
        return np.sign(r) * np.abs(r)**self.c


class Confidence(Loss):
    # this isn't "confidence" in any meaningful way; (e.g. Bayesian)
    # it's just a metric of how large the value is of the predicted class.
    # when using it for loss, it acts like a crappy regularizer.
    # it really just measures how much of a hot-shot the network thinks it is.

    def forward(self, p, y=None):
        categories = p.shape[-1]
        confidence = (np.max(p, axis=-1) - 1/categories) / (1 - 1/categories)
        # the exponent in softmax puts a maximum on confidence,
        # but we don't compensate for that. if necessary,
        # it'd be better to use an activation that doesn't have this limit.
        return np.mean(confidence)

    def backward(self, p, y=None):
        # in order to agree with the forward pass,
        # using this backwards pass as-is will minimize confidence.
        categories = p.shape[-1]
        detc = p / categories / (1 - 1/categories)
        dmax = p == np.max(p, axis=-1, keepdims=True)
        return detc * dmax
