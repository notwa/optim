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


class HingeWW(Loss):
    # multi-class hinge-loss, Weston & Watkins variant.

    def forward(self, p, y):
        # TODO: rename score since less is better.
        score = p * (1 - y) - p * y
        return np.mean(np.sum(np.maximum(1 + score, 0), axis=-1))

    def backward(self, p, y):
        score = p * (1 - y) - p * y
        d_score = 1 - y - y
        return (score >= -1) * d_score / len(y)


class HingeCS(Loss):
    # multi-class hinge-loss, Crammer & Singer variant.
    # this has been loosely extended to support multiple true classes.
    # however, it should generally be used such that
    # p is a vector that sums to 1 with values in [0, 1],
    # and y is a one-hot encoding of the correct class.

    def forward(self, p, y):
        wrong = np.max((1 - y) * p, axis=-1)
        right = np.max(y * p, axis=-1)
        f = np.maximum(1 + wrong - right, 0)
        return np.mean(f)

    def backward(self, p, y):
        wrong_in = (1 - y) * p
        right_in = y * p
        wrong = np.max(wrong_in, axis=-1, keepdims=True)
        right = np.max(right_in, axis=-1, keepdims=True)
        # note: this could go haywire if the maximum is not unique.
        delta = (1 - y) * (wrong_in == wrong) - y * (right_in == right)
        return (wrong - right >= -1) * delta / len(y)


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
