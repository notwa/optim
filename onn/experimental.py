import numpy as np

from .float import _f, _0, _1
from .layer import Layer
from .loss import Loss
from .optimizer import Optimizer
from .ritual import Ritual
from .learner import Learner
from .parametric import Dense
from .regularizer import Regularizer


class AddSignClip(Optimizer):
    # paper: https://arxiv.org/abs/1709.07417
    # with heavy-handed gradient clipping of my own concoction.

    def __init__(self, lr=0.01, mu=0.9, alpha=1.0, clip=1.0):
        self.mu = _f(mu)
        self.alpha = _f(alpha)
        self.clip = _f(clip)

        super().__init__(lr)

    def reset(self):
        self.accum = None

    def compute(self, dW, W):
        if self.accum is None:
            self.accum = np.zeros_like(dW)

        self.accum[:] = self.accum * self.mu + dW

        signed = np.sign(dW) * np.sign(self.accum)
        #signed *= decay

        inter = dW * (self.alpha + signed)

        total_norm = np.linalg.norm(inter)
        # based on softplus.
        inter /= np.log(1 + np.exp(total_norm / self.clip - 1)) + 1

        return -self.lr * inter


class PowerSignClip(Optimizer):
    # paper: https://arxiv.org/abs/1709.07417
    # with heavy-handed gradient clipping of my own concoction.

    def __init__(self, lr=0.01, mu=0.9, alpha=np.e, clip=1.0):
        self.mu = _f(mu)
        self.alpha = _f(alpha)
        self.use_exp = np.isclose(self.alpha, _f(np.e))
        self.clip = _f(clip)

        super().__init__(lr)

    def reset(self):
        self.accum = None

    def compute(self, dW, W):
        if self.accum is None:
            self.accum = np.zeros_like(dW)

        self.accum[:] = self.accum * self.mu + dW

        signed = np.sign(dW) * np.sign(self.accum)
        #signed *= decay

        if self.use_exp:
            inter = dW * np.exp(signed)
        else:
            inter = dW * np.power(self.alpha, signed)

        total_norm = np.linalg.norm(inter)
        # based on softplus.
        inter /= np.log(1 + np.exp(total_norm / self.clip - 1)) + 1

        return -self.lr * inter


class L1L2avg(Regularizer):
    def __init__(self, l1=0.0, l2=0.0):
        self.l1 = _f(l1)
        self.l2 = _f(l2)

    def forward(self, X):
        f = _0
        if self.l1:
            f += np.average(self.l1 * np.abs(X))
        if self.l2:
            f += np.average(self.l2 * np.square(X))
        return f

    def backward(self, X):
        df = np.zeros_like(X)
        if self.l1:
            df += self.l1 / len(X) * np.sign(X)
        if self.l2:
            df += self.l2 / len(X) * 2 * X
        return df


class NoiseInjector(Layer):
    def __init__(self, scale=1.0, uniform=False, absolute=False,
                 forwards=True, backwards=False):
        self.scale = _f(scale)
        self.uniform = bool(uniform)
        self.absolute = bool(absolute)
        self.forwards = bool(forwards)
        self.backwards = bool(backwards)

        super().__init__()

    def forward(self, X):
        s = self.scale
        if self.uniform:
            self.noise = np.random.uniform(-s, s, size=X.shape)
        else:
            self.noise = np.random.normal(0, s, size=X.shape)
        if not self.forwards:
            return X
        if self.absolute:
            return X + np.abs(self.noise)
        else:
            return X + self.noise

    def forward_deterministic(self, X):
        return X

    def backward(self, dY):
        if not self.backwards:
            return dY
        if self.absolute:
            return dY + np.abs(self.noise)
        else:
            return dY + self.noise


class NoiseMultiplier(Layer):
    def __init__(self, scale=1.0, uniform=False,
                 forwards=True, backwards=True):
        self.scale = _f(scale)
        self.uniform = bool(uniform)

        super().__init__()

    def forward(self, X):
        s = self.scale
        if self.uniform:
            self.noise = np.exp(np.random.uniform(-s, s, size=X.shape))
        else:
            self.noise = np.exp(np.random.normal(0, s, size=X.shape))
        if not self.forwards:
            return X
        return X * self.noise

    def forward_deterministic(self, X):
        return X

    def backward(self, dY):
        if not self.backwards:
            return dY
        return dY * self.noise


class LookupLearner(Learner):
    per_batch = True

    def __init__(self, optim, epochs=1, rates=(1,), lerp=False):
        self.rates = tuple(rates)
        self.lerp = bool(lerp)
        self.per_batch = self.lerp
        super().__init__(optim, epochs, rates[0])

    def rate_at(self, epoch):
        if self.lerp:
            ind = min(max(int(epoch), 1), len(self.rates) - 1)
            t = _f(epoch % 1)
            a = _f(self.rates[ind-1])
            b = _f(self.rates[ind])
            return (_1 - t) * a + t * b
        else:
            ind = min(int(epoch), len(self.rates) - 1)
            return _f(self.rates[ind])
