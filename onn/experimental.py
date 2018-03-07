from .float import *
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
