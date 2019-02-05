import numpy as np

from .float import _f


class Optimizer:
    def __init__(self, lr=0.1):
        self.lr = _f(lr)  # learning rate
        self.base_rate = self.lr
        self.reset()

    def reset(self):
        self.lr = self.base_rate

    def compute(self, dW, W):
        return -self.lr * dW

    def update(self, dW, W):
        W += self.compute(dW, W)
