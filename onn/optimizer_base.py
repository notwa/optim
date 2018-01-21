import numpy as np

from .floats import *

class Optimizer:
    def __init__(self, lr=0.1):
        self.lr = _f(lr) # learning rate
        self.reset()

    def reset(self):
        pass

    def compute(self, dW, W):
        return -self.lr * dW

    def update(self, dW, W):
        W += self.compute(dW, W)


