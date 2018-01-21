import numpy as np

from .float import *
from .initialization import *
from .ritual_base import *

def stochastic_multiply(W, gamma=0.5, allow_negation=False):
    # paper: https://arxiv.org/abs/1606.01981

    assert W.ndim == 1, W.ndim
    assert 0 < gamma < 1, gamma
    size = len(W)
    alpha = np.max(np.abs(W))
    # NOTE: numpy gives [low, high) but the paper advocates [low, high]
    mult = np.random.uniform(gamma, 1/gamma, size=size)
    if allow_negation:
        # NOTE: i have yet to see this do anything but cause divergence.
        # i've referenced the paper several times yet still don't understand
        # what i'm doing wrong, so i'm disabling it by default in my code.
        # maybe i just need *a lot* more weights to compensate.
        prob = (W / alpha + 1) / 2
        samples = np.random.random_sample(size=size)
        mult *= np.where(samples < prob, 1, -1)
    np.multiply(W, mult, out=W)

class StochMRitual(Ritual):
    # paper: https://arxiv.org/abs/1606.01981
    # this probably doesn't make sense for regression problems,
    # let alone small models, but here it is anyway!

    def __init__(self, learner=None, gamma=0.5):
        super().__init__(learner)
        self.gamma = _f(gamma)

    def prepare(self, model):
        self.W = np.copy(model.W)
        super().prepare(model)

    def learn(self, inputs, outputs):
        # an experiment:
        #assert self.learner.rate < 10, self.learner.rate
        #self.gamma = 1 - 1/2**(1 - np.log10(self.learner.rate))

        self.W[:] = self.model.W
        for layer in self.model.ordered_nodes:
            if isinstance(layer, Dense):
                stochastic_multiply(layer.coeffs.ravel(), gamma=self.gamma)
        residual = super().learn(inputs, outputs)
        self.model.W[:] = self.W
        return residual

    def update(self):
        super().update()
        f = 0.5
        for layer in self.model.ordered_nodes:
            if isinstance(layer, Dense):
                np.clip(layer.W, -layer.std * f, layer.std * f, out=layer.W)
            #   np.clip(layer.W, -1, 1, out=layer.W)

class NoisyRitual(Ritual):
    def __init__(self, learner=None,
                 input_noise=0, output_noise=0, gradient_noise=0):
        self.input_noise = _f(input_noise)
        self.output_noise = _f(output_noise)
        self.gradient_noise = _f(gradient_noise)
        super().__init__(learner)

    def learn(self, inputs, outputs):
        # this is pretty crude
        if self.input_noise > 0:
            s = self.input_noise
            inputs =   inputs + np.random.normal(0, s, size=inputs.shape)
        if self.output_noise > 0:
            s = self.output_noise
            outputs = outputs + np.random.normal(0, s, size=outputs.shape)
        return super().learn(inputs, outputs)

    def update(self):
        # gradient noise paper: https://arxiv.org/abs/1511.06807
        if self.gradient_noise > 0:
            size = len(self.model.dW)
            gamma = 0.55
            #s = self.gradient_noise / (1 + self.bn) ** gamma
            # experiments:
            s = self.gradient_noise * np.sqrt(self.learner.rate)
            #s = np.square(self.learner.rate)
            #s = self.learner.rate / self.en
            self.model.dW += np.random.normal(0, max(s, 1e-8), size=size)
        super().update()

