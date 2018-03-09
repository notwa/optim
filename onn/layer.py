from .layer_base import *
from .initialization import *
from .float import *
from .regularizer import Regularizer


# Nonparametric Layers {{{1

class Input(Layer):
    def __init__(self, shape, gradient=False):
        assert shape is not None
        super().__init__()
        self.shape = tuple(shape)
        self.input_shape = self.shape
        self.output_shape = self.shape
        self.gradient = bool(gradient)

    def forward(self, X):
        return X

    def backward(self, dY):
        if self.gradient:
            return dY
        else:
            return np.zeros_like(dY)


class Reshape(Layer):
    def __init__(self, new_shape):
        super().__init__()
        self.shape = tuple(new_shape)
        self.output_shape = self.shape

    def forward(self, X):
        self.batch_size = X.shape[0]
        return X.reshape(self.batch_size, *self.output_shape)

    def backward(self, dY):
        assert dY.shape[0] == self.batch_size
        return dY.reshape(self.batch_size, *self.input_shape)


class Flatten(Layer):
    def make_shape(self, parent):
        shape = parent.output_shape
        self.input_shape = shape
        self.output_shape = (np.prod(shape),)

    def forward(self, X):
        self.batch_size = X.shape[0]
        return X.reshape(self.batch_size, *self.output_shape)

    def backward(self, dY):
        assert dY.shape[0] == self.batch_size
        return dY.reshape(self.batch_size, *self.input_shape)


class ConstAffine(Layer):
    def __init__(self, a=1, b=0):
        super().__init__()
        self.a = _f(a)
        self.b = _f(b)

    def forward(self, X):
        return self.a * X + self.b

    def backward(self, dY):
        return dY * self.a


class Sum(Layer):
    def _propagate(self, edges, deterministic):
        return np.sum(edges, axis=0)

    def _backpropagate(self, edges):
        # assert len(edges) == 1, "unimplemented"
        return edges[0]  # TODO: does this always work?


class ActivityRegularizer(Layer):
    def __init__(self, reg):
        super().__init__()
        assert isinstance(reg, Regularizer), reg
        self.reg = reg

    def forward(self, X):
        self.X = X
        self.loss = np.sum(self.reg.forward(X))
        return X

    def backward(self, dY):
        return dY + self.reg.backward(self.X)


class Dropout(Layer):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.p = _f(1 - dropout)
        assert 0 <= self.p <= 1

    def forward(self, X):
        self.mask = (np.random.rand(*X.shape) < self.p) / self.p
        return X * self.mask

    def forward_deterministic(self, X):
        # self.mask = _1
        return X

    def backward(self, dY):
        return dY * self.mask


# more

class AlphaDropout(Layer):
    # to be used alongside Selu activations.
    # paper: https://arxiv.org/abs/1706.02515

    def __init__(self, dropout=0.0, alpha=1.67326324, lamb=1.05070099):
        super().__init__()
        self.alpha = _f(alpha)
        self.lamb = _f(lamb)
        self.saturated = -self.lamb * self.alpha
        self.dropout = _f(dropout)

    @property
    def dropout(self):
        return self._dropout

    @dropout.setter
    def dropout(self, x):
        self._dropout = _f(x)
        self.q = 1 - self._dropout
        assert 0 <= self.q <= 1

        sat = self.saturated

        self.a = 1 / np.sqrt(self.q + sat * sat * self.q * self._dropout)
        self.b = -self.a * (self._dropout * sat)

    def forward(self, X):
        self.mask = np.random.rand(*X.shape) < self.q
        return self.a * np.where(self.mask, X, self.saturated) + self.b

    def forward_deterministic(self, X):
        return X

    def backward(self, dY):
        return dY * self.a * self.mask


class Decimate(Layer):
    # simple decimaton layer that drops every other sample from the last axis.

    def __init__(self, phase='even'):
        super().__init__()
        # phase is the set of samples we keep in the forward pass.
        assert phase in ('even', 'odd'), phase
        self.phase = phase

    def make_shape(self, parent):
        shape = parent.output_shape
        self.input_shape = shape
        divy = (shape[-1] + 1) // 2 if self.phase == 'even' else shape[-1] // 2
        self.output_shape = tuple(list(shape[:-1]) + [divy])
        self.dX = np.zeros(self.input_shape, dtype=_f)

    def forward(self, X):
        self.batch_size = X.shape[0]
        if self.phase == 'even':
            return X.ravel()[0::2].reshape(self.batch_size, *self.output_shape)
        elif self.phase == 'odd':
            return X.ravel()[1::2].reshape(self.batch_size, *self.output_shape)

    def backward(self, dY):
        assert dY.shape[0] == self.batch_size
        dX = np.zeros((self.batch_size, *self.input_shape), dtype=_f)
        if self.phase == 'even':
            dX.ravel()[0::2] = dY.ravel()
        elif self.phase == 'odd':
            dX.ravel()[1::2] = dY.ravel()
        return dX


class Undecimate(Layer):
    # inverse operation of Decimate. not quite interpolation.

    def __init__(self, phase='even'):
        super().__init__()
        # phase is the set of samples we keep in the backward pass.
        assert phase in ('even', 'odd'), phase
        self.phase = phase

    def make_shape(self, parent):
        shape = parent.output_shape
        self.input_shape = shape
        mult = shape[-1] * 2
        self.output_shape = tuple(list(shape[:-1]) + [mult])

    def forward(self, X):
        self.batch_size = X.shape[0]
        Y = np.zeros((self.batch_size, *self.output_shape), dtype=_f)
        if self.phase == 'even':
            Y.ravel()[0::2] = X.ravel()
        elif self.phase == 'odd':
            Y.ravel()[1::2] = X.ravel()
        return Y

    def backward(self, dY):
        assert dY.shape[0] == self.batch_size
        if self.phase == 'even':
            return dY.ravel()[0::2].reshape(self.batch_size, *self.input_shape)
        elif self.phase == 'odd':
            return dY.ravel()[1::2].reshape(self.batch_size, *self.input_shape)
