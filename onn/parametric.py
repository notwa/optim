import numpy as np

from .float import _f
from .layer_base import *
from .initialization import *


class Bias(Layer):
    # TODO: support axes other than -1 and shapes other than 1D.

    serialized = {
        'b': 'biases',
    }

    def __init__(self, init=init_zeros, reg_b=None):
        super().__init__()
        self.biases = self._new_weights('biases', init=init, regularizer=reg_b)

    def make_shape(self, parent):
        shape = parent.output_shape
        self.input_shape = shape
        self.output_shape = shape
        self.biases.shape = (shape[-1],)

    def forward(self, X):
        return X + self.biases.f

    def backward(self, dY):
        self.biases.g += dY.sum(0)
        return dY


class Dense(Layer):
    serialized = {
        'W': 'coeffs',
        'b': 'biases',
    }

    def __init__(self, dim, init=init_he_uniform, reg_w=None, reg_b=None):
        super().__init__()
        self.dim = int(dim)
        self.output_shape = (dim,)
        self.coeffs = self._new_weights('coeffs', init=init,
                                        regularizer=reg_w)
        self.biases = self._new_weights('biases', init=init_zeros,
                                        regularizer=reg_b)

    def make_shape(self, parent):
        shape = parent.output_shape
        self.input_shape = shape
        assert len(shape) == 1, shape
        self.coeffs.shape = (shape[0], self.dim)
        self.biases.shape = (1, self.dim)

    def forward(self, X):
        self.X = X
        return X @ self.coeffs.f + self.biases.f

    def backward(self, dY):
        self.coeffs.g += self.X.T @ dY
        self.biases.g += dY.sum(0, keepdims=True)
        return dY @ self.coeffs.f.T


class DenseUnbiased(Layer):
    serialized = {
        'W': 'coeffs',
    }

    def __init__(self, dim, init=init_he_uniform, reg_w=None):
        super().__init__()
        self.dim = int(dim)
        self.output_shape = (dim,)
        self.coeffs = self._new_weights('coeffs', init=init,
                                        regularizer=reg_w)

    def make_shape(self, parent):
        shape = parent.output_shape
        self.input_shape = shape
        assert len(shape) == 1, shape
        self.coeffs.shape = (shape[0], self.dim)

    def forward(self, X):
        self.X = X
        return X @ self.coeffs.f

    def backward(self, dY):
        self.coeffs.g += self.X.T @ dY
        return dY @ self.coeffs.f.T


# more

class Conv1Dper(Layer):
    # periodic (circular) convolution.
    # currently only supports one channel I/O.
    # some notes:
    # we could use FFTs for larger convolutions.
    # i think storing the coefficients backwards would
    # eliminate reversal in the critical code.

    serialize = {
        'W': 'coeffs',
    }

    def __init__(self, kernel_size, pos=None,
                 init=init_glorot_uniform, reg_w=None):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.coeffs = self._new_weights('coeffs', init=init, regularizer=reg_w)
        if pos is None:
            self.wrap0 = (self.kernel_size - 0) // 2
            self.wrap1 = (self.kernel_size - 1) // 2
        elif pos == 'alt':
            self.wrap0 = (self.kernel_size - 1) // 2
            self.wrap1 = (self.kernel_size - 0) // 2
        elif pos == 'left':
            self.wrap0 = 0
            self.wrap1 = self.kernel_size - 1
        elif pos == 'right':
            self.wrap0 = self.kernel_size - 1
            self.wrap1 = 0
        else:
            raise Exception("pos parameter not understood: {}".format(pos))

    def make_shape(self, parent):
        shape = parent.output_shape
        self.input_shape = shape
        assert len(shape) == 1, shape
        self.output_shape = shape
        self.coeffs.shape = (1, self.kernel_size)

    def forward(self, X):
        if self.wrap0 == 0:
            Xper = np.hstack((X, X[:, :self.wrap1]))
        elif self.wrap1 == 0:
            Xper = np.hstack((X[:, -self.wrap0:], X))
        else:
            Xper = np.hstack((X[:, -self.wrap0:], X, X[:, :self.wrap1]))
        self.cols = rolling_batch(Xper, self.kernel_size)
        convolved = (self.cols * self.coeffs.f[:, ::-1]).sum(2)
        return convolved

    def backward(self, dY):
        self.coeffs.g += (dY[:, :, None] * self.cols).sum(0)[:, ::-1].sum(
            0, keepdims=True)
        return (dY[:, :, None] * self.coeffs.f[:, ::-1]).sum(2)


class LayerNorm(Layer):
    # paper: https://arxiv.org/abs/1607.06450
    # note: nonparametric when affine == False

    def __init__(self, eps=1e-5, affine=True):
        super().__init__()
        self.eps = _f(eps)
        self.affine = bool(affine)

        if self.affine:
            self.gamma = self._new_weights('gamma', init=init_ones)
            self.beta = self._new_weights('beta', init=init_zeros)
            self.serialized = {
                'gamma': 'gamma',
                'beta': 'beta',
            }

    def make_shape(self, parent):
        shape = parent.output_shape
        self.input_shape = shape
        self.output_shape = shape
        assert len(shape) == 1, shape
        if self.affine:
            self.gamma.shape = (shape[0],)
            self.beta.shape = (shape[0],)

    def forward(self, X):
        self.mean = X.mean(0)
        self.center = X - self.mean
        self.var = self.center.var(0) + self.eps
        self.std = np.sqrt(self.var)

        self.Xnorm = self.center / self.std
        if self.affine:
            return self.gamma.f * self.Xnorm + self.beta.f
        return self.Xnorm

    def backward(self, dY):
        length = dY.shape[0]

        if self.affine:
            dXnorm = dY * self.gamma.f
            self.gamma.g += (dY * self.Xnorm).sum(0)
            self.beta.g += dY.sum(0)
        else:
            dXnorm = dY

        dstd = (dXnorm * self.center).sum(0) / -self.var
        dcenter = dXnorm / self.std + dstd / self.std * self.center / length
        dmean = -dcenter.sum(0)
        dX = dcenter + dmean / length

        return dX


class Denses(Layer):  # TODO: rename?
    # acts as a separate Dense for each row or column. only for 2D arrays.

    serialized = {
        'W': 'coeffs',
        'b': 'biases',
    }

    def __init__(self, dim, init=init_he_uniform,
                 reg_w=None, reg_b=None, axis=-1):
        super().__init__()
        self.dim = int(dim)
        self.weight_init = init
        self.axis = int(axis)
        self.coeffs = self._new_weights('coeffs', init=init,
                                        regularizer=reg_w)
        self.biases = self._new_weights('biases', init=init_zeros,
                                        regularizer=reg_b)

    def make_shape(self, parent):
        shape = parent.output_shape
        self.input_shape = shape
        assert len(shape) == 2, shape

        assert -len(shape) <= self.axis < len(shape)
        self.axis = self.axis % len(shape)

        self.output_shape = list(shape)
        self.output_shape[self.axis] = self.dim
        self.output_shape = tuple(self.output_shape)

        in_rows = self.input_shape[0]
        in_cols = self.input_shape[1]
        out_rows = self.output_shape[0]
        out_cols = self.output_shape[1]

        self.coeffs.shape = (in_rows, in_cols, self.dim)
        self.biases.shape = (1, out_rows, out_cols)

    def forward(self, X):
        self.X = X
        if self.axis == 0:
            return np.einsum('ixj,xjk->ikj', X, self.coeffs.f) + self.biases.f
        elif self.axis == 1:
            return np.einsum('ijx,jxk->ijk', X, self.coeffs.f) + self.biases.f

    def backward(self, dY):
        self.biases.g += dY.sum(0, keepdims=True)
        if self.axis == 0:
            self.coeffs.g += np.einsum('ixj,ikj->xjk', self.X, dY)
            return np.einsum('ikj,xjk->ixj', dY, self.coeffs.f)
        elif self.axis == 1:
            self.coeffs.g += np.einsum('ijx,ijk->jxk', self.X, dY)
            return np.einsum('ijk,jxk->ijx', dY, self.coeffs.f)


class CosineDense(Dense):
    # paper: https://arxiv.org/abs/1702.05870
    # another implementation:
    # https://github.com/farizrahman4u/keras-contrib/pull/36
    # the paper doesn't mention bias,
    # so we treat bias as an additional weight with a constant input of 1.
    # this is correct in Dense layers, so i hope it's correct here too.

    eps = 1e-4

    def forward(self, X):
        self.X = X
        self.X_norm = np.sqrt(np.square(X).sum(-1, keepdims=True)
                              + 1 + self.eps)
        self.W_norm = np.sqrt(np.square(self.coeffs.f).sum(0, keepdims=True)
                              + np.square(self.biases.f) + self.eps)
        self.dot = X @ self.coeffs.f + self.biases.f
        Y = self.dot / (self.X_norm * self.W_norm)
        return Y

    def backward(self, dY):
        ddot = dY / self.X_norm / self.W_norm
        dX_norm = -(dY * self.dot / self.W_norm).sum(-1, keepdims=True) \
            / self.X_norm**2
        dW_norm = -(dY * self.dot / self.X_norm).sum(0, keepdims=True) \
            / self.W_norm**2

        self.coeffs.g += self.X.T @ ddot \
            + dW_norm / self.W_norm * self.coeffs.f
        self.biases.g += ddot.sum(0, keepdims=True) \
            + dW_norm / self.W_norm * self.biases.f
        dX = ddot @ self.coeffs.f.T + dX_norm / self.X_norm * self.X

        return dX


class Sparse(Layer):
    # (WIP)
    # roughly implements a structured, sparsely-connected layer.
    # paper: https://arxiv.org/abs/1812.01164

    # TODO: (re)implement serialization.

    def __init__(self, dim, con, init=init_he_uniform, reg=None):
        super().__init__()
        self.dim = int(dim)
        self.con = int(con)
        self.output_shape = (dim,)
        self.coeffs = self._new_weights('coeffs', init=init, regularizer=reg)
        self.indices = None

    def make_shape(self, parent):
        shape = parent.output_shape
        self.input_shape = shape
        assert len(shape) == 1, shape
        self.coeffs.shape = (self.con, self.dim)
        self.size_in = shape[0]
        self.make_indices(self.size_in, self.con, self.dim)

    def make_indices(self, size_in, connectivity, size_out):
        basic = np.arange(size_in)
        indices = []
        inv_ind = []
        count = 0
        desired = size_out * connectivity
        # TODO: replace with a for loop.
        while count < desired:
            np.random.shuffle(basic)
            indices.append(basic.copy())
            inverse = np.zeros_like(basic)
            inverse[basic] = np.arange(len(basic)) + count
            inv_ind.append(inverse)
            count += len(basic)
        self.indices = np.concatenate(indices)[:desired].copy()
        self.inv_ind = np.concatenate(inv_ind)

    def forward(self, X):
        self.X = X
        self.O = X[:,self.indices].reshape(len(X), self.con, self.dim)
        return np.sum(self.O * self.coeffs.f, 1)

    def backward(self, dY):
        dY = np.expand_dims(dY, 1)
        self.coeffs.g += np.sum(dY * self.O, 0)
        dO = dY * self.coeffs.f

        x = dO
        batch_size = len(x)
        x = x.reshape(batch_size, -1)
        if x.shape[1] % self.size_in != 0:
            x = np.pad(x, ((0, 0), (0, self.size_in - x.shape[1] % self.size_in)))
        x = x[:, self.inv_ind].reshape(batch_size, -1, self.size_in)
        return x.sum(1)
