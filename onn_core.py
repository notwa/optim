import numpy as np
_f = np.float32

# just for speed, not strictly essential:
from scipy.special import expit as sigmoid

# used for numbering layers like Keras, and keeping initialization consistent:
from collections import defaultdict, OrderedDict
_layer_counters = defaultdict(lambda: 0)

def _check(a):
    assert isinstance(a, np.ndarray) or type(a) == _f, type(a)
    assert a.dtype == _f, a.dtype
    return a

_0 = _f(0)
_1 = _f(1)
_2 = _f(2)
_inv2 = _f(1/2)
_sqrt2 = _f(np.sqrt(2))
_invsqrt2 = _f(1/np.sqrt(2))
_pi = _f(np.pi)

class LayerIncompatibility(Exception):
    pass

# Initializations {{{1

# note: these are currently only implemented for 2D shapes.

def init_zeros(size, ins=None, outs=None):
    return np.zeros(size)

def init_ones(size, ins=None, outs=None):
    return np.ones(size)

def init_he_normal(size, ins, outs):
    s = np.sqrt(2 / ins)
    return np.random.normal(0, s, size=size)

def init_he_uniform(size, ins, outs):
    s = np.sqrt(6 / ins)
    return np.random.uniform(-s, s, size=size)

def init_glorot_normal(size, ins, outs):
    s = np.sqrt(2 / (ins + outs))
    return np.random.normal(0, s, size=size)

def init_glorot_uniform(size, ins, outs):
    s = np.sqrt(6 / (ins + outs))
    return np.random.uniform(-s, s, size=size)

# Weight container {{{1

class Weights:
    # we may or may not contain weights -- or any information, for that matter.

    def __init__(self, **kwargs):
        self.f = None # forward weights
        self.g = None # backward weights (gradients)
        self.shape = None
        self.init = None
        self.allocator = None
        self.regularizer = None

        self.configure(**kwargs)

    def configure(self, **kwargs):
        for k, v in kwargs.items():
            getattr(self, k) # ensures the key already exists
            setattr(self, k, v)

    @property
    def size(self):
        assert self.shape is not None
        return np.prod(self.shape)

    def allocate(self, *args, **kwargs):
        self.configure(**kwargs)

        # intentionally not using isinstance
        assert type(self.shape) == tuple, self.shape

        f, g = self.allocator(self.size)
        assert len(f) == self.size, "{} != {}".format(f.shape, self.size)
        assert len(g) == self.size, "{} != {}".format(g.shape, self.size)
        f[:] = self.init(self.size, *args)
        g[:] = self.init(self.size, *args)
        self.f = f.reshape(self.shape)
        self.g = g.reshape(self.shape)

    def forward(self):
        if self.regularizer is None:
            return 0.0
        return self.regularizer.forward(self.f)

    def backward(self):
        if self.regularizer is None:
            return 0.0
        return self.regularizer.backward(self.f)

    def update(self):
        if self.regularizer is None:
            return
        self.g += self.regularizer.backward(self.f)

# Loss functions {{{1

class Loss:
    pass

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
    # use CategoricalCrossentropy for that instead.

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

# Regularizers {{{1

class Regularizer:
    pass

class L1L2(Regularizer):
    def __init__(self, l1=0.0, l2=0.0):
        self.l1 = _f(l1)
        self.l2 = _f(l2)

    def forward(self, X):
        f = _0
        if self.l1:
            f += np.sum(self.l1 * np.abs(X))
        if self.l2:
            f += np.sum(self.l2 * np.square(X))
        return f

    def backward(self, X):
        df = np.zeros_like(X)
        if self.l1:
            df += self.l1 * np.sign(X)
        if self.l2:
            df += self.l2 * 2 * X
        return df

# Optimizers {{{1

class Optimizer:
    def __init__(self, alpha=0.1):
        self.alpha = _f(alpha) # learning rate
        self.reset()

    def reset(self):
        pass

    def compute(self, dW, W):
        return -self.alpha * dW

    def update(self, dW, W):
        W += self.compute(dW, W)

# the following optimizers are blatantly lifted from tiny-dnn:
# https://github.com/tiny-dnn/tiny-dnn/blob/master/tiny_dnn/optimizers/optimizer.h

class Momentum(Optimizer):
    def __init__(self, alpha=0.01, mu=0.9, nesterov=False):
        self.mu = _f(mu) # momentum
        self.nesterov = bool(nesterov)

        super().__init__(alpha)

    def reset(self):
        self.Vprev = None

    def compute(self, dW, W):
        if self.Vprev is None:
            self.Vprev = np.copy(dW)

        V = self.mu * self.Vprev - self.alpha * dW
        self.Vprev[:] = V
        if self.nesterov:
            return self.mu * V - self.alpha * dW

        return V

class RMSprop(Optimizer):
    # RMSprop generalizes* Adagrad, etc.

    # TODO: verify this is correct:
    # * RMSprop == Adagrad when
    #   RMSprop.mu == 1

    def __init__(self, alpha=0.0001, mu=0.99, eps=1e-8):
        self.mu = _f(mu) # decay term
        self.eps = _f(eps)

        # one might consider the following equation when specifying mu:
        # mu = e**(-1/t)
        # default: t = -1/ln(0.99) = ~99.5
        # therefore the default of mu=0.99 means
        # an input decays to 1/e its original amplitude over 99.5 epochs.
        # (this is from DSP, so how relevant it is in SGD is debatable)

        super().__init__(alpha)

    def reset(self):
        self.g = None

    def compute(self, dW, W):
        if self.g is None:
            self.g = np.zeros_like(dW)

        # basically apply a first-order low-pass filter to delta squared
        self.g[:] = self.mu * self.g + (1 - self.mu) * dW * dW
        # equivalent (though numerically different?):
        #self.g += (dW * dW - self.g) * (1 - self.mu)

        # finally sqrt it to complete the running root-mean-square approximation
        return -self.alpha * dW / (np.sqrt(self.g) + self.eps)

class Adam(Optimizer):
    # paper: https://arxiv.org/abs/1412.6980
    # Adam generalizes* RMSprop, and
    # adds a decay term to the regular (non-squared) delta, and
    # does some decay-gain voodoo. (i guess it's compensating
    # for the filtered deltas starting from zero)

    # * Adam == RMSprop when
    #   Adam.b1 == 0
    #   Adam.b2 == RMSprop.mu

    def __init__(self, alpha=0.002, b1=0.9, b2=0.999, eps=1e-8):
        self.b1 = _f(b1) # decay term
        self.b2 = _f(b2) # decay term
        self.b1_t_default = _f(b1) # decay term power t
        self.b2_t_default = _f(b2) # decay term power t
        self.eps = _f(eps)

        super().__init__(alpha)

    def reset(self):
        self.mt = None
        self.vt = None
        self.b1_t = self.b1_t_default
        self.b2_t = self.b2_t_default

    def compute(self, dW, W):
        if self.mt is None:
            self.mt = np.zeros_like(dW)
        if self.vt is None:
            self.vt = np.zeros_like(dW)

        # decay gain
        self.b1_t *= self.b1
        self.b2_t *= self.b2

        # filter
        self.mt[:] = self.b1 * self.mt + (1 - self.b1) * dW
        self.vt[:] = self.b2 * self.vt + (1 - self.b2) * dW * dW

        return -self.alpha * (self.mt / (1 - self.b1_t)) \
                   / (np.sqrt(self.vt / (1 - self.b2_t)) + self.eps)

class Nadam(Optimizer):
    # paper: https://arxiv.org/abs/1412.6980
    # paper: http://cs229.stanford.edu/proj2015/054_report.pdf
    # TODO: double-check this implementation. also read the damn paper.
    # lifted from https://github.com/fchollet/keras/blob/5d38b04/keras/optimizers.py#L530
    # lifted from https://github.com/jpilaul/IFT6266_project/blob/master/Models/Algo_Momentum.py

    def __init__(self, alpha=0.002, b1=0.9, b2=0.999, eps=1e-8):
        self.b1 = _f(b1) # decay term
        self.b2 = _f(b2) # decay term
        self.eps = _f(eps)

        super().__init__(alpha)

    def reset(self):
        self.mt = None
        self.vt = None
        self.t = 0
        self.sched = 1

    def compute(self, dW, W):
        self.t += 1

        if self.mt is None:
            self.mt = np.zeros_like(dW)
        if self.vt is None:
            self.vt = np.zeros_like(dW)

        ut0 = self.b1 * (1 - 0.5 * 0.96**(self.t + 0))
        ut1 = self.b1 * (1 - 0.5 * 0.96**(self.t + 1))

        sched0 = self.sched * ut0
        sched1 = self.sched * ut0 * ut1
        self.sched = sched0

        gp = dW / (1 - sched0)

        self.mt[:] = self.b1 * self.mt + (1 - self.b1) * dW
        self.vt[:] = self.b2 * self.vt + (1 - self.b2) * np.square(dW)

        mtp = self.mt / (1 - sched1)
        vtp = self.vt / (1 - self.b2**self.t)

        mt_bar = (1 - ut0) * gp + ut1 * mtp

        return -self.alpha * mt_bar / (np.sqrt(vtp) + self.eps)

# Abstract Layers {{{1

class Layer:
    def __init__(self):
        self.parents = []
        self.children = []
        self.weights = OrderedDict()
        self.loss = None # for activity regularizers
        self.input_shape = None
        self.output_shape = None
        kind = self.__class__.__name__
        global _layer_counters
        _layer_counters[kind] += 1
        self.name = "{}_{}".format(kind, _layer_counters[kind])
        self.unsafe = False # disables assertions for better performance

    def __str__(self):
        return self.name

    # methods we might want to override:

    def forward(self, X):
        raise NotImplementedError("unimplemented", self)

    def forward_deterministic(self, X):
        return self.forward(X)

    def backward(self, dY):
        raise NotImplementedError("unimplemented", self)

    def make_shape(self, parent):
        if self.input_shape == None:
            self.input_shape = parent.output_shape
        if self.output_shape == None:
            self.output_shape = self.input_shape

    def do_feed(self, child):
        self.children.append(child)

    def be_fed(self, parent):
        self.parents.append(parent)

    # TODO: better names for these (still)
    def _propagate(self, edges, deterministic):
        if not self.unsafe:
            assert len(edges) == 1, self
        if deterministic:
            return self.forward_deterministic(edges[0])
        else:
            return self.forward(edges[0])

    def _backpropagate(self, edges):
        if len(edges) == 1:
            return self.backward(edges[0])
        return sum((self.backward(dY) for dY in edges))

    # general utility methods:

    def is_compatible(self, parent):
        return np.all(self.input_shape == parent.output_shape)

    def feed(self, child):
        assert self.output_shape is not None, self
        child.make_shape(self)
        if not child.is_compatible(self):
            fmt = "{} is incompatible with {}: shape mismatch: {} vs. {}"
            raise LayerIncompatibility(fmt.format(self, child, self.output_shape, child.input_shape))
        self.do_feed(child)
        child.be_fed(self)
        return child

    def validate_input(self, X):
        assert X.shape[1:] == self.input_shape,  (str(self), X.shape[1:], self.input_shape)

    def validate_output(self, Y):
        assert Y.shape[1:] == self.output_shape, (str(self), Y.shape[1:], self.output_shape)

    def _new_weights(self, name, **kwargs):
        w = Weights(**kwargs)
        assert name not in self.weights, name
        self.weights[name] = w
        return w

    @property
    def size(self):
        return sum((w.size for w in self.weights.values()))

    def init(self, allocator):
        ins, outs = self.input_shape[0], self.output_shape[0]
        for k, w in self.weights.items():
            w.allocate(ins, outs, allocator=allocator)

    def propagate(self, values, deterministic):
        if not self.unsafe:
            assert self.parents, self
        edges = []
        for parent in self.parents:
            # TODO: skip over irrelevant nodes (if any)
            X = values[parent]
            if not self.unsafe:
                self.validate_input(X)
            edges.append(X)
        Y = self._propagate(edges, deterministic)
        if not self.unsafe:
            self.validate_output(Y)
        return Y

    def backpropagate(self, values):
        if not self.unsafe:
            assert self.children, self
        edges = []
        for child in self.children:
            # TODO: skip over irrelevant nodes (if any)
            dY = values[child]
            if not self.unsafe:
                self.validate_output(dY)
            edges.append(dY)
        dX = self._backpropagate(edges)
        if not self.unsafe:
            self.validate_input(dX)
        return dX

# Nonparametric Layers {{{1

class Input(Layer):
    def __init__(self, shape):
        assert shape is not None
        super().__init__()
        self.shape = tuple(shape)
        self.input_shape = self.shape
        self.output_shape = self.shape

    def forward(self, X):
        return X

    def backward(self, dY):
        #self.dY = dY
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
        #assert len(edges) == 1, "unimplemented"
        return edges[0] # TODO: does this always work?

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
        #self.mask = _1
        return X

    def backward(self, dY):
        return dY * self.mask

# Activation Layers {{{2

class Sigmoid(Layer): # aka Logistic, Expit (inverse of Logit)
    def forward(self, X):
        self.sig = sigmoid(X)
        return self.sig

    def backward(self, dY):
        return dY * self.sig * (1 - self.sig)

class Softplus(Layer):
    # integral of Sigmoid.

    def forward(self, X):
        self.X = X
        return np.log(1 + np.exp(X))

    def backward(self, dY):
        return sigmoid(self.X)

class Tanh(Layer):
    def forward(self, X):
        self.sig = np.tanh(X)
        return self.sig

    def backward(self, dY):
        return dY * (1 - self.sig * self.sig)

class LeCunTanh(Layer):
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

class Relu(Layer):
    def forward(self, X):
        self.cond = X >= 0
        return np.where(self.cond, X, 0)

    def backward(self, dY):
        return np.where(self.cond, dY, 0)

class Elu(Layer):
    # paper: https://arxiv.org/abs/1511.07289

    def __init__(self, alpha=1):
        super().__init__()
        self.alpha = _f(alpha) # FIXME: unused

    def forward(self, X):
        self.cond = X >= 0
        self.neg = np.exp(X) - 1
        return np.where(self.cond, X, self.neg)

    def backward(self, dY):
        return dY * np.where(self.cond, 1, self.neg + 1)

class GeluApprox(Layer):
    # paper: https://arxiv.org/abs/1606.08415
    #  plot: https://www.desmos.com/calculator/ydzgtccsld

    def forward(self, X):
        self.a = 1.704 * X
        self.sig = sigmoid(self.a)
        return X * self.sig

    def backward(self, dY):
        return dY * self.sig * (1 + self.a * (1 - self.sig))

class Softmax(Layer):
    def __init__(self, axis=-1):
        super().__init__()
        self.axis = int(axis)

    def forward(self, X):
        alpha = np.max(X, axis=-1, keepdims=True)
        num = np.exp(X - alpha)
        den = np.sum(num, axis=-1, keepdims=True)
        self.sm = num / den
        return self.sm

    def backward(self, dY):
        return (dY - np.sum(dY * self.sm, axis=-1, keepdims=True)) * self.sm

class LogSoftmax(Softmax):
    def __init__(self, axis=-1, eps=1e-6):
        super().__init__()
        self.axis = int(axis)
        self.eps = _f(eps)

    def forward(self, X):
        return np.log(super().forward(X) + self.eps)

    def backward(self, dY):
        return dY - np.sum(dY, axis=-1, keepdims=True) * self.sm

class Cos(Layer):
    # performs well on MNIST for some strange reason.

    def forward(self, X):
        self.X = X
        return np.cos(X)

    def backward(self, dY):
        return dY * -np.sin(self.X)

# Parametric Layers {{{1

class Dense(Layer):
    serialized = {
        'W': 'coeffs',
        'b': 'biases',
    }

    def __init__(self, dim, init=init_he_uniform, reg_w=None, reg_b=None):
        super().__init__()
        self.dim = int(dim)
        self.output_shape = (dim,)
        self.coeffs = self._new_weights('coeffs', init=init, regularizer=reg_w)
        self.biases = self._new_weights('biases', init=init_zeros, regularizer=reg_b)

    def make_shape(self, parent):
        shape = parent.output_shape
        self.input_shape = shape
        assert len(shape) == 1, shape
        self.coeffs.shape = (shape[0], self.dim)
        self.biases.shape = (1, self.dim)

    def forward(self, X):
        self.X = X
        return X.dot(self.coeffs.f) + self.biases.f

    def backward(self, dY):
        self.coeffs.g[:] = self.X.T.dot(dY)
        self.biases.g[:] = dY.sum(0, keepdims=True)
        return dY.dot(self.coeffs.f.T)

# Models {{{1

class Model:
    def __init__(self, x, y, unsafe=False):
        assert isinstance(x, Layer), x
        assert isinstance(y, Layer), y
        self.x = x
        self.y = y
        self.ordered_nodes = self.traverse([], self.y)
        self.make_weights()
        for node in self.ordered_nodes:
            node.unsafe = unsafe

    def make_weights(self):
        self.param_count = sum((node.size for node in self.ordered_nodes))
        self.W  = np.zeros(self.param_count, dtype=_f)
        self.dW = np.zeros(self.param_count, dtype=_f)

        offset = 0
        for node in self.ordered_nodes:
            if node.size > 0:
                end = offset + node.size
                inner_offset = 0

                def allocate(size):
                    nonlocal inner_offset
                    o = offset + inner_offset
                    ret = self.W[o:o+size], self.dW[o:o+size]
                    inner_offset += size
                    assert len(ret[0]) == len(ret[1])
                    assert size == len(ret[0]), (size, len(ret[0]))
                    return ret

                node.init(allocate)
                assert inner_offset <= node.size, "Layer {} allocated more weights than it said it would".format(node)
                # i don't care if "less" is grammatically incorrect.
                # you're mom is grammatically incorrect.
                assert inner_offset >= node.size, "Layer {} allocated less weights than it said it would".format(node)
                offset += node.size

    def traverse(self, nodes, node):
        if node == self.x:
            return [node]
        for parent in node.parents:
            if parent not in nodes:
                new_nodes = self.traverse(nodes, parent)
                for new_node in new_nodes:
                    if new_node not in nodes:
                        nodes.append(new_node)
        if nodes:
            nodes.append(node)
        return nodes

    def forward(self, X, deterministic=False):
        values = dict()
        input_node = self.ordered_nodes[0]
        output_node = self.ordered_nodes[-1]
        values[input_node] = input_node._propagate(np.expand_dims(X, 0), deterministic)
        for node in self.ordered_nodes[1:]:
            values[node] = node.propagate(values, deterministic)
        return values[output_node]

    def backward(self, error):
        values = dict()
        output_node = self.ordered_nodes[-1]
        values[output_node] = output_node._backpropagate(np.expand_dims(error, 0))
        for node in reversed(self.ordered_nodes[:-1]):
            values[node] = node.backpropagate(values)
        return self.dW

    def regulate_forward(self):
        loss = _0
        for node in self.ordered_nodes:
            if node.loss is not None:
                loss += node.loss
            for k, w in node.weights.items():
                loss += w.forward()
        return loss

    def regulate(self):
        for node in self.ordered_nodes:
            for k, w in node.weights.items():
                w.update()

    def load_weights(self, fn):
        # seemingly compatible with keras' Dense layers.
        import h5py
        open(fn) # just ensure the file exists (python's error is better)
        f = h5py.File(fn, 'r')
        weights = {}
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                weights[name.split('/')[-1]] = np.array(obj[:], dtype=_f)
        f.visititems(visitor)
        f.close()

        used = {}
        for k in weights.keys():
            used[k] = False

        nodes = [node for node in self.ordered_nodes if node.size > 0]
        for node in nodes:
            full_name = str(node).lower()
            for s_name, o_name in node.serialized.items():
                key = full_name + '_' + s_name
                data = weights[key]
                target = getattr(node, o_name)
                target.f[:] = data
                used[key] = True

        for k, v in used.items():
            if not v:
                lament("WARNING: unused weight", k)

    def save_weights(self, fn, overwrite=False):
        import h5py
        f = h5py.File(fn, 'w')

        counts = defaultdict(lambda: 0)

        nodes = [node for node in self.ordered_nodes if node.size > 0]
        for node in nodes:
            full_name = str(node).lower()
            grp = f.create_group(full_name)
            for s_name, o_name in node.serialized.items():
                key = full_name + '_' + s_name
                target = getattr(node, o_name)
                data = grp.create_dataset(key, target.shape, dtype=_f)
                data[:] = target.f
                counts[key] += 1
                if counts[key] > 1:
                    lament("WARNING: rewrote weight", key)

        f.close()

# Rituals {{{1

class Ritual: # i'm just making up names at this point
    def __init__(self, learner=None, loss=None, mloss=None):
        self.learner = learner if learner is not None else Learner(Optimizer())
        self.loss = loss if loss is not None else Squared()
        self.mloss = mloss if mloss is not None else loss
        self.model = None

    def reset(self):
        self.learner.reset(optim=True)
        self.en = 0
        self.bn = 0

    def measure(self, p, y):
        return self.mloss.forward(p, y)

    def forward(self, p, y):
        return self.loss.forward(p, y) + self.model.regulate_forward()

    def backward(self, p, y):
        return self.loss.backward(p, y)

    def learn(self, inputs, outputs):
        predicted = self.model.forward(inputs)
        self.model.backward(self.backward(predicted, outputs))
        self.model.regulate()
        return predicted

    def update(self):
        self.learner.optim.update(self.model.dW, self.model.W)

    def prepare(self, model):
        self.en = 0
        self.bn = 0
        self.model = model

    def train_batched_gen(self, generator, batch_count,
                          return_losses=False, test_only=False):
        assert isinstance(return_losses, bool) or return_losses == 'both'

        if not test_only:
            self.en += 1

        cumsum_loss, cumsum_mloss = _0, _0
        losses, mlosses = [], []

        prev_batch_size = None

        for b in range(batch_count):
            if not test_only:
                self.bn += 1

            # TODO: pass a GeneratorData object containing en, bn, ritual/model fields.
            #       ...is there a pythonic way of doing that?
            batch_inputs, batch_outputs = next(generator)

            batch_size = batch_inputs.shape[0]
            assert batch_size == prev_batch_size or prev_batch_size is None, \
              "non-constant batch size (got {} expected {})".format(
                batch_size, prev_batch_size)  # TODO: lift this restriction
            prev_batch_size = batch_size

            # same from hereon

            if not test_only and self.learner.per_batch:
                self.learner.batch(b / batch_count)

            if test_only:
                predicted = self.model.forward(batch_inputs, deterministic=True)
            else:
                predicted = self.learn(batch_inputs, batch_outputs)
                self.update()

            if return_losses == 'both':
                batch_loss = self.forward(predicted, batch_outputs)
                if np.isnan(batch_loss):
                    raise Exception("nan")
                losses.append(batch_loss)
                cumsum_loss += batch_loss

            # NOTE: this can use the non-deterministic predictions. fixme?
            batch_mloss = self.measure(predicted, batch_outputs)
            if np.isnan(batch_mloss):
                raise Exception("nan")
            if return_losses:
                mlosses.append(batch_mloss)
            cumsum_mloss += batch_mloss

        avg_mloss = cumsum_mloss / _f(batch_count)
        if return_losses == 'both':
            avg_loss = cumsum_loss / _f(batch_count)
            return avg_loss, avg_mloss, losses, mlosses
        elif return_losses:
            return avg_mloss, mlosses
        return avg_mloss

    def train_batched(self, inputs, outputs, batch_size,
                      return_losses=False, test_only=False, shuffle=True):
        assert isinstance(return_losses, bool) or return_losses == 'both'

        if not test_only:
            self.en += 1

        if shuffle:
            indices = np.arange(inputs.shape[0])
            np.random.shuffle(indices)
            inputs = inputs[indices]
            outputs = outputs[indices]

        cumsum_loss, cumsum_mloss = _0, _0
        batch_count = inputs.shape[0] // batch_size
        losses, mlosses = [], []

        assert inputs.shape[0] % batch_size == 0, \
          "inputs is not evenly divisible by batch_size" # TODO: lift this restriction
        for b in range(batch_count):
            if not test_only:
                self.bn += 1

            bi = b * batch_size
            batch_inputs  = inputs[ bi:bi+batch_size]
            batch_outputs = outputs[bi:bi+batch_size]

            # same from hereon

            if not test_only and self.learner.per_batch:
                self.learner.batch(b / batch_count)

            if test_only:
                predicted = self.model.forward(batch_inputs, deterministic=True)
            else:
                predicted = self.learn(batch_inputs, batch_outputs)
                self.update()

            if return_losses == 'both':
                batch_loss = self.forward(predicted, batch_outputs)
                if np.isnan(batch_loss):
                    raise Exception("nan")
                losses.append(batch_loss)
                cumsum_loss += batch_loss

            # NOTE: this can use the non-deterministic predictions. fixme?
            batch_mloss = self.measure(predicted, batch_outputs)
            if np.isnan(batch_mloss):
                raise Exception("nan")
            if return_losses:
                mlosses.append(batch_mloss)
            cumsum_mloss += batch_mloss

        avg_mloss = cumsum_mloss / _f(batch_count)
        if return_losses == 'both':
            avg_loss = cumsum_loss / _f(batch_count)
            return avg_loss, avg_mloss, losses, mlosses
        elif return_losses:
            return avg_mloss, mlosses
        return avg_mloss

    def test_batched(self, *args, **kwargs):
        return self.train_batched(*args, test_only=True, **kwargs)

# Learners {{{1

class Learner:
    per_batch = False

    def __init__(self, optim, epochs=100, rate=None):
        assert isinstance(optim, Optimizer)
        self.optim = optim
        self.start_rate = optim.alpha if rate is None else _f(rate)
        self.epochs = int(epochs)
        self.reset()

    def reset(self, optim=False):
        self.started = False
        self.epoch = 0
        if optim:
            self.optim.reset()

    @property
    def epoch(self):
        return self._epoch

    @epoch.setter
    def epoch(self, new_epoch):
        self._epoch = int(new_epoch)
        if 0 <= self.epoch <= self.epochs:
            self.rate = self.rate_at(self._epoch)

    @property
    def rate(self):
        return self.optim.alpha

    @rate.setter
    def rate(self, new_rate):
        self.optim.alpha = new_rate

    def rate_at(self, epoch):
        return self.start_rate

    def next(self):
        # prepares the next epoch. returns whether or not to continue training.
        if not self.started:
            self.started = True
        self.epoch += 1
        if self.epoch > self.epochs:
            return False
        return True

    def batch(self, progress): # TODO: rename
        # interpolates rates between epochs.
        # unlike epochs, we do not store batch number as a state.
        # i.e. calling next() will not respect progress.
        assert 0 <= progress <= 1
        self.rate = self.rate_at(self._epoch - 1 + progress)

    @property
    def final_rate(self):
        return self.rate_at(self.epochs - 1)

class AnnealingLearner(Learner):
    def __init__(self, optim, epochs=100, rate=None, halve_every=10):
        self.halve_every = _f(halve_every)
        self.anneal = _f(0.5**(1/self.halve_every))
        super().__init__(optim, epochs, rate)

    def rate_at(self, epoch):
        return self.start_rate * self.anneal**epoch

def cosmod(x):
    # plot: https://www.desmos.com/calculator/hlgqmyswy2
    return (_1 + np.cos((x % _1) * _pi)) * _inv2

class SGDR(Learner):
    # Stochastic Gradient Descent with Restarts
    # paper: https://arxiv.org/abs/1608.03983
    # NOTE: this is missing a couple features.

    per_batch = True

    def __init__(self, optim, epochs=100, rate=None,
                 restarts=0, restart_decay=0.5, callback=None,
                 expando=0):
        self.restart_epochs = int(epochs)
        self.decay = _f(restart_decay)
        self.restarts = int(restarts)
        self.restart_callback = callback
        # TODO: rename expando to something not insane
        self.expando = expando if expando is not None else lambda i: i
        if type(self.expando) == int:
            inc = self.expando
            self.expando = self.expando = lambda i: inc

        self.splits = []
        epochs = 0
        for i in range(0, self.restarts + 1):
            split = epochs + self.restart_epochs + int(self.expando(i))
            self.splits.append(split)
            epochs = split
        super().__init__(optim, epochs, rate)

    def split_num(self, epoch):
        shit = [0] + self.splits # hack
        for i in range(0, len(self.splits)):
            if epoch < self.splits[i]:
                sub_epoch = epoch - shit[i]
                next_restart = self.splits[i] - shit[i]
                return i, sub_epoch, next_restart
        if epoch == self.splits[-1]:
            return len(self.splits) - 1, epoch, self.splits[-1]
        raise Exception('this should never happen.')

    def rate_at(self, epoch):
        restart, sub_epoch, next_restart = self.split_num(epoch)
        x = _f(sub_epoch) / _f(next_restart)
        return self.start_rate * self.decay**_f(restart) * cosmod(x)

    def next(self):
        if not super().next():
            return False
        restart, sub_epoch, next_restart = self.split_num(self.epoch)
        if restart > 0 and sub_epoch == 0:
            if self.restart_callback is not None:
                self.restart_callback(restart)
        return True

class TriangularCLR(Learner):
    # note: i haven't actually read (nor seen) the paper(s) on CLR,
    # but this case (triangular) should be pretty difficult to get wrong.

    per_batch = True

    def __init__(self, optim, epochs=400, upper_rate=None, lower_rate=0,
                 frequency=100, callback=None):
        # NOTE: start_rate is treated as upper_rate
        self.frequency = int(frequency)
        assert self.frequency > 0
        self.callback = callback
        self.lower_rate = _f(lower_rate)
        super().__init__(optim, epochs, upper_rate)

    def _t(self, epoch):
        # NOTE: this could probably be simplified
        offset = self.frequency / 2
        return np.abs(((epoch + offset) % self.frequency) - offset) / offset

    def rate_at(self, epoch):
        # NOTE: start_rate is treated as upper_rate
        return self._t(epoch) * (self.start_rate - self.lower_rate) + self.lower_rate

    def next(self):
        if not super().next():
            return False
        if self.epoch > 1 and self.epoch % self.frequency == 0:
            if self.callback is not None:
                self.callback(self.epoch // self.frequency)
        return True

class SineCLR(TriangularCLR):
    def _t(self, epoch):
        return np.sin(_pi * _inv2 * super()._t(epoch))

class WaveCLR(TriangularCLR):
    def _t(self, epoch):
        return _inv2 * (_1 - np.cos(_pi * super()._t(epoch)))