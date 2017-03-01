import numpy as np
_f = np.float32

# just for speed, not strictly essential:
from scipy.special import expit as sigmoid

# used for numbering layers like Keras:
from collections import defaultdict
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

# Loss functions {{{1

class Loss:
    pass

class CategoricalCrossentropy(Loss):
    # lifted from theano

    def __init__(self, eps=1e-6):
        self.eps = _f(eps)

    def forward(self, p, y):
        # TODO: assert dimensionality and p > 0 (if not self.unsafe?)
        p = np.clip(p, self.eps, 1 - self.eps)
        f = np.sum(-y * np.log(p) - (1 - y) * np.log(1 - p), axis=-1)
        return np.mean(f)

    def backward(self, p, y):
        p = np.clip(p, self.eps, 1 - self.eps)
        df = (p - y) / (p * (1 - p))
        return df / len(y)

class Accuracy(Loss):
    # returns percentage of categories correctly predicted.
    # utilizes max(), so it cannot be used for gradient descent.
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
        return -self.alpha * dW / np.sqrt(self.g + self.eps)

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
                   / np.sqrt((self.vt / (1 - self.b2_t)) + self.eps)

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
        self.input_shape = None
        self.output_shape = None
        kind = self.__class__.__name__
        global _layer_counters
        _layer_counters[kind] += 1
        self.name = "{}_{}".format(kind, _layer_counters[kind])
        self.size = None # total weight count (if any)
        self.unsafe = False # disables assertions for better performance

    def __str__(self):
        return self.name

    # methods we might want to override:

    def forward(self, X):
        raise NotImplementedError("unimplemented", self)

    def backward(self, dY):
        raise NotImplementedError("unimplemented", self)

    def do_feed(self, child):
        self.children.append(child)

    def be_fed(self, parent):
        self.parents.append(parent)

    def make_shape(self, shape):
        if not self.unsafe:
            assert shape is not None
        if self.output_shape is None:
            self.output_shape = shape
        return shape

    # TODO: better names for these (still)

    def _propogate(self, edges):
        if not self.unsafe:
            assert len(edges) == 1, self
        return self.forward(edges[0])

    def _backpropogate(self, edges):
        if len(edges) == 1:
            return self.backward(edges[0])
        return sum((self.backward(dY) for dY in edges))

    # general utility methods:

    def is_compatible(self, parent):
        if self.input_shape is None:
            # inherit shape from output
            shape = self.make_shape(parent.output_shape)
            if shape is None:
                return False
            self.input_shape = shape
        return np.all(self.input_shape == parent.output_shape)

    def feed(self, child):
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

    def init(self, W, dW):
        assert  W.ndim == 1  and W.shape[0] == self.size, W.shape
        assert dW.ndim == 1 and dW.shape[0] == self.size, dW.shape
        self.W = W
        self.dW = dW

    def propagate(self, values):
        if not self.unsafe:
            assert self.parents, self
        edges = []
        for parent in self.parents:
            # TODO: skip over irrelevant nodes (if any)
            X = values[parent]
            if not self.unsafe:
                self.validate_input(X)
            edges.append(X)
        Y = self._propogate(edges)
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
        dX = self._backpropogate(edges)
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
    def make_shape(self, shape):
        super().make_shape(shape)
        self.output_shape = (np.prod(shape),)
        return shape

    def forward(self, X):
        self.batch_size = X.shape[0]
        return X.reshape(self.batch_size, *self.output_shape)

    def backward(self, dY):
        assert dY.shape[0] == self.batch_size
        return dY.reshape(self.batch_size, *self.input_shape)

class Affine(Layer):
    def __init__(self, a=1, b=0):
        super().__init__()
        self.a = _f(a)
        self.b = _f(b)

    def forward(self, X):
        return self.a * X + self.b

    def backward(self, dY):
        return dY * self.a

class Sum(Layer):
    def _propogate(self, edges):
        return np.sum(edges, axis=0)

    def _backpropogate(self, edges):
        #assert len(edges) == 1, "unimplemented"
        return edges[0] # TODO: does this always work?

class Sigmoid(Layer): # aka Logistic
    def forward(self, X):
        self.sig = sigmoid(X)
        return self.sig

    def backward(self, dY):
        return dY * self.sig * (1 - self.sig)

class Tanh(Layer):
    def forward(self, X):
        self.sig = np.tanh(X)
        return self.sig

    def backward(self, dY):
        return dY * (1 - self.sig * self.sig)

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
        self.alpha = _f(alpha)

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
    # lifted from theano

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
        dYsm = dY * self.sm
        dX = dYsm - np.sum(dYsm, axis=-1, keepdims=True) * self.sm
        return dX

# Parametric Layers {{{1

class Dense(Layer):
    serialized = {
        'W': 'coeffs',
        'b': 'biases',
    }

    def __init__(self, dim, init=init_he_uniform):
        super().__init__()
        self.dim = int(dim)
        self.output_shape = (dim,)
        self.weight_init = init
        self.size = None

    def make_shape(self, shape):
        super().make_shape(shape)
        if len(shape) != 1:
            return False
        self.nW = self.dim * shape[0]
        self.nb = self.dim
        self.size = self.nW + self.nb
        return shape

    def init(self, W, dW):
        super().init(W, dW)

        ins, outs = self.input_shape[0], self.output_shape[0]

        self.coeffs = self.W[:self.nW].reshape(ins, outs)
        self.biases = self.W[self.nW:].reshape(1, outs)
        self.dcoeffs = self.dW[:self.nW].reshape(ins, outs)
        self.dbiases = self.dW[self.nW:].reshape(1, outs)

        self.coeffs.flat = self.weight_init(self.nW, ins, outs)
        self.biases.flat = 0

        self.std = np.std(self.W)

    def forward(self, X):
        self.X = X
        return X.dot(self.coeffs) + self.biases

    def backward(self, dY):
        self.dcoeffs[:] = self.X.T.dot(dY)
        self.dbiases[:] = dY.sum(0, keepdims=True)
        return dY.dot(self.coeffs.T)

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
        self.param_count = 0
        for node in self.ordered_nodes:
            if node.size is not None:
                self.param_count += node.size
        self.W  = np.zeros(self.param_count, dtype=_f)
        self.dW = np.zeros(self.param_count, dtype=_f)

        offset = 0
        for node in self.ordered_nodes:
            if node.size is not None:
                end = offset + node.size
                node.init(self.W[offset:end], self.dW[offset:end])
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

    def forward(self, X):
        values = dict()
        input_node = self.ordered_nodes[0]
        output_node = self.ordered_nodes[-1]
        values[input_node] = input_node._propogate(np.expand_dims(X, 0))
        for node in self.ordered_nodes[1:]:
            values[node] = node.propagate(values)
        return values[output_node]

    def backward(self, error):
        values = dict()
        output_node = self.ordered_nodes[-1]
        values[output_node] = output_node._backpropogate(np.expand_dims(error, 0))
        for node in reversed(self.ordered_nodes[:-1]):
            values[node] = node.backpropagate(values)
        return self.dW

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

        nodes = [node for node in self.ordered_nodes if node.size is not None]
        for node in nodes:
            full_name = str(node).lower()
            for s_name, o_name in node.serialized.items():
                key = full_name + '_' + s_name
                data = weights[key]
                target = getattr(node, o_name)
                target[:] = data
                used[key] = True

        for k, v in used.items():
            if not v:
                lament("WARNING: unused weight", k)

    def save_weights(self, fn, overwrite=False):
        import h5py
        f = h5py.File(fn, 'w')

        counts = defaultdict(lambda: 0)

        nodes = [node for node in self.ordered_nodes if node.size is not None]
        for node in nodes:
            full_name = str(node).lower()
            grp = f.create_group(full_name)
            for s_name, o_name in node.serialized.items():
                key = full_name + '_' + s_name
                target = getattr(node, o_name)
                data = grp.create_dataset(key, target.shape, dtype=_f)
                data[:] = target
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

    def reset(self):
        self.learner.reset(optim=True)
        self.en = 0
        self.bn = 0

    def measure(self, p, y):
        return self.mloss.forward(p, y)

    def derive(self, p, y):
        return self.loss.backward(p, y)

    def learn(self, inputs, outputs):
        predicted = self.model.forward(inputs)
        self.model.backward(self.derive(predicted, outputs))
        return predicted

    def update(self):
        self.learner.optim.update(self.model.dW, self.model.W)

    def prepare(self, model):
        self.en = 0
        self.bn = 0
        self.model = model

    def train_batched(self, inputs, outputs, batch_size,
                      return_losses=False, test_only=False):
        assert isinstance(return_losses, bool) or return_losses == 'both'

        if not test_only:
            self.en += 1

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

            if not test_only and self.learner.per_batch:
                self.learner.batch(b / batch_count)

            predicted = self.learn(batch_inputs, batch_outputs)
            if not test_only:
                self.update()

            if return_losses == 'both':
                batch_loss = self.loss.forward(predicted, batch_outputs)
                if np.isnan(batch_loss):
                    raise Exception("nan")
                losses.append(batch_loss)
                cumsum_loss += batch_loss

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
        if self.epoch + 1 >= self.epochs:
            return False
        if self.started:
            self.epoch += 1
        else:
            self.started = True
            self.epoch = self.epoch # poke property setter just in case
        return True

    def batch(self, progress): # TODO: rename
        # interpolates rates between epochs.
        # unlike epochs, we do not store batch number as a state.
        # i.e. calling next() will not respect progress.
        assert 0 <= progress <= 1
        self.rate = self.rate_at(self._epoch + progress)

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
                 expando=None):
        self.restart_epochs = int(epochs)
        self.decay = _f(restart_decay)
        self.restarts = int(restarts)
        self.restart_callback = callback
        # TODO: rename expando to something not insane
        self.expando = expando if expando is not None else lambda i: i

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
