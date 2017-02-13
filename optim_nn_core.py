import numpy as np
# ugly shorthand:
nf = np.float32
nfa = lambda x: np.array(x, dtype=nf)
ni = np.int
nia = lambda x: np.array(x, dtype=ni)

# just for speed, not strictly essential:
from scipy.special import expit as sigmoid

# used for numbering layers like Keras:
from collections import defaultdict
_layer_counters = defaultdict(lambda: 0)

# Initializations {{{1

# note: these are currently only implemented for 2D shapes.

def init_he_normal(size, ins, outs):
    s = np.sqrt(2 / ins)
    return np.random.normal(0, s, size=size)

def init_he_uniform(size, ins, outs):
    s = np.sqrt(6 / ins)
    return np.random.uniform(-s, s, size=size)

# Loss functions {{{1

class Loss:
    def mean(self, r):
        return np.average(self.f(r))

    def dmean(self, r):
        d = self.df(r)
        return d / len(d)

class Squared(Loss):
    def f(self, r):
        return np.square(r)

    def df(self, r):
        return 2 * r

class Absolute(Loss):
    def f(self, r):
        return np.abs(r)

    def df(self, r):
        return np.sign(r)

# Optimizers {{{1

class Optimizer:
    def __init__(self, alpha=0.1):
        self.alpha = nf(alpha)
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
    def __init__(self, alpha=0.01, lamb=0, mu=0.9, nesterov=False):
        self.alpha = np.asfarray(alpha) # learning rate
        self.lamb = np.asfarray(lamb) # weight decay
        self.mu = np.asfarray(mu) # momentum
        self.nesterov = bool(nesterov)

        self.reset()

    def reset(self):
        self.dWprev = None

    def compute(self, dW, W):
        if self.dWprev is None:
            #self.dWprev = np.zeros_like(dW)
            self.dWprev = np.copy(dW)

        V = self.mu * self.dWprev - self.alpha * (dW + W * self.lamb)
        self.dWprev[:] = V
        if self.nesterov: # TODO: is this correct? looks weird
            return self.mu * V - self.alpha * (dW + W * self.lamb)
        else:
            return V

class Adam(Optimizer):
    def __init__(self, alpha=0.001, b1=0.9, b2=0.999, b1_t=0.9, b2_t=0.999, eps=1e-8):
        self.alpha = nf(alpha) # learning rate
        self.b1 = nf(b1) # decay term
        self.b2 = nf(b2) # decay term
        self.b1_t_default = nf(b1_t) # decay term power t
        self.b2_t_default = nf(b2_t) # decay term power t
        self.eps = nf(eps)

        self.reset()

    def reset(self):
        self.mt = None
        self.vt = None
        self.b1_t = self.b1_t_default
        self.b2_t = self.b2_t_default

    def compute(self, dW, W):
        if self.mt is None:
            self.mt = np.zeros_like(W)
        if self.vt is None:
            self.vt = np.zeros_like(W)

        # decay
        self.b1_t *= self.b1
        self.b2_t *= self.b2

        self.mt[:] = self.b1 * self.mt + (1 - self.b1) * dW
        self.vt[:] = self.b2 * self.vt + (1 - self.b2) * dW * dW

        return -self.alpha * (self.mt / (1 - self.b1_t)) \
                   / np.sqrt((self.vt / (1 - self.b2_t)) + self.eps)

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

    def F(self, X):
        raise NotImplementedError("unimplemented", self)

    def dF(self, dY):
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

    # TODO: rename this multi and B crap to something actually relevant.

    def multi(self, B):
        if not self.unsafe:
            assert len(B) == 1, self
        return self.F(B[0])

    def dmulti(self, dB):
        if len(dB) == 1:
            return self.dF(dB[0])
        else:
            dX = None
            for dY in dB:
                if dX is None:
                    dX = self.dF(dY)
                else:
                    dX += self.dF(dY)
            return dX

    # general utility methods:

    def compatible(self, parent):
        if self.input_shape is None:
            # inherit shape from output
            shape = self.make_shape(parent.output_shape)
            if shape is None:
                return False
            self.input_shape = shape
        if np.all(self.input_shape == parent.output_shape):
            return True
        else:
            return False

    def feed(self, child):
        if not child.compatible(self):
            fmt = "{} is incompatible with {}: shape mismatch: {} vs. {}"
            raise Exception(fmt.format(self, child, self.output_shape, child.input_shape))
        self.do_feed(child)
        child.be_fed(self)
        return child

    def validate_input(self, X):
        assert X.shape[1:] == self.input_shape,  (str(self), X.shape[1:], self.input_shape)

    def validate_output(self, Y):
        assert Y.shape[1:] == self.output_shape, (str(self), Y.shape[1:], self.output_shape)

    def forward(self, lut):
        if not self.unsafe:
            assert len(self.parents) > 0, self
        B = []
        for parent in self.parents:
            # TODO: skip over irrelevant nodes (if any)
            X = lut[parent]
            if not self.unsafe:
                self.validate_input(X)
            B.append(X)
        Y = self.multi(B)
        if not self.unsafe:
            self.validate_output(Y)
        return Y

    def backward(self, lut):
        if not self.unsafe:
            assert len(self.children) > 0, self
        dB = []
        for child in self.children:
            # TODO: skip over irrelevant nodes (if any)
            dY = lut[child]
            if not self.unsafe:
                self.validate_output(dY)
            dB.append(dY)
        dX = self.dmulti(dB)
        if not self.unsafe:
            self.validate_input(dX)
        return dX

# Nonparametric Layers {{{1

class Sum(Layer):
    def multi(self, B):
        return np.sum(B, axis=0)

    def dmulti(self, dB):
        #assert len(dB) == 1, "unimplemented"
        return dB[0] # TODO: does this always work?

class Input(Layer):
    def __init__(self, shape):
        assert shape is not None
        super().__init__()
        self.shape = tuple(shape)
        self.input_shape = self.shape
        self.output_shape = self.shape

    def F(self, X):
        return X

    def dF(self, dY):
        #self.dY = dY
        return np.zeros_like(dY)

class Affine(Layer):
    def __init__(self, a=1, b=0):
        super().__init__()
        self.a = nf(a)
        self.b = nf(b)

    def F(self, X):
        return self.a * X + self.b

    def dF(self, dY):
        return dY * self.a

class Sigmoid(Layer): # aka Logistic
    def F(self, X):
        self.sig = sigmoid(X)
        return X * self.sig

    def dF(self, dY):
        return dY * self.sig * (1 - self.sig)

class Tanh(Layer):
    def F(self, X):
        self.sig = np.tanh(X)
        return X * self.sig

    def dF(self, dY):
        return dY * (1 - self.sig * self.sig)

class Relu(Layer):
    def F(self, X):
        self.cond = X >= 0
        return np.where(self.cond, X, 0)

    def dF(self, dY):
        return np.where(self.cond, dY, 0)

class Elu(Layer):
    # paper: https://arxiv.org/abs/1511.07289

    def __init__(self, alpha=1):
        super().__init__()
        self.alpha = nf(alpha)

    def F(self, X):
        self.cond = X >= 0
        self.neg = np.exp(X) - 1
        return np.where(self.cond, X, self.neg)

    def dF(self, dY):
        return dY * np.where(self.cond, 1, self.neg + 1)

class GeluApprox(Layer):
    # paper: https://arxiv.org/abs/1606.08415
    #  plot: https://www.desmos.com/calculator/ydzgtccsld

    def F(self, X):
        self.a = 1.704 * X
        self.sig = sigmoid(self.a)
        return X * self.sig

    def dF(self, dY):
        return dY * self.sig * (1 + self.a * (1 - self.sig))

# Parametric Layers {{{1

class Dense(Layer):
    def __init__(self, dim, init=init_he_uniform):
        super().__init__()
        self.dim = ni(dim)
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
        ins, outs = self.input_shape[0], self.output_shape[0]

        self.W = W
        self.dW = dW
        self.coeffs = self.W[:self.nW].reshape(ins, outs)
        self.biases = self.W[self.nW:].reshape(1, outs)
        self.dcoeffs = self.dW[:self.nW].reshape(ins, outs)
        self.dbiases = self.dW[self.nW:].reshape(1, outs)

        self.coeffs.flat = self.weight_init(self.nW, ins, outs)
        self.biases.flat = 0

        self.std = np.std(self.W)

    def F(self, X):
        self.X = X
        Y = X.dot(self.coeffs) + self.biases
        return Y

    def dF(self, dY):
        dX = dY.dot(self.coeffs.T)
        self.dcoeffs[:] = self.X.T.dot(dY)
        self.dbiases[:] = dY.sum(0, keepdims=True)
        return dX

class DenseOneLess(Dense):
    def init(self, W, dW):
        super().init(W, dW)
        ins, outs = self.input_shape[0], self.output_shape[0]
        assert ins == outs, (ins, outs)

    def F(self, X):
        np.fill_diagonal(self.coeffs, 0)
        self.X = X
        Y = X.dot(self.coeffs) + self.biases
        return Y

    def dF(self, dY):
        dX = dY.dot(self.coeffs.T)
        self.dcoeffs[:] = self.X.T.dot(dY)
        self.dbiases[:] = dY.sum(0, keepdims=True)
        np.fill_diagonal(self.dcoeffs, 0)
        return dX

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
        self.W  = np.zeros(self.param_count, dtype=nf)
        self.dW = np.zeros(self.param_count, dtype=nf)

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
        lut = dict()
        input_node = self.ordered_nodes[0]
        output_node = self.ordered_nodes[-1]
        lut[input_node] = input_node.multi(np.expand_dims(X, 0))
        for node in self.ordered_nodes[1:]:
            lut[node] = node.forward(lut)
        return lut[output_node]

    def backward(self, error):
        lut = dict()
        input_node = self.ordered_nodes[0]
        output_node = self.ordered_nodes[-1]
        lut[output_node] = output_node.dmulti(np.expand_dims(error, 0))
        for node in reversed(self.ordered_nodes[:-1]):
            lut[node] = node.backward(lut)
        #return lut[input_node] # meaningless value
        return self.dW

    def load_weights(self, fn):
        # seemingly compatible with keras' Dense layers.
        # ignores any non-Dense layer types.
        # TODO: assert file actually exists
        import h5py
        f = h5py.File(fn)
        weights = {}
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                weights[name.split('/')[-1]] = nfa(obj[:])
        f.visititems(visitor)
        f.close()

        denses = [node for node in self.ordered_nodes if isinstance(node, Dense)]
        for i in range(len(denses)):
            a, b = i, i + 1
            b_name = "dense_{}".format(b)
            # TODO: write a Dense method instead of assigning directly
            denses[a].coeffs[:] = weights[b_name+'_W']
            denses[a].biases[:] = np.expand_dims(weights[b_name+'_b'], 0)

    def save_weights(self, fn, overwrite=False):
        import h5py
        f = h5py.File(fn, 'w')

        denses = [node for node in self.ordered_nodes if isinstance(node, Dense)]
        for i in range(len(denses)):
            a, b = i, i + 1
            b_name = "dense_{}".format(b)
            # TODO: write a Dense method instead of assigning directly
            grp = f.create_group(b_name)
            data = grp.create_dataset(b_name+'_W', denses[a].coeffs.shape, dtype=nf)
            data[:] = denses[a].coeffs
            data = grp.create_dataset(b_name+'_b', denses[a].biases.shape, dtype=nf)
            data[:] = denses[a].biases

        f.close()

# Rituals {{{1

class Ritual: # i'm just making up names at this point
    def __init__(self, learner=None, loss=None, mloss=None):
        self.learner = learner if learner is not None else Learner(Optimizer())
        self.loss = loss if loss is not None else Squared()
        self.mloss = mloss if mloss is not None else loss

    def reset(self):
        self.learner.reset(optim=True)

    def measure(self, residual):
        return self.mloss.mean(residual)

    def derive(self, residual):
        return self.loss.dmean(residual)

    def learn(self, inputs, outputs):
        predicted = self.model.forward(inputs)
        residual = predicted - outputs
        self.model.backward(self.derive(residual))
        return residual

    def update(self):
        self.learner.optim.update(self.model.dW, self.model.W)

    def prepare(self, model):
        self.en = 0
        self.bn = 0
        self.model = model

    def train_batched(self, inputs, outputs, batch_size, return_losses=False):
        self.en += 1
        cumsum_loss = 0
        batch_count = inputs.shape[0] // batch_size
        losses = []
        for b in range(batch_count):
            self.bn += 1
            bi = b * batch_size
            batch_inputs  = inputs[ bi:bi+batch_size]
            batch_outputs = outputs[bi:bi+batch_size]

            if self.learner.per_batch:
                self.learner.batch(b / batch_count)

            residual = self.learn(batch_inputs, batch_outputs)
            self.update()

            batch_loss = self.measure(residual)
            if np.isnan(batch_loss):
                raise Exception("nan")
            cumsum_loss += batch_loss
            if return_losses:
                losses.append(batch_loss)
        avg_loss = cumsum_loss / batch_count
        if return_losses:
            return avg_loss, losses
        else:
            return avg_loss

# Learners {{{1

class Learner:
    per_batch = False

    def __init__(self, optim, epochs=100, rate=None):
        assert isinstance(optim, Optimizer)
        self.optim = optim
        self.start_rate = optim.alpha if rate is None else float(rate)
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
        self.halve_every = float(halve_every)
        self.anneal = 0.5**(1/self.halve_every)
        super().__init__(optim, epochs, rate)

    def rate_at(self, epoch):
        return self.start_rate * self.anneal**epoch

def cosmod(x):
    # plot: https://www.desmos.com/calculator/hlgqmyswy2
    return (1 + np.cos((x % 1) * np.pi)) / 2

class SGDR(Learner):
    # Stochastic Gradient Descent with Restarts
    # paper: https://arxiv.org/abs/1608.03983
    # NOTE: this is missing a couple features.

    per_batch = True

    def __init__(self, optim, epochs=100, rate=None,
                 restarts=0, restart_decay=0.5, callback=None,
                 expando=None):
        self.restart_epochs = int(epochs)
        self.decay = float(restart_decay)
        self.restarts = int(restarts)
        self.restart_callback = callback
        # TODO: rename expando to something not insane
        self.expando = expando if expando is not None else lambda i: 1

        self.splits = []
        epochs = 0
        for i in range(0, self.restarts + 1):
            split = epochs + int(self.restart_epochs * self.expando(i))
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
        x = sub_epoch / next_restart
        return self.start_rate * self.decay**restart * cosmod(x)

    def next(self):
        if not super().next():
            return False
        restart, sub_epoch, next_restart = self.split_num(self.epoch)
        if restart > 0 and sub_epoch == 0:
            if self.restart_callback is not None:
                self.restart_callback(restart)
        return True
