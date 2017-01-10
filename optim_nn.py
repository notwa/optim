#!/usr/bin/env python3

import numpy as np
nf = np.float32
nfa = lambda x: np.array(x, dtype=nf)
ni = np.int
nia = lambda x: np.array(x, dtype=ni)

from collections import defaultdict

# Loss functions

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

class SquaredHalved(Loss):
    def f(self, r):
        return np.square(r) / 2

    def df(self, r):
        return r

# Optimizers

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
        self.dWprev = V
        if self.nesterov:
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

        self.mt = self.b1 * self.mt + (1 - self.b1) * dW
        self.vt = self.b2 * self.vt + (1 - self.b2) * dW * dW

        return -self.alpha * (self.mt / (1 - self.b1_t)) \
                   / np.sqrt((self.vt / (1 - self.b2_t)) + self.eps)

# Abstract Layers

_layer_counters = defaultdict(lambda: 0)

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
        assert shape is not None
        if self.output_shape is None:
            self.output_shape = shape
        return shape

    # TODO: rename this multi and B crap to something actually relevant.

    def multi(self, B):
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
        assert len(self.parents) > 0, self
        #print("      forwarding", self)
        B = []
        for parent in self.parents:
            # TODO: skip over irrelevant nodes (if any)
            X = lut[parent]
            #print("collected parent", parent)
            self.validate_input(X)
            B.append(X)
        Y = self.multi(B)
        self.validate_output(Y)
        return Y

    def backward(self, lut):
        assert len(self.children) > 0, self
        #print("     backwarding", self)
        dB = []
        for child in self.children:
            # TODO: skip over irrelevant nodes (if any)
            dY = lut[child]
            #print(" collected child", child)
            self.validate_output(dY)
            dB.append(dY)
        dX = self.dmulti(dB)
        self.validate_input(dX)
        return dX

# Final Layers

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
        from scipy.special import expit as sigmoid
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

class GeluApprox(Layer):
    # paper: https://arxiv.org/abs/1606.08415
    #  plot: https://www.desmos.com/calculator/ydzgtccsld
    def F(self, X):
        from scipy.special import expit as sigmoid
        self.a = 1.704 * X
        self.sig = sigmoid(self.a)
        return X * self.sig

    def dF(self, dY):
        return dY * self.sig * (1 + self.a * (1 - self.sig))

class Dense(Layer):
    def __init__(self, dim):
        super().__init__()
        self.dim = ni(dim)
        self.output_shape = (dim,)
        self.size = None

    def init(self, W, dW):
        ins, outs = self.input_shape[0], self.output_shape[0]

        self.W = W
        self.dW = dW
        self.coeffs = self.W[:self.nW].reshape(ins, outs)
        self.biases = self.W[self.nW:].reshape(1, outs)
        self.dcoeffs = self.dW[:self.nW].reshape(ins, outs)
        self.dbiases = self.dW[self.nW:].reshape(1, outs)

        # he_normal initialization
        s = np.sqrt(2 / ins)
        self.coeffs.flat = np.random.normal(0, s, size=self.nW)
        self.biases.flat = 0

    def make_shape(self, shape):
        super().make_shape(shape)
        if len(shape) != 1:
            return False
        self.nW = self.dim * shape[0]
        self.nb = self.dim
        self.size = self.nW + self.nb
        return shape

    def F(self, X):
        self.X = X
        Y = X.dot(self.coeffs) \
          + self.biases
        return Y

    def dF(self, dY):
        dX = dY.dot(self.coeffs.T)
        self.dcoeffs[:] = self.X.T.dot(dY)
        self.dbiases[:] = np.sum(dY, axis=0, keepdims=True)
        return dX

# Model

class Model:
    def __init__(self, x, y):
        assert isinstance(x, Layer), x
        assert isinstance(y, Layer), y
        self.x = x
        self.y = y

        self.ordered_nodes = self.traverse([], self.y)
        node_names = ' '.join([str(node) for node in self.ordered_nodes])
        print('{} nodes: {}'.format(len(self.ordered_nodes), node_names))

        self.make_weights()

    def make_weights(self):
        self.param_count = 0
        for node in self.ordered_nodes:
            if node.size is not None:
                self.param_count += node.size
        print(self.param_count)
        self.W  = np.zeros(self.param_count, dtype=nf)
        self.dW = np.zeros(self.param_count, dtype=nf)

        offset = 0
        for node in self.ordered_nodes:
            if node.size is not None:
                end = offset + node.size
                node.init(self.W[offset:end], self.dW[offset:end])
                offset += node.size

        #print(self.W, self.dW)

    def traverse(self, nodes, node):
        if node == x:
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
        # seemingly compatible with keras models at the moment
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
            denses[a].coeffs = weights[b_name+'_W']
            denses[a].biases = np.expand_dims(weights[b_name+'_b'], 0)

    def save_weights(self, fn, overwrite=False):
        raise NotImplementedError("unimplemented", self)

if __name__ == '__main__':
    # Config

    from dotmap import DotMap
    config = DotMap(
        fn = 'ml/cie_mlp_min.h5',

        # multi-residual network parameters
        res_width = 12,
        res_depth = 3,
        res_block = 2, # normally 2 for plain resnet
        res_multi = 4, # normally 1 for plain resnet

        # style of resnet
        # only one is implemented so far
        parallel_style = 'batchless',
        activation = 'relu',

        optim = 'adam',
        nesterov = False, # only used with SGD or Adam
        momentum = 0.33, # only used with SGD

        # learning parameters: SGD with restarts
        LR = 1e-2,
        epochs = 6,
        LR_halve_every = 2,
        restarts = 3,
        LR_restart_advance = 3,

        # misc
        batch_size = 64,
        init = 'he_normal',
        loss = 'mse',
    )

    # toy CIE-2000 data
    from ml.cie_mlp_data import rgbcompare, input_samples, output_samples, x_scale, y_scale

    def read_data(fn):
        data = np.load(fn)
        try:
            inputs, outputs = data['inputs'], data['outputs']
        except KeyError:
            # because i'm bad at video games.
            inputs, outputs = data['arr_0'], data['arr_1']
        return inputs, outputs

    inputs, outputs = read_data("ml/cie_mlp_data.npz")
    valid_inputs, valid_outputs = read_data("ml/cie_mlp_vdata.npz")

    # Our Test Model

    x = Input(shape=(input_samples,))
    y = x
    last_size = input_samples

    activations = dict(sigmoid=Sigmoid, tanh=Tanh, relu=Relu, gelu=GeluApprox)
    activation = activations[config.activation]

    for blah in range(config.res_depth):
        size = config.res_width

        if last_size != size:
            y = y.feed(Dense(size))

        assert config.parallel_style == 'batchless'
        skip = y
        merger = Sum()
        skip.feed(merger)
        z_start = skip.feed(activation())
        for i in range(config.res_multi):
            z = z_start
            for i in range(config.res_block):
                if i > 0:
                    z = z.feed(activation())
                z = z.feed(Dense(size))
            z.feed(merger)
        y = merger

        last_size = size

    if last_size != output_samples:
        y = y.feed(Dense(output_samples))

    model = Model(x, y)

    training = config.epochs > 0 and config.restarts >= 0

    if not training:
        model.load_weights(config.fn)

    if config.optim == 'adam':
        assert not config.nesterov, "unimplemented"
        optim = Adam()
    elif config.optim == 'sgd':
        if config.momentum != 0:
            optim = Momentum(mu=config.momentum, nesterov=config.nesterov)
        else:
            optim = Optimizer()
    else:
        raise Exception('unknown optimizer', config.optim)

    if config.loss == 'mse':
        loss = Squared()
    elif config.loss == 'mshe': # mushy
        loss = SquaredHalved()
    else:
        raise Exception('unknown objective', config.loss)

    LR = config.LR
    LRprod = 0.5**(1/config.LR_halve_every)

    # Training

    def measure_loss():
        predicted = model.forward(inputs / x_scale)
        residual = predicted - outputs / y_scale
        err = loss.mean(residual)
        print("train loss: {:11.7f}".format(err))
        print("improvement: {:+7.2f}%".format((0.0007031 / err - 1) * 100))

        predicted = model.forward(valid_inputs / x_scale)
        residual = predicted - valid_outputs / y_scale
        err = loss.mean(residual)
        print("valid loss: {:11.7f}".format(err))
        print("improvement: {:+7.2f}%".format((0.0007159 / err - 1) * 100))

    for i in range(config.restarts + 1):
        measure_loss()

        if i > 0:
            print("restarting")

        assert inputs.shape[0] % config.batch_size == 0, \
               "inputs is not evenly divisible by batch_size" # TODO: lift this restriction
        batch_count = inputs.shape[0] // config.batch_size
        for e in range(config.epochs):
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
            shuffled_inputs = inputs[indices] / x_scale
            shuffled_outputs = outputs[indices] / y_scale

            optim.alpha = LR * LRprod**e

            cumsum_loss = 0
            for b in range(batch_count):
                bi = b * config.batch_size
                batch_inputs  = shuffled_inputs[ bi:bi+config.batch_size]
                batch_outputs = shuffled_outputs[bi:bi+config.batch_size]

                predicted = model.forward(batch_inputs)
                residual = predicted - batch_outputs
                dW = model.backward(loss.dmean(residual))
                optim.update(dW, model.W)

                # note: we don't actually need this for training, only monitoring.
                cumsum_loss += loss.mean(residual)
            print("avg loss: {:10.6f}".format(cumsum_loss / batch_count))

        LR *= LRprod**config.LR_restart_advance

    measure_loss()

    #if training:
    #    model.save_weights(config.fn, overwrite=True)

    # Evaluation

    a = (192, 128, 64)
    b = (64, 128, 192)
    X = np.expand_dims(np.hstack((a, b)), 0) / x_scale
    P = model.forward(X) * y_scale
    print("truth:", rgbcompare(a, b))
    print("network:", np.squeeze(P))
