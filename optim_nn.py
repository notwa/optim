#!/usr/bin/env python3

import numpy as np
nf = np.float32
nfa = lambda x: np.array(x, dtype=nf)
ni = np.int
nia = lambda x: np.array(x, dtype=ni)

from scipy.special import expit as sigmoid

from collections import defaultdict

# Initializations

# note: these are currently only implemented for 2D shapes.

def init_he_normal(size, ins, outs):
    s = np.sqrt(2 / ins)
    return np.random.normal(0, s, size=size)

def init_he_uniform(size, ins, outs):
    s = np.sqrt(6 / ins)
    return np.random.uniform(-s, s, size=size)

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

class SomethingElse(Loss):
    # generalizes Absolute and SquaredHalved
    # plot: https://www.desmos.com/calculator/fagjg9vuz7
    def __init__(self, a=4/3):
        assert 1 <= a <= 2, "parameter out of range"
        self.a = nf(a / 2)
        self.b = nf(2 / a)
        self.c = nf(2 / a - 1)

    def f(self, r):
        return self.a * np.abs(r)**self.b

    def df(self, r):
        return np.sign(r) * np.abs(r)**self.c

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
        self.dWprev[:] = V
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

        self.mt[:] = self.b1 * self.mt + (1 - self.b1) * dW
        self.vt[:] = self.b2 * self.vt + (1 - self.b2) * dW * dW

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

class Dense(Layer):
    def __init__(self, dim, init=init_he_uniform):
        super().__init__()
        self.dim = ni(dim)
        self.output_shape = (dim,)
        self.size = None
        self.weight_init = init

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

class Ritual:
    def __init__(self,
        optim=None,
        learn_rate=1e-3, learn_anneal=1, learn_advance=0,
        loss=None, mloss=None):
        self.optim = optim if optim is not None else SGD()
        self.loss = loss if loss is not None else Squared()
        self.mloss = mloss if mloss is not None else loss
        self.learn_rate = nf(learn_rate)
        self.learn_anneal = nf(learn_anneal)
        self.learn_advance = nf(learn_advance)

    def measure(self, residual):
        return self.mloss.mean(residual)

    def derive(self, residual):
        return self.loss.dmean(residual)

    def update(self, dW, W):
        self.optim.update(dW, W)

    def prepare(self, epoch):
        self.optim.alpha = self.learn_rate * self.learn_anneal**epoch

    def restart(self, optim=False):
        self.learn_rate *= self.learn_anneal**self.learn_advance
        if optim:
            self.optim.reset()

    def train_batched(self, model, inputs, outputs, batch_size, return_losses=False):
        cumsum_loss = 0
        batch_count = inputs.shape[0] // batch_size
        losses = []
        for b in range(batch_count):
            bi = b * batch_size
            batch_inputs  = inputs[ bi:bi+batch_size]
            batch_outputs = outputs[bi:bi+batch_size]

            predicted = model.forward(batch_inputs)
            residual = predicted - batch_outputs

            model.backward(self.derive(residual))
            self.update(model.dW, model.W)

            batch_loss = self.measure(residual)
            cumsum_loss += batch_loss
            if return_losses:
                losses.append(batch_loss)
        avg_loss = cumsum_loss / batch_count
        if return_losses:
            return avg_loss, losses
        else:
            return avg_loss

def multiresnet(x, width, depth, block=2, multi=1,
                activation=Relu, style='batchless',
                init=init_he_normal):
    y = x
    last_size = x.output_shape[0]

    for d in range(depth):
        size = width

        if last_size != size:
            y = y.feed(Dense(size, init))

        if style == 'batchless':
            skip = y
            merger = Sum()
            skip.feed(merger)
            z_start = skip.feed(activation())
            for i in range(multi):
                z = z_start
                for i in range(block):
                    if i > 0:
                        z = z.feed(activation())
                    z = z.feed(Dense(size, init))
                z.feed(merger)
            y = merger
        elif style == 'onelesssum':
            is_last = d + 1 == depth
            needs_sum = not is_last or multi > 1
            skip = y
            if needs_sum:
                merger = Sum()
            if not is_last:
                skip.feed(merger)
            z_start = skip.feed(activation())
            for i in range(multi):
                z = z_start
                for i in range(block):
                    if i > 0:
                        z = z.feed(activation())
                    z = z.feed(Dense(size, init))
                if needs_sum:
                    z.feed(merger)
            if needs_sum:
                y = merger
            else:
                y = z
        else:
            raise Exception('unknown resnet style', style)

        last_size = size

    return y

inits = dict(he_normal=init_he_normal, he_uniform=init_he_uniform)
activations = dict(sigmoid=Sigmoid, tanh=Tanh, relu=Relu, elu=Elu, gelu=GeluApprox)

def run(program, args=[]):
    import sys
    lament = lambda *args, **kwargs: print(*args, file=sys.stderr, **kwargs)
    def log(left, right):
        lament("{:>20}:   {}".format(left, right))

    # Config

    from dotmap import DotMap
    config = DotMap(
        fn_load = None,
        fn_save = 'optim_nn.h5',
        log_fn = 'losses.npz',

        # multi-residual network parameters
        res_width = 49,
        res_depth = 1,
        res_block = 4, # normally 2 for plain resnet
        res_multi = 1, # normally 1 for plain resnet

        # style of resnet (order of layers, which layers, etc.)
        parallel_style = 'onelesssum',
        activation = 'gelu',

        optim = 'adam',
        nesterov = False, # only used with SGD or Adam
        momentum = 0.33, # only used with SGD

        # learning parameters: SGD with restarts (kinda)
        learn = 1e-2,
        epochs = 24,
        learn_halve_every = 16,
        restarts = 2,
        learn_restart_advance = 16,

        # misc
        batch_size = 64,
        init = 'he_normal',
        loss = SomethingElse(4/3),
        mloss = 'mse',
        restart_optim = True, # restarts also reset internal state of optimizer
        unsafe = True, # aka gotta go fast mode
        train_compare = None,
        #valid_compare = 0.0007159,
        valid_compare = 0.0000946,
    )

    config.pprint()

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

    init = inits[config.init]
    activation = activations[config.activation]

    x = Input(shape=(input_samples,))
    y = x
    y = multiresnet(y,
                    config.res_width, config.res_depth,
                    config.res_block, config.res_multi,
                    activation=activation, init=init)
    if y.output_shape[0] != output_samples:
        y = y.feed(Dense(output_samples, init))

    model = Model(x, y, unsafe=config.unsafe)

    node_names = ' '.join([str(node) for node in model.ordered_nodes])
    log('{} nodes'.format(len(model.ordered_nodes)), node_names)
    log('parameters', model.param_count)

    training = config.epochs > 0 and config.restarts >= 0

    if config.fn_load is not None:
        log('loading weights', config.fn_load)
        model.load_weights(config.fn_load)

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

    def lookup_loss(maybe_name):
        if isinstance(maybe_name, Loss):
            return maybe_name
        elif maybe_name == 'mse':
            return Squared()
        elif maybe_name == 'mshe': # mushy
            return SquaredHalved()
        raise Exception('unknown objective', maybe_name)

    loss = lookup_loss(config.loss)
    mloss = lookup_loss(config.mloss) if config.mloss else loss

    anneal = 0.5**(1/config.learn_halve_every)
    ritual = Ritual(optim=optim,
                    learn_rate=config.learn, learn_anneal=anneal,
                    learn_advance=config.learn_restart_advance,
                    loss=loss, mloss=mloss)

    learn_end = config.learn * (anneal**config.learn_restart_advance)**config.restarts * anneal**(config.epochs - 1)
    log("final learning rate", "{:10.8f}".format(learn_end))

    # Training

    batch_losses = []
    train_losses = []
    valid_losses = []

    def measure_error():
        def print_error(name, inputs, outputs, comparison=None):
            predicted = model.forward(inputs)
            residual = predicted - outputs
            err = ritual.measure(residual)
            log(name + " loss", "{:11.7f}".format(err))
            if comparison:
                log("improvement", "{:+7.2f}%".format((comparison / err - 1) * 100))
            return err

        train_err = print_error("train",
                                inputs / x_scale, outputs / y_scale,
                                config.train_compare)
        valid_err = print_error("valid",
                                valid_inputs / x_scale, valid_outputs / y_scale,
                                config.valid_compare)
        train_losses.append(train_err)
        valid_losses.append(valid_err)

    for i in range(config.restarts + 1):
        measure_error()

        if i > 0:
            log("restarting", i)
            ritual.restart(optim=config.restart_optim)

        assert inputs.shape[0] % config.batch_size == 0, \
               "inputs is not evenly divisible by batch_size" # TODO: lift this restriction
        for e in range(config.epochs):
            indices = np.arange(inputs.shape[0])
            np.random.shuffle(indices)
            shuffled_inputs = inputs[indices] / x_scale
            shuffled_outputs = outputs[indices] / y_scale

            ritual.prepare(e)
            #log("learning rate", "{:10.8f}".format(ritual.optim.alpha))

            avg_loss, losses = ritual.train_batched(model,
                shuffled_inputs, shuffled_outputs,
                config.batch_size,
                return_losses=True)
            log("average loss", "{:11.7f}".format(avg_loss))
            batch_losses += losses

    measure_error()

    if config.fn_save is not None:
        log('saving weights', config.fn_save)
        model.save_weights(config.fn_save, overwrite=True)

    # Evaluation

    a = (192, 128, 64)
    b = (64, 128, 192)
    X = np.expand_dims(np.hstack((a, b)), 0) / x_scale
    P = model.forward(X) * y_scale
    log("truth", rgbcompare(a, b))
    log("network", np.squeeze(P))

    if config.log_fn is not None:
        np.savez_compressed(config.log_fn,
                            batch_losses=nfa(batch_losses),
                            train_losses=nfa(train_losses),
                            valid_losses=nfa(valid_losses))

    return 0

if __name__ == '__main__':
    import sys
    sys.exit(run(sys.argv[0], sys.argv[1:]))
