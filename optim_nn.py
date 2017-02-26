#!/usr/bin/env python3

# external packages required for full functionality:
# numpy scipy h5py sklearn dotmap

# BIG TODO: ensure numpy isn't upcasting to float64 *anywhere*.
#           this is gonna take some work.

from optim_nn_core import *
from optim_nn_core import _check, _f

import sys

def lament(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

_log_was_update = False
def log(left, right, update=False):
    s = "\x1B[1m  {:>20}:\x1B[0m   {}".format(left, right)
    global _log_was_update
    if update and _log_was_update:
        lament('\x1B[F' + s)
    else:
        lament(s)
    _log_was_update = update

class Dummy:
    pass

# Loss functions {{{1

class SquaredHalved(ResidualLoss):
    def f(self, r):
        return np.square(r) / 2

    def df(self, r):
        return r

class SomethingElse(ResidualLoss):
    # generalizes Absolute and SquaredHalved
    # plot: https://www.desmos.com/calculator/fagjg9vuz7
    def __init__(self, a=4/3):
        assert 1 <= a <= 2, "parameter out of range"
        self.a = _f(a / 2)
        self.b = _f(2 / a)
        self.c = _f(2 / a - 1)

    def f(self, r):
        return self.a * np.abs(r)**self.b

    def df(self, r):
        return np.sign(r) * np.abs(r)**self.c

# Nonparametric Layers {{{1

# Parametric Layers {{{1

class LayerNorm(Layer):
    # paper: https://arxiv.org/abs/1607.06450
    # note: nonparametric when affine == False

    def __init__(self, eps=1e-5, affine=True):
        super().__init__()
        self.eps = _f(eps)
        self.affine = bool(affine)
        self.size = None

    def make_shape(self, shape):
        super().make_shape(shape)
        if len(shape) != 1:
            return False
        self.features = shape[0]
        if self.affine:
            self.size = 2 * self.features
        return shape

    def init(self, W, dW):
        super().init(W, dW)

        f = self.features

        self.gamma, self.dgamma = self.W[0*f:1*f], self.dW[0*f:1*f]
        self.beta, self.dbeta   = self.W[1*f:2*f], self.dW[1*f:2*f]

        self.gamma[:] = 1
        self.beta[:] = 0

    def F(self, X):
        self.mean = X.mean(0)
        self.center = X - self.mean
        self.var = self.center.var(0) + self.eps
        self.std = np.sqrt(self.var)

        self.Xnorm = self.center / self.std
        if self.affine:
            return self.gamma * self.Xnorm + self.beta
        return self.Xnorm

    def dF(self, dY):
        length = dY.shape[0]

        if self.affine:
            dXnorm = dY * self.gamma
            self.dgamma[:] = (dY * self.Xnorm).sum(0)
            self.dbeta[:] = dY.sum(0)
        else:
            dXnorm = dY

        dstd = (dXnorm * self.center).sum(0) / -self.var
        dcenter = dXnorm / self.std + dstd / self.std * self.center / length
        dmean = -dcenter.sum(0)
        dX = dcenter + dmean / length

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

class CosineDense(Dense):
    # paper: https://arxiv.org/abs/1702.05870
    # another implementation: https://github.com/farizrahman4u/keras-contrib/pull/36
    # the paper doesn't mention bias,
    # so we treat bias as an additional weight with a constant input of 1.
    # this is correct in Dense layers, so i hope it's correct here too.

    eps = 1e-4

    def F(self, X):
        self.X = X
        self.X_norm = np.sqrt(np.square(X).sum(-1, keepdims=True) \
          + 1 + self.eps)
        self.W_norm = np.sqrt(np.square(self.coeffs).sum(0, keepdims=True) \
          + np.square(self.biases) + self.eps)
        self.dot = X.dot(self.coeffs) + self.biases
        Y = self.dot / (self.X_norm * self.W_norm)
        return Y

    def dF(self, dY):
        ddot = dY / self.X_norm / self.W_norm
        dX_norm = -(dY * self.dot / self.W_norm).sum(-1, keepdims=True) / self.X_norm**2
        dW_norm = -(dY * self.dot / self.X_norm).sum( 0, keepdims=True) / self.W_norm**2

        self.dcoeffs[:] = self.X.T.dot(ddot)         \
          + dW_norm / self.W_norm * self.coeffs
        self.dbiases[:] = ddot.sum(0, keepdims=True) \
          + dW_norm / self.W_norm * self.biases
        dX = ddot.dot(self.coeffs.T) + dX_norm / self.X_norm * self.X

        return dX

# Rituals {{{1

def stochastic_multiply(W, gamma=0.5, allow_negation=True):
    # paper: https://arxiv.org/abs/1606.01981

    assert W.ndim == 1, W.ndim
    assert 0 < gamma < 1, gamma
    size = len(W)
    alpha = np.max(np.abs(W))
    # NOTE: numpy gives [low, high) but the paper advocates [low, high]
    mult = np.random.uniform(gamma, 1/gamma, size=size)
    if allow_negation: # TODO: verify this is correct. seems to wreak havok.
        prob = (W / alpha + 1) / 2
        samples = np.random.random_sample(size=size)
        mult *= np.where(samples < prob, 1, -1)
    np.multiply(W, mult, out=W)

class StochMRitual(Ritual):
    # paper: https://arxiv.org/abs/1606.01981
    # this probably doesn't make sense for regression problems,
    # let alone small models, but here it is anyway!

    def __init__(self, learner=None, loss=None, mloss=None, gamma=0.5):
        super().__init__(learner, loss, mloss)
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
                stochastic_multiply(layer.coeffs.ravel(), gamma=self.gamma,
                                    allow_negation=True)
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
    def __init__(self, learner=None, loss=None, mloss=None,
                 input_noise=0, output_noise=0, gradient_noise=0):
        self.input_noise = _f(input_noise)
        self.output_noise = _f(output_noise)
        self.gradient_noise = _f(gradient_noise)
        super().__init__(learner, loss, mloss)

    def learn(self, inputs, outputs):
        # this is pretty crude
        s = self.input_noise
        noisy_inputs =   inputs + np.random.normal(0, s, size=inputs.shape)
        s = self.output_noise
        noisy_outputs = outputs + np.random.normal(0, s, size=outputs.shape)
        return super().learn(noisy_inputs, noisy_outputs)

    def update(self):
        # gradient noise paper: https://arxiv.org/abs/1511.06807
        if self.gradient_noise > 0:
            size = len(self.model.dW)
            gamma = 0.55
            s = self.gradient_noise / (1 + self.bn) ** gamma
            # experiments:
            #s = np.sqrt(self.learner.rate)
            #s = np.square(self.learner.rate)
            #s = self.learner.rate / self.en
            self.model.dW += np.random.normal(0, s, size=size)
        super().update()

# Learners {{{1

class DumbLearner(AnnealingLearner):
    # this is my own awful contraption. it's not really "SGD with restarts".
    def __init__(self, optim, epochs=100, rate=None, halve_every=10,
                 restarts=0, restart_advance=20, callback=None):
        self.restart_epochs = int(epochs)
        self.restarts = int(restarts)
        self.restart_advance = float(restart_advance)
        self.restart_callback = callback
        epochs = self.restart_epochs * (self.restarts + 1)
        super().__init__(optim, epochs, rate, halve_every)

    def rate_at(self, epoch):
        sub_epoch = epoch % self.restart_epochs
        restart = epoch // self.restart_epochs
        return super().rate_at(sub_epoch) * (self.anneal**self.restart_advance)**restart

    def next(self):
        if not super().next():
            return False
        sub_epoch = self.epoch % self.restart_epochs
        restart = self.epoch // self.restart_epochs
        if restart > 0 and sub_epoch == 0:
            if self.restart_callback is not None:
                self.restart_callback(restart)
        return True

# Components {{{1

def _mr_make_norm(norm):
    def _mr_norm(y, width, depth, block, multi, activation, style, FC, d):
        skip = y
        merger = Sum()
        skip.feed(merger)
        z_start = skip
        z_start = z_start.feed(norm())
        z_start = z_start.feed(activation())
        for _ in range(multi):
            z = z_start
            for j in range(block):
                if j > 0:
                    z = z.feed(norm())
                    z = z.feed(activation())
                z = z.feed(FC())
            z.feed(merger)
        y = merger
        return y
    return _mr_norm

def _mr_batchless(y, width, depth, block, multi, activation, style, FC, d):
    skip = y
    merger = Sum()
    skip.feed(merger)
    z_start = skip.feed(activation())
    for _ in range(multi):
        z = z_start
        for j in range(block):
            if j > 0:
                z = z.feed(activation())
            z = z.feed(FC())
        z.feed(merger)
    y = merger
    return y

def _mr_onelesssum(y, width, depth, block, multi, activation, style, FC, d):
    # this is my own awful contraption.
    is_last = d + 1 == depth
    needs_sum = not is_last or multi > 1
    skip = y
    if needs_sum:
        merger = Sum()
    if not is_last:
        skip.feed(merger)
    z_start = skip.feed(activation())
    for _ in range(multi):
        z = z_start
        for j in range(block):
            if j > 0:
                z = z.feed(activation())
            z = z.feed(FC())
        if needs_sum:
            z.feed(merger)
    if needs_sum:
        y = merger
    else:
        y = z
    return y

_mr_styles = dict(
    lnorm=_mr_make_norm(LayerNorm),
    batchless=_mr_batchless,
    onelesssum=_mr_onelesssum,
)

def multiresnet(x, width, depth, block=2, multi=1,
                activation=Relu, style='batchless',
                init=init_he_normal):
    if style == 'cossim':
        style = 'batchless'
        DenseClass = CosineDense
    else:
        DenseClass = Dense
    if style not in _mr_styles:
        raise Exception('unknown resnet style', style)

    y = x
    last_size = x.output_shape[0]

    for d in range(depth):
        size = width
        FC = lambda: DenseClass(size, init)

        if last_size != size:
            y = y.feed(FC())

        y = _mr_styles[style](y, width, depth, block, multi, activation, style, FC, d)

        last_size = size

    return y

# Toy Data {{{1

inits = dict(he_normal=init_he_normal, he_uniform=init_he_uniform)
activations = dict(sigmoid=Sigmoid, tanh=Tanh, relu=Relu, elu=Elu, gelu=GeluApprox)

def prettyize(data):
    if isinstance(data, np.ndarray):
        s = ', '.join(('{:8.2e}'.format(n) for n in data))
        s = '[' + s + ']'
    else:
        s = '{:8.2e}'.format(data)
    return s

def normalize_data(data, mean=None, std=None):
    # in-place
    if mean is None or std is None:
        mean = np.mean(data, axis=0)
        std  =  np.std(data, axis=0)
        mean_str = prettyize(mean)
        std_str = prettyize(std)
        lament('nod(...,\n    {},\n    {})'.format(mean_str, std_str))
        sys.exit(1)
    data -= _f(mean)
    data /= _f(std)

def toy_data(train_samples, valid_samples, problem=2):
    total_samples = train_samples + valid_samples

    nod = normalize_data # shorthand to keep a sane indentation

    if problem == 0:
        from ml.cie_mlp_data import inputs, outputs, valid_inputs, valid_outputs
        inputs, outputs = _f(inputs), _f(outputs)
        valid_inputs, valid_outputs = _f(valid_inputs), _f(valid_outputs)

        nod(inputs, 127.5, 73.9)
        nod(outputs, 44.8, 21.7)
        nod(valid_inputs, 127.5, 73.9)
        nod(valid_outputs, 44.8, 21.7)

    elif problem == 1:
        from sklearn.datasets import make_friedman1
        inputs, outputs = make_friedman1(total_samples)
        inputs, outputs = _f(inputs), _f(outputs)
        outputs = np.expand_dims(outputs, -1)

        nod(inputs, 0.5, 1/np.sqrt(12))
        nod(outputs, 14.4, 4.9)

    elif problem == 2:
        from sklearn.datasets import make_friedman2
        inputs, outputs = make_friedman2(total_samples)
        inputs, outputs = _f(inputs), _f(outputs)
        outputs = np.expand_dims(outputs, -1)

        nod(inputs,
            [5.00e+01, 9.45e+02, 5.01e-01, 5.98e+00],
            [2.89e+01, 4.72e+02, 2.89e-01, 2.87e+00])

        nod(outputs, [482], [380])

    elif problem == 3:
        from sklearn.datasets import make_friedman3
        inputs, outputs = make_friedman3(total_samples)
        inputs, outputs = _f(inputs), _f(outputs)
        outputs = np.expand_dims(outputs, -1)

        nod(inputs,
            [4.98e+01, 9.45e+02, 4.99e-01, 6.02e+00],
            [2.88e+01, 4.73e+02, 2.90e-01, 2.87e+00])

        nod(outputs, [1.32327931], [0.31776295])

    else:
        raise Exception("unknown toy data set", problem)

    if problem != 0:
        # split off a validation set
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
        valid_inputs  =  inputs[indices][-valid_samples:]
        valid_outputs = outputs[indices][-valid_samples:]
        inputs  =  inputs[indices][:-valid_samples]
        outputs = outputs[indices][:-valid_samples]

    return (inputs, outputs), (valid_inputs, valid_outputs)

# Model Creation {{{1

def optim_from_config(config):
    if config.optim == 'adam':
        d1 = config.optim_decay1 if 'optim_decay1' in config else 9.5
        d2 = config.optim_decay2 if 'optim_decay2' in config else 999.5
        b1 = np.exp(-1/d1)
        b2 = np.exp(-1/d2)
        o = Nadam if config.nesterov else Adam
        optim = o(b1=b1, b2=b2)
    elif config.optim in ('rms', 'rmsprop'):
        d2 = config.optim_decay2 if 'optim_decay2' in config else 99.5
        mu = np.exp(-1/d2)
        optim = RMSprop(mu=mu)
    elif config.optim == 'sgd':
        d1 = config.optim_decay1 if 'optim_decay1' in config else 0
        if d1 > 0:
            b1 = np.exp(-1/d1)
            optim = Momentum(mu=b1, nesterov=config.nesterov)
        else:
            optim = Optimizer()
    else:
        raise Exception('unknown optimizer', config.optim)

    return optim

def learner_from_config(config, optim, rscb):
    if config.learner == 'sgdr':
        expando = config.expando if 'expando' in config else None
        learner = SGDR(optim, epochs=config.epochs, rate=config.learn,
                       restart_decay=config.restart_decay, restarts=config.restarts,
                       callback=rscb, expando=expando)
        # final learning rate isn't of interest here; it's gonna be close to 0.
        log('total epochs', learner.epochs)
    elif config.learner == 'anneal':
        learner = AnnealingLearner(optim, epochs=config.epochs, rate=config.learn,
                                   halve_every=config.learn_halve_every)
        log("final learning rate", "{:10.8f}".format(learner.final_rate))
    elif config.learner == 'dumb':
        learner = DumbLearner(optim, epochs=config.epochs, rate=config.learn,
                              halve_every=config.learn_halve_every,
                              restarts=config.restarts,
                              restart_advance=config.learn_restart_advance,
                              callback=rscb)
        log("final learning rate", "{:10.8f}".format(learner.final_rate))
    elif config.learner == 'sgd':
        learner = Learner(optim, epochs=config.epochs, rate=config.learn)
    else:
        raise Exception('unknown learner', config.learner)

    return learner

def lookup_loss(maybe_name):
    if isinstance(maybe_name, Loss):
        return maybe_name
    elif maybe_name == 'mse':
        return Squared()
    elif maybe_name == 'mshe': # mushy
        return SquaredHalved()
    elif maybe_name == 'mae':
        return Absolute()
    elif maybe_name == 'msee':
        return SomethingElse()
    raise Exception('unknown objective', maybe_name)

def ritual_from_config(config, learner, loss, mloss):
    if config.ritual == 'default':
        ritual = Ritual(learner=learner, loss=loss, mloss=mloss)
    elif config.ritual == 'stochm':
        ritual = StochMRitual(learner=learner, loss=loss, mloss=mloss)
    elif config.ritual == 'noisy':
        ritual = NoisyRitual(learner=learner, loss=loss, mloss=mloss,
                             input_noise=1e-1, output_noise=1e-2,
                             gradient_noise=2e-7)
    else:
        raise Exception('unknown ritual', config.ritual)

    return ritual

def model_from_config(config, input_features, output_features, callbacks):
    # Our Test Model

    init = inits[config.init]
    activation = activations[config.activation]

    x = Input(shape=(input_features,))
    y = x
    y = multiresnet(y,
                    config.res_width, config.res_depth,
                    config.res_block, config.res_multi,
                    activation=activation, init=init,
                    style=config.parallel_style)
    if y.output_shape[0] != output_features:
        y = y.feed(Dense(output_features, init))

    model = Model(x, y, unsafe=config.unsafe)

    if config.fn_load is not None:
        log('loading weights', config.fn_load)
        model.load_weights(config.fn_load)

    optim = optim_from_config(config)

    def rscb(restart):
        callbacks.restart()
        log("restarting", restart)
        if config.restart_optim:
            optim.reset()

    learner = learner_from_config(config, optim, rscb)

    loss = lookup_loss(config.loss)
    mloss = lookup_loss(config.mloss) if config.mloss else loss

    ritual = ritual_from_config(config, learner, loss, mloss)

    return model, learner, ritual

# main program {{{1

def run(program, args=None):
    args = args if args else []

    np.random.seed(42069)

    # Config {{{2

    from dotmap import DotMap
    config = DotMap(
        fn_load = None,
        fn_save = 'optim_nn.h5',
        log_fn = 'losses.npz',

        # multi-residual network parameters
        res_width = 28,
        res_depth = 2,
        res_block = 3, # normally 2 for plain resnet
        res_multi = 2, # normally 1 for plain resnet

        # style of resnet (order of layers, which layers, etc.)
        parallel_style = 'onelesssum',
        activation = 'gelu',

        optim = 'adam', # note: most features only implemented for Adam
        optim_decay1 = 2, #  first momentum given in epochs (optional)
        optim_decay2 = 100, # second momentum given in epochs (optional)
        nesterov = True,
        batch_size = 64,

        # learning parameters
        learner = 'sgdr',
        learn = 1e-2,
        epochs = 24,
        learn_halve_every = 16, # only used with anneal/dumb
        restarts = 8,
        restart_decay = 0.25, # only used with SGDR
        expando = lambda i: 24 * i,

        # misc
        init = 'he_normal',
        loss = 'msee',
        mloss = 'mse',
        ritual = 'default',
        restart_optim = False, # restarts also reset internal state of optimizer
        warmup = True, # train a couple epochs on gaussian noise and reset

        # logging/output
        log10_loss = True, # personally, i'm sick of looking linear loss values!
        #fancy_logs = True, # unimplemented (can't turn it off yet)

        problem = 2,
        compare = (
            # best results for ~10,000 parameters
            # training/validation pairs for each problem (starting from problem 0):
            #(5.08e-05, 6.78e-05),
            (7.577717e-04, 1.255284e-03),
            # 1080 epochs on these...
            (1.790511e-07, 2.785208e-07),
            (  10**-7.774,   10**-7.626),
            (5.266719e-07, 5.832677e-06), # overfitting? bad valid set?
        ),

        unsafe = True, # aka gotta go fast mode
    )

    for k in ['parallel_style', 'activation', 'optim', 'learner',
              'init', 'loss', 'mloss', 'ritual']:
        config[k] = config[k].lower()

    config.pprint()

    # Toy Data {{{2

    (inputs, outputs), (valid_inputs, valid_outputs) = \
      toy_data(2**14, 2**11, problem=config.problem)
    input_features  =  inputs.shape[-1]
    output_features = outputs.shape[-1]

    callbacks = Dummy()

    model, learner, ritual = \
      model_from_config(config, input_features, output_features, callbacks)

    # Model Information {{{2

    for node in model.ordered_nodes:
        children = [str(n) for n in node.children]
        if children:
            sep = '->'
            print(str(node) + sep + ('\n' + str(node) + sep).join(children))
    log('parameters', model.param_count)

    # Training {{{2

    batch_losses = []
    train_losses = []
    valid_losses = []

    def measure_error():
        def print_error(name, inputs, outputs, comparison=None):
            predicted = model.forward(inputs)
            err = ritual.measure(predicted, outputs)
            if config.log10_loss:
                print(name, "{:12.6e}".format(err))
                if comparison:
                    err10 = np.log10(err)
                    cmp10 = np.log10(comparison)
                    color = '\x1B[31m' if err10 > cmp10 else '\x1B[32m'
                    log(name + " log10-loss", "{:+6.3f} {}({:+6.3f})\x1B[0m".format(err10, color, err10 - cmp10))
                else:
                    log(name + " log10-loss", "{:+6.3f}".format(err, np.log10(err)))
            else:
                log(name + " loss", "{:12.6e}".format(err))
                if comparison:
                    fmt = "10**({:+7.4f}) times"
                    log("improvement", fmt.format(np.log10(comparison / err)))
            return err

        train_err = print_error("train",
                                inputs, outputs,
                                config.compare[config.problem][0])
        valid_err = print_error("valid",
                                valid_inputs, valid_outputs,
                                config.compare[config.problem][1])
        train_losses.append(train_err)
        valid_losses.append(valid_err)

    callbacks.restart = measure_error

    training = config.epochs > 0 and config.restarts >= 0

    ritual.prepare(model)

    if training and config.warmup:
        log("warming", "up")

        # use plain SGD in warmup to prevent (or possibly cause?) numeric issues
        temp_optim = learner.optim
        temp_loss = ritual.loss
        learner.optim = Optimizer(alpha=0.001)
        ritual.loss = Absolute() # less likely to blow up; more general

        # NOTE: experiment: trying const batches and batch_size
        bs = 256
        target = 1 * 1024 * 1024
        # 4 being sizeof(float)
        batches = (target / 4 / np.prod(inputs.shape[1:])) // bs * bs
        ins  = [int(batches)] + list( inputs.shape[1:])
        outs = [int(batches)] + list(outputs.shape[1:])

        for _ in range(4):
            ritual.train_batched(
                np.random.normal(size=ins),
                np.random.normal(size=outs),
                batch_size=bs)
            ritual.reset()

        learner.optim = temp_optim
        ritual.loss = temp_loss

    if training:
        measure_error()

    while training and learner.next():
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
        shuffled_inputs = inputs[indices]
        shuffled_outputs = outputs[indices]

        avg_loss, losses = ritual.train_batched(
            shuffled_inputs, shuffled_outputs,
            config.batch_size,
            return_losses=True)
        batch_losses += losses

        if config.log10_loss:
            fmt = "epoch {:4.0f}, rate {:10.8f}, log10-loss {:+6.3f}"
            log("info", fmt.format(learner.epoch + 1, learner.rate, np.log10(avg_loss)),
                update=True)
        else:
            fmt = "epoch {:4.0f}, rate {:10.8f}, loss {:12.6e}"
            log("info", fmt.format(learner.epoch + 1, learner.rate, avg_loss),
                update=True)

    measure_error()

    if training and config.fn_save is not None:
        log('saving weights', config.fn_save)
        model.save_weights(config.fn_save, overwrite=True)

    if training and config.log_fn is not None:
        log('saving losses', config.log_fn)
        np.savez_compressed(config.log_fn,
                            batch_losses=np.array(batch_losses, dtype=_f),
                            train_losses=np.array(train_losses, dtype=_f),
                            valid_losses=np.array(valid_losses, dtype=_f))

    # Evaluation {{{2
    # TODO: write this portion again

    return 0

# run main program {{{1

if __name__ == '__main__':
    sys.exit(run(sys.argv[0], sys.argv[1:]))
