#!/usr/bin/env python3

from onn import *
from onn_core import _f

#np.random.seed(42069)

use_emnist = False

measure_every_epoch = True

if use_emnist:
    lr = 0.0005
    epochs = 48
    starts = 2
    bs = 400

    learner_class = SGDR
    restart_decay = 0.5

    n_dense = 2
    n_denses = 0
    new_dims = (28, 28)
    activation = GeluApprox

    reg = L1L2(3.2e-5, 3.2e-4)
    final_reg = L1L2(3.2e-5, 1e-3)
    dropout = 0.05
    actreg_lamb = None

    load_fn = None
    save_fn = 'emnist.h5'
    log_fn = 'emnist_losses.npz'

    fn = 'emnist-balanced.npz'
    mnist_dim = 28
    mnist_classes = 47

else:
    lr = 0.0005
    epochs = 60
    starts = 3
    bs = 500

    learner_class = SGDR
    restart_decay = 0.5

    n_dense = 2
    n_denses = 0
    new_dims = (4, 12)
    activation = Relu

    reg = L1L2(3.2e-5, 3.2e-4)
    final_reg = L1L2(3.2e-5, 1e-3)
    dropout = 0.05
    actreg_lamb = None #1e-4

    load_fn = None
    save_fn = 'mnist.h5'
    log_fn = 'mnist_losses.npz'

    fn = 'mnist.npz'
    mnist_dim = 28
    mnist_classes = 10

def get_mnist(fn='mnist.npz'):
    import os.path
    if fn == 'mnist.npz' and not os.path.exists(fn):
        from keras.datasets import mnist
        from keras.utils.np_utils import to_categorical
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train.reshape(X_train.shape[0], 1, mnist_dim, mnist_dim)
        X_test = X_test.reshape(X_test.shape[0], 1, mnist_dim, mnist_dim)
        X_train = X_train.astype('float32') / 255
        X_test = X_test.astype('float32') / 255
        Y_train = to_categorical(y_train, mnist_classes)
        Y_test = to_categorical(y_test, mnist_classes)
        np.savez_compressed(fn,
                            X_train=X_train,
                            Y_train=Y_train,
                            X_test=X_test,
                            Y_test=Y_test)
        lament("mnist successfully saved to", fn)
        lament("please re-run this program to continue")
        sys.exit(1)

    with np.load(fn) as f:
        return f['X_train'], f['Y_train'], f['X_test'], f['Y_test']

inputs, outputs, valid_inputs, valid_outputs = get_mnist(fn)

def regulate(y):
    if actreg_lamb:
        assert activation == Relu, activation
        lamb = actreg_lamb # * np.prod(y.output_shape)
        reg = SaturateRelu(lamb)
        act = ActivityRegularizer(reg)
        reg.lamb_orig = reg.lamb # HACK
        y = y.feed(act)
    if dropout:
        y = y.feed(Dropout(dropout))
    return y

x = Input(shape=inputs.shape[1:])
y = x

y = y.feed(Reshape(new_shape=(mnist_dim, mnist_dim)))
for i in range(n_denses):
    if i > 0:
        y = regulate(y)
        y = y.feed(activation())
    y = y.feed(Denses(new_dims[0], axis=0, init=init_he_normal,
                      reg_w=reg, reg_b=reg))
    y = y.feed(Denses(new_dims[1], axis=1, init=init_he_normal,
                      reg_w=reg, reg_b=reg))
y = y.feed(Flatten())
for i in range(n_dense):
    if i > 0:
        y = regulate(y)
        y = y.feed(activation())
    y = y.feed(Dense(y.output_shape[0], init=init_he_normal,
                     reg_w=reg, reg_b=reg))
y = regulate(y)
y = y.feed(activation())

y = y.feed(Dense(mnist_classes, init=init_glorot_uniform,
                 reg_w=final_reg, reg_b=final_reg))
y = y.feed(Softmax())

model = Model(x, y, unsafe=True)

lr *= np.sqrt(bs)

optim = Adam()
if learner_class == SGDR:
    learner = learner_class(optim, epochs=epochs//starts, rate=lr,
                            restarts=starts-1, restart_decay=restart_decay,
                            expando=lambda i:0)
else:
    assert learner_class in (TriangularCLR, SineCLR, WaveCLR)
    learner = learner_class(optim, epochs=epochs, lower_rate=0, upper_rate=lr,
                            frequency=epochs//starts)

loss = CategoricalCrossentropy()
mloss = Accuracy()

ritual = Ritual(learner=learner, loss=loss, mloss=mloss)
#ritual = NoisyRitual(learner=learner, loss=loss, mloss=mloss,
#                     input_noise=1e-1, output_noise=3.2e-2, gradient_noise=1e-1)

model.print_graph()
log('parameters', model.param_count)

ritual.prepare(model)

batch_losses, batch_mlosses = [], []
train_losses, train_mlosses = [], []
valid_losses, valid_mlosses = [], []

train_confid, valid_confid = [], []

def measure_error(quiet=False):
    def print_error(name, inputs, outputs, comparison=None):
        loss, mloss, _, _ = ritual.test_batched(inputs, outputs, bs, return_losses='both')

        c = Confidence()
        predicted = ritual.model.forward(inputs, deterministic=True)
        confid = c.forward(predicted)

        if not quiet:
            log(name + " loss", "{:12.6e}".format(loss))
            log(name + " accuracy", "{:6.2f}%".format(mloss * 100))
            log(name + " confidence", "{:6.2f}%".format(confid * 100))

        return loss, mloss, confid

    loss, mloss, confid = print_error("train", inputs, outputs)
    train_losses.append(loss)
    train_mlosses.append(mloss)
    train_confid.append(confid)
    loss, mloss, confid = print_error("valid", valid_inputs, valid_outputs)
    valid_losses.append(loss)
    valid_mlosses.append(mloss)
    valid_confid.append(confid)

measure_error()

while learner.next():
    act_t = (learner.epoch - 1) / (learner.epochs - 1)
    if actreg_lamb:
        for node in model.ordered_nodes:
            if isinstance(node, ActivityRegularizer):
                node.reg.lamb = act_t * node.reg.lamb_orig # HACK

    avg_loss, avg_mloss, losses, mlosses = ritual.train_batched(
        inputs, outputs,
        batch_size=bs,
        return_losses='both')
    fmt = "rate {:10.8f}, loss {:12.6e}, accuracy {:6.2f}%"
    log("epoch {}".format(learner.epoch),
        fmt.format(learner.rate, avg_loss, avg_mloss * 100))

    batch_losses += losses
    batch_mlosses += mlosses

    if measure_every_epoch:
        quiet = learner.epoch != learner.epochs
        measure_error(quiet=quiet)

if not measure_every_epoch:
    measure_error()

if save_fn is not None:
    log('saving weights', save_fn)
    model.save_weights(save_fn, overwrite=True)

if log_fn:
    log('saving losses', log_fn)
    np.savez_compressed(log_fn,
                        batch_losses =np.array(batch_losses,  dtype=_f),
                        batch_mlosses=np.array(batch_mlosses, dtype=_f),
                        train_losses =np.array(train_losses,  dtype=_f),
                        train_mlosses=np.array(train_mlosses, dtype=_f),
                        valid_losses =np.array(valid_losses,  dtype=_f),
                        valid_mlosses=np.array(valid_mlosses, dtype=_f),
                        train_confid =np.array(train_confid,  dtype=_f),
                        valid_confid =np.array(valid_confid,  dtype=_f))
