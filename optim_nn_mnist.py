#!/usr/bin/env python3

from optim_nn import *
from optim_nn_core import _f

#np.random.seed(42069)

#           train loss:   7.048363e-03
#       train accuracy:    99.96%
#           valid loss:   3.062232e-01
#       valid accuracy:    97.22%
# TODO: add dropout or something to lessen overfitting

lr = 0.0032
epochs = 125
starts = 5
restart_decay = 0.5
bs = 100

log_fn = 'mnist_losses.npz'
measure_every_epoch = True

mnist_dim = 28
mnist_classes = 10
def get_mnist(fn='mnist.npz'):
    import os.path
    if not os.path.exists(fn):
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

inputs, outputs, valid_inputs, valid_outputs = get_mnist()

x = Input(shape=inputs.shape[1:])
y = x

y = y.feed(Reshape(new_shape=(mnist_dim, mnist_dim,)))
y = y.feed(Denses(4, axis=0, init=init_he_normal))
y = y.feed(Denses(12, axis=1, init=init_he_normal))
y = y.feed(Flatten())
y = y.feed(Dense(y.output_shape[0], init=init_he_normal))
y = y.feed(Relu())

y = y.feed(Dense(mnist_classes, init=init_glorot_uniform))
y = y.feed(Softmax())

model = Model(x, y, unsafe=True)

optim = Adam()
if 0:
    learner = SGDR(optim, epochs=epochs//starts, rate=lr,
                   restarts=starts-1, restart_decay=restart_decay,
                   expando=lambda i:0)
else:
#   learner = TriangularCLR(optim, epochs=epochs, lower_rate=0, upper_rate=lr,
#                           frequency=epochs//starts)
    learner = SineCLR(optim, epochs=epochs, lower_rate=0, upper_rate=lr,
                      frequency=epochs//starts)

loss = CategoricalCrossentropy()
mloss = Accuracy()

ritual = Ritual(learner=learner, loss=loss, mloss=mloss)

log('parameters', model.param_count)

ritual.prepare(model)

batch_losses, batch_mlosses = [], []
train_losses, train_mlosses = [], []
valid_losses, valid_mlosses = [], []

def measure_error(quiet=False):
    def print_error(name, inputs, outputs, comparison=None):
        loss, mloss, _, _ = ritual.test_batched(inputs, outputs, bs, return_losses='both')
        if not quiet:
            log(name + " loss", "{:12.6e}".format(loss))
            log(name + " accuracy", "{:6.2f}%".format(mloss * 100))
        return loss, mloss

    if not quiet:
        loss, mloss = print_error("train", inputs, outputs)
        train_losses.append(loss)
        train_mlosses.append(mloss)
    loss, mloss = print_error("valid", valid_inputs, valid_outputs)
    valid_losses.append(loss)
    valid_mlosses.append(mloss)

measure_error()

while learner.next():
    indices = np.arange(inputs.shape[0])
    np.random.shuffle(indices)
    shuffled_inputs = inputs[indices]
    shuffled_outputs = outputs[indices]

    avg_loss, avg_mloss, losses, mlosses = ritual.train_batched(
        shuffled_inputs, shuffled_outputs,
        batch_size=bs,
        return_losses='both')
    fmt = "rate {:10.8f}, loss {:12.6e}, accuracy {:6.2f}%"
    log("epoch {}".format(learner.epoch + 1),
        fmt.format(learner.rate, avg_loss, avg_mloss * 100))

    batch_losses += losses
    batch_mlosses += mlosses

    if measure_every_epoch:
        quiet = learner.epoch + 1 != learner.epochs
        measure_error(quiet=quiet)

if not measure_every_epoch:
    measure_error()

if log_fn:
    log('saving losses', log_fn)
    np.savez_compressed(log_fn,
                        batch_losses =np.array(batch_losses,  dtype=_f),
                        batch_mlosses=np.array(batch_mlosses, dtype=_f),
                        train_losses =np.array(train_losses,  dtype=_f),
                        train_mlosses=np.array(train_mlosses, dtype=_f),
                        valid_losses =np.array(valid_losses,  dtype=_f),
                        valid_mlosses=np.array(valid_mlosses, dtype=_f))
