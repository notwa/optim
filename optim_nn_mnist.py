#!/usr/bin/env python3

from optim_nn import *

#np.random.seed(42069)

#           train loss:   4.194040e-02
#       train accuracy:    99.46%
#           valid loss:   1.998158e-01
#       valid accuracy:    97.26%
# TODO: add dropout or something to lessen overfitting

lr = 0.01
epochs = 24
starts = 2
bs = 100

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
learner = SGDR(optim, epochs=epochs//starts, rate=lr,
               restarts=starts - 1, restart_decay=0.5,
               expando=lambda i:0)

loss = CategoricalCrossentropy()
mloss = Accuracy()

ritual = Ritual(learner=learner, loss=loss, mloss=mloss)

log('parameters', model.param_count)

ritual.prepare(model)

def measure_error():
    def print_error(name, inputs, outputs, comparison=None):
        loss, mloss, _, _ = ritual.test_batched(inputs, outputs, bs, return_losses='both')
        log(name + " loss", "{:12.6e}".format(loss))
        log(name + " accuracy", "{:6.2f}%".format(mloss * 100))
        return loss, mloss

    print_error("train", inputs, outputs)
    print_error("valid", valid_inputs, valid_outputs)

measure_error()

while learner.next():
    indices = np.arange(inputs.shape[0])
    np.random.shuffle(indices)
    shuffled_inputs = inputs[indices]
    shuffled_outputs = outputs[indices]

    avg_loss, avg_mloss, _, _ = ritual.train_batched(
        shuffled_inputs, shuffled_outputs,
        batch_size=bs,
        return_losses='both')
    fmt = "rate {:10.8f}, loss {:12.6e}, accuracy {:6.2f}%"
    log("epoch {}".format(learner.epoch + 1),
        fmt.format(learner.rate, avg_loss, avg_mloss * 100))

measure_error()
