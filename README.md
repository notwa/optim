# neural network stuff

not unlike [my dsp repo,](https://github.com/notwa/dsp)
this is a bunch of half-baked python code that's kinda handy.
i give no guarantee anything provided here is correct.

don't expect commits, docs, or comments to be any verbose.

## other stuff

if you're coming here from Google: sorry, keep searching.
i know Google sometimes likes to give random repositories a high search ranking.
maybe consider one of the following:

* [keras](https://github.com/fchollet/keras)
  for easy tensor-optimized networks.
  strong [tensorflow](http://tensorflow.org) integration as of version 2.0.
  also check out the
  [keras-contrib](https://github.com/farizrahman4u/keras-contrib)
  library for more components based on recent papers.
* [theano's source code](https://github.com/Theano/theano/blob/master/theano/tensor/nnet/nnet.py)
  contains pure numpy test methods to reference against.
* [minpy](https://github.com/dmlc/minpy)
  for tensor-powered numpy routines and automatic differentiation.
* [autograd](https://github.com/HIPS/autograd)
  for automatic differentiation without tensors.

## dependencies

python 3.5+

numpy scipy h5py sklearn dotmap

## minimal example

```python
#!/usr/bin/env python3
from onn import *
bs = 500
lr = 0.01
reg = L1L2(3.2e-5, 3.2e-4)
final_reg = L1L2(3.2e-5, 1e-3)

def get_mnist(fn='mnist.npz'):
    with np.load(fn) as f:
        return f['X_train'], f['Y_train'], f['X_test'], f['Y_test']
inputs, outputs, valid_inputs, valid_outputs = get_mnist()

x = Input(shape=inputs.shape[1:])
y = x
y = y.feed(Flatten())
y = y.feed(Dense(y.output_shape[0], init=init_he_normal, reg_w=reg, reg_b=reg))
y = y.feed(Relu())
y = y.feed(Dense(y.output_shape[0], init=init_he_normal, reg_w=reg, reg_b=reg))
y = y.feed(Dropout(0.05))
y = y.feed(Relu())
y = y.feed(Dense(10, init=init_glorot_uniform, reg_w=final_reg, reg_b=final_reg))
y = y.feed(Softmax())
model = Model(x, y, loss=CategoricalCrossentropy(), mloss=Accuracy(), unsafe=True)

optim = Adam()
learner = SGDR(optim, epochs=20, rate=lr, restarts=1)
ritual = Ritual(learner=learner)
ritual.prepare(model)
while learner.next():
    print("epoch", learner.epoch)
    mloss, _ = ritual.train_batched(inputs, outputs, batch_size=bs, return_losses=True)
    print("train accuracy", "{:6.2f}%".format(mloss * 100))

def print_error(name, inputs, outputs):
    loss, mloss, _, _ = ritual.test_batched(inputs, outputs, bs, return_losses='both')
    print(name + " loss", "{:12.6e}".format(loss))
    print(name + " accuracy", "{:6.2f}%".format(mloss * 100))
print_error("train", inputs, outputs)
print_error("valid", valid_inputs, valid_outputs)
predicted = model.evaluate(inputs) # use this as you will!
```

## contributing

i'm just throwing this code out there,
so i don't actually expect anyone to contribute,
*but* if you do find a blatant issue,
maybe [yell at me on twitter.](https://twitter.com/antiformant)
