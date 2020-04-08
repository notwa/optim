# neural network stuff

not unlike [my DSP repo,](https://github.com/notwa/dsp)
`onn` is a bunch of half-baked python code that's kinda handy.
i give no guarantee anything provided here is correct.

don't expect commits, docs, or comments to be any verbose.
however, i do attempt to cite and source any techniques used.

## alternatives

when creating this, i wanted a library free of compilation
and heavy dependencies — other than numpy and scipy, but these are commonplace.
although `onn` is significantly faster than equivalent autograd code,
performance is not a concern and it cannot run on GPU.

since this is my personal repo, i recommend that others do not rely on it.
instead, consider one of the following:

* [keras](https://github.com/fchollet/keras)
  it's now integrated directly into [tensorflow.](https://tensorflow.org).
  it runs on CPU and GPU. however, it requires a compilation stage.
* also check out the
  [keras-contrib](https://github.com/farizrahman4u/keras-contrib)
  library for more keras components based on recent papers.
* the library itself may be discontinued, but
  [theano's source code](https://github.com/Theano/theano/blob/master/theano/tensor/nnet/nnet.py)
  contains pure numpy test methods as reference.
* [minpy](https://github.com/dmlc/minpy)
  for tensor-powered numpy routines and automatic differentiation.
  deprecated by [mxnet.](https://github.com/apache/incubator-mxnet)
  i've never used either so i don't know what mxnet is like.
* [autograd](https://github.com/HIPS/autograd)
  for automatic differentiation without tensors.
  this is my personal favorite, although it is a little slow.
* autograd has been discontinued in favor of
  [Google's JAX,](https://github.com/google/jax)
  however, JAX is quite heavy and non-portable in comparison.
  JAX runs on CPU and GPU and it can skip compilation on CPU.

## dependencies

python 3.5+

mandatory packages: `numpy` `scipy`

needed for saving weights: `h5py`

used in example code: `dotmap`

## minimal example

```python
#!/usr/bin/env python3
import numpy as np
import mnists  # https://github.com/notwa/mnists
from onn import *

train_x, train_y, valid_x, valid_y = mnists.prepare("mnist")
learning_rate = 0.01
epochs = 20
batch_size = 500
hidden_size = 196  # 1/4 the number of pixels in an mnist sample
reg = L1L2(1e-5, 1e-4)
final_reg = None

x = Input(shape=train_x.shape[1:])  # give the shape of a single example
y = x  # superficial code just to make changing layer order a little easier
y = y.feed(Flatten())
y = y.feed(Dense(hidden_size, init=init_he_normal, reg_w=reg, reg_b=reg))
y = y.feed(Dropout(0.5))
y = y.feed(GeluApprox())
y = y.feed(Dense(10, init=init_glorot_uniform, reg_w=final_reg, reg_b=final_reg))
y = y.feed(Softmax())
model = Model(x, y,  # follow the graph from node x to y
              loss=CategoricalCrossentropy(), mloss=Accuracy(),
              unsafe=True)  # skip some sanity checks to go faster

optim = Adam()  # good ol' adam
learner = WaveCLR(optim, upper_rate=learning_rate,
                  epochs=epochs, period=epochs)  # ramp up and down the rate
ritual = Ritual(learner=learner)  # the accursed deep-learning ritual

ritual.prepare(model)  # reset training
while learner.next():
    print("epoch", learner.epoch)
    losses = ritual.train(*batchize(train_x, train_y, batch_size))
    print("train accuracy", "{:6.2%}".format(losses.avg_mloss))

def print_error(name, train_x, train_y):
    losses = ritual.test_batched(train_x, train_y, batch_size)
    print(name + " loss", "{:12.6e}".format(losses.avg_loss))
    print(name + " accuracy", "{:6.2%}".format(losses.avg_mloss))
print_error("train", train_x, train_y)
print_error("valid", valid_x, valid_y)
predicted = model.evaluate(train_x)  # use this as you will!
```

[(mnists is available here)](https://github.com/notwa/mnists)

## contributing

i'm just throwing this code out there,
so i don't actually expect anyone to contribute,
*but* if you do find a blatant issue,
maybe [yell at me on twitter.](https://twitter.com/antiformant)
