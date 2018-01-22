import numpy as np


class Weights:
    # we may or may not contain weights -- or any information, for that matter.

    def __init__(self, **kwargs):
        self.f = None  # forward weights
        self.g = None  # backward weights (gradients)
        self.shape = None
        self.init = None
        self.allocator = None
        self.regularizer = None
        self._allocated = False

        self.configure(**kwargs)

    def configure(self, **kwargs):
        for k, v in kwargs.items():
            getattr(self, k)  # ensures the key already exists
            setattr(self, k, v)

    @property
    def size(self):
        assert self.shape is not None
        return np.prod(self.shape)

    def allocate(self, *args, **kwargs):
        if self._allocated:
            raise Exception("attempted to allocate existing weights")
        self.configure(**kwargs)

        # intentionally not using isinstance
        assert type(self.shape) == tuple, self.shape

        f, g = self.allocator(self.size)
        assert len(f) == self.size, "{} != {}".format(f.shape, self.size)
        assert len(g) == self.size, "{} != {}".format(g.shape, self.size)
        f[:] = self.init(self.size, *args)
        g[:] = self.init(self.size, *args)
        self.f = f.reshape(self.shape)
        self.g = g.reshape(self.shape)

        self._allocated = True

    def forward(self):
        if self.regularizer is None:
            return 0.0
        return self.regularizer.forward(self.f)

    def backward(self):
        if self.regularizer is None:
            return 0.0
        return self.regularizer.backward(self.f)

    def update(self):
        if self.regularizer is None:
            return
        self.g += self.regularizer.backward(self.f)
