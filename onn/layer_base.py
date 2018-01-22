import numpy as np

from collections import defaultdict, OrderedDict

from .weight import *


# used for numbering layers like Keras:
_layer_counters = defaultdict(lambda: 0)


class LayerIncompatibility(Exception):
    pass


class Layer:
    def __init__(self):
        self.parents = []
        self.children = []
        self.weights = OrderedDict()
        self.loss = None  # for activity regularizers
        self.input_shape = None
        self.output_shape = None
        kind = self.__class__.__name__
        global _layer_counters
        _layer_counters[kind] += 1
        self.name = "{}_{}".format(kind, _layer_counters[kind])
        self.unsafe = False  # disables assertions for better performance
        self.shared = False  # as in weight sharing

    def __str__(self):
        return self.name

    # methods we might want to override:

    def forward(self, X):
        raise NotImplementedError("unimplemented", self)

    def forward_deterministic(self, X):
        return self.forward(X)

    def backward(self, dY):
        raise NotImplementedError("unimplemented", self)

    def make_shape(self, parent):
        if self.input_shape is None:
            self.input_shape = parent.output_shape
        if self.output_shape is None:
            self.output_shape = self.input_shape

    def do_feed(self, child):
        self.children.append(child)

    def be_fed(self, parent):
        self.parents.append(parent)

    # TODO: better names for these (still)
    def _propagate(self, edges, deterministic):
        if not self.unsafe:
            assert len(edges) == 1, self
        if deterministic:
            return self.forward_deterministic(edges[0])
        else:
            return self.forward(edges[0])

    def _backpropagate(self, edges):
        if len(edges) == 1:
            return self.backward(edges[0])
        return sum((self.backward(dY) for dY in edges))

    # general utility methods:

    def is_compatible(self, parent):
        return np.all(self.input_shape == parent.output_shape)

    def feed(self, child):
        assert self.output_shape is not None, self
        child.make_shape(self)
        if not child.is_compatible(self):
            fmt = "{} is incompatible with {}: shape mismatch: {} vs. {}"
            raise LayerIncompatibility(fmt.format(
                self, child, self.output_shape, child.input_shape))
        self.do_feed(child)
        child.be_fed(self)
        return child

    def validate_input(self, X):
        assert X.shape[1:] == self.input_shape, \
            (str(self), X.shape[1:], self.input_shape)

    def validate_output(self, Y):
        assert Y.shape[1:] == self.output_shape, \
            (str(self), Y.shape[1:], self.output_shape)

    def _new_weights(self, name, **kwargs):
        w = Weights(**kwargs)
        assert name not in self.weights, name
        self.weights[name] = w
        return w

    def share(self, node):
        self.weights = node.weights  # TODO: this should be all it takes.
        for k, v in self.weights.items():
            # hack: key isn't necessarily attribute name!
            vs = getattr(node, k)
            setattr(self, k, vs)
        self.shared = True

    def clear_grad(self):
        for name, w in self.weights.items():
            w.g[:] = 0

    @property
    def size(self):
        return sum((w.size for w in self.weights.values()))

    def init(self, allocator):
        ins, outs = self.input_shape[0], self.output_shape[0]
        for k, w in self.weights.items():
            w.allocate(ins, outs, allocator=allocator)

    def propagate(self, values, deterministic):
        if not self.unsafe:
            assert self.parents, self
        edges = []
        for parent in self.parents:
            if parent in values:
                X = values[parent]
                if not self.unsafe:
                    self.validate_input(X)
                edges.append(X)
        Y = self._propagate(edges, deterministic)
        if not self.unsafe:
            self.validate_output(Y)
        return Y

    def backpropagate(self, values):
        if not self.unsafe:
            assert self.children, self
        edges = []
        for child in self.children:
            if child in values:
                dY = values[child]
                if not self.unsafe:
                    self.validate_output(dY)
                edges.append(dY)
        dX = self._backpropagate(edges)
        if not self.unsafe:
            self.validate_input(dX)
        return dX
