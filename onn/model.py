import sys

from .float import _f, _0
from .nodal import *
from .layer_base import *
from .utility import *


class Model:
    def __init__(self, nodes_in, nodes_out,
                 loss=None, mloss=None, unsafe=False):
        self.loss = loss if loss is not None else SquaredHalved()
        self.mloss = mloss if mloss is not None else loss

        nodes_in = [nodes_in] if isinstance(nodes_in, Layer) else nodes_in
        nodes_out = [nodes_out] if isinstance(nodes_out, Layer) else nodes_out
        assert type(nodes_in) == list, type(nodes_in)
        assert type(nodes_out) == list, type(nodes_out)
        self.nodes_in = nodes_in
        self.nodes_out = nodes_out

        self.nodes = traverse_all(self.nodes_in, self.nodes_out)
        self.make_weights()
        for node in self.nodes:
            node.unsafe = unsafe
        # TODO: handle the same layer being in more than one node.

    @property
    def ordered_nodes(self):
        # deprecated? we don't guarantee an order like we did before.
        return self.nodes

    def make_weights(self):
        self.param_count = sum((node.size for node in self.nodes
                                if not node.shared))
        self.W = np.zeros(self.param_count, dtype=_f)
        self.dW = np.zeros(self.param_count, dtype=_f)

        offset = 0
        for node in self.nodes:
            if node.size > 0 and not node.shared:
                inner_offset = 0

                def allocate(size):
                    nonlocal inner_offset
                    o = offset + inner_offset
                    ret = self.W[o:o+size], self.dW[o:o+size]
                    inner_offset += size
                    assert len(ret[0]) == len(ret[1])
                    assert size == len(ret[0]), (size, len(ret[0]))
                    return ret

                fmt = "Layer {} allocated {} weights than it said it would"
                node.init(allocate)
                assert inner_offset <= node.size, fmt.format("more", node)
                # i don't care if "less" is grammatically incorrect.
                # you're mom is grammatically incorrect.
                assert inner_offset >= node.size, fmt.format("less", node)
                offset += node.size

    def evaluate(self, input_, deterministic=True):
        fmt = "ambiguous input in multi-{} network; use {}() instead"
        assert len(self.nodes_in) == 1, fmt.format("input", "evaluate_multi")
        assert len(self.nodes_out) == 1, fmt.format("output", "evaluate_multi")
        node_in = self.nodes_in[0]
        node_out = self.nodes_out[0]
        outputs = self.evaluate_multi({node_in: input_}, deterministic)
        return outputs[node_out]

    def apply(self, error):  # TODO: better name?
        fmt = "ambiguous input in multi-{} network; use {}() instead"
        assert len(self.nodes_in) == 1, fmt.format("input", "apply_multi")
        assert len(self.nodes_out) == 1, fmt.format("output", "apply_multi")
        node_in = self.nodes_in[0]
        node_out = self.nodes_out[0]
        inputs = self.apply_multi({node_out: error})
        return inputs[node_in]

    def evaluate_multi(self, inputs, deterministic=True):
        fmt = "missing {} for node {}"
        values = dict()
        outputs = dict()
        for node in self.nodes:
            if node in self.nodes_in:
                assert node in inputs, fmt.format("input", node.name)
                X = inputs[node]
                values[node] = node._propagate(np.expand_dims(X, 0),
                                               deterministic)
            else:
                values[node] = node.propagate(values, deterministic)
            if node in self.nodes_out:
                outputs[node] = values[node]
        return outputs

    def apply_multi(self, outputs):
        fmt = "missing {} for node {}"
        values = dict()
        inputs = dict()
        for node in reversed(self.nodes):
            if node in self.nodes_out:
                assert node in outputs, fmt.format("output", node.name)
                X = outputs[node]
                values[node] = node._backpropagate(np.expand_dims(X, 0))
            else:
                values[node] = node.backpropagate(values)
            if node in self.nodes_in:
                inputs[node] = values[node]
        return inputs

    def forward(self, inputs, outputs, measure=False, deterministic=False):
        predicted = self.evaluate(inputs, deterministic=deterministic)
        if measure:
            error = self.mloss.forward(predicted, outputs)
        else:
            error = self.loss.forward(predicted, outputs)
        return error, predicted

    def backward(self, predicted, outputs, measure=False):
        if measure:
            error = self.mloss.backward(predicted, outputs)
        else:
            error = self.loss.backward(predicted, outputs)
        # input_delta is rarely useful; it's just to match the forward pass.
        input_delta = self.apply(error)
        return self.dW, input_delta

    def clear_grad(self):
        for node in self.nodes:
            node.clear_grad()

    def regulate_forward(self):
        loss = _0
        for node in self.nodes:
            if node.loss is not None:
                loss += node.loss
            for k, w in node.weights.items():
                loss += w.forward()
        return loss

    def regulate(self):
        for node in self.nodes:
            for k, w in node.weights.items():
                w.update()

    def load_weights(self, fn):
        # seemingly compatible with keras' Dense layers.
        weights = {}

        import h5py
        open(fn)  # just ensure the file exists (python's error is better)

        f = h5py.File(fn, 'r')

        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                weights[name.split('/')[-1]] = np.array(obj[:], dtype=_f)

        f.visititems(visitor)
        f.close()

        used = {}
        for k in weights.keys():
            used[k] = False

        nodes = [node for node in self.nodes if node.size > 0]
        # TODO: support shared weights.
        for node in nodes:
            full_name = str(node).lower()
            for s_name, o_name in node.serialized.items():
                key = full_name + '_' + s_name
                data = weights[key]
                target = getattr(node, o_name)
                target.f[:] = data
                used[key] = True

        for k, v in used.items():
            if not v:
                lament("WARNING: unused weight", k)

    def save_weights(self, fn, overwrite=False):
        import h5py
        from collections import defaultdict

        f = h5py.File(fn, 'w')

        counts = defaultdict(lambda: 0)

        nodes = [node for node in self.nodes if node.size > 0]
        # TODO: support shared weights.
        for node in nodes:
            full_name = str(node).lower()
            grp = f.create_group(full_name)
            for s_name, o_name in node.serialized.items():
                key = full_name + '_' + s_name
                target = getattr(node, o_name)
                data = grp.create_dataset(key, target.shape, dtype=_f)
                data[:] = target.f
                counts[key] += 1
                if counts[key] > 1:
                    lament("WARNING: rewrote weight", key)

        f.close()

    def print_graph(self, file=sys.stdout):
        print('digraph G {', file=file)
        for node in self.nodes:
            children = [str(n) for n in node.children]
            if children:
                sep = '->'
                print('\t' + str(node) + sep +
                      (';\n\t' + str(node) + sep).join(children) + ';',
                      file=file)
        print('}', file=file)
