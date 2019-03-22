import sys
import numpy as np


def lament(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def lower_priority():
    """Set the priority of the process to below-normal."""
    # via https://stackoverflow.com/a/1023269
    if sys.platform == 'win32':
        try:
            import win32api
            import win32process
            import win32con
            pid = win32api.GetCurrentProcessId()
            handle = win32api.OpenProcess(
                win32con.PROCESS_ALL_ACCESS, True, pid)
            win32process.SetPriorityClass(
                handle, win32process.BELOW_NORMAL_PRIORITY_CLASS)
        except ImportError:
            lament("you do not have pywin32 installed.")
            lament("the process priority could not be lowered.")
            lament("consider: python -m pip install pywin32")
            lament("consider: conda install pywin32")
    else:
        import os
        os.nice(1)


def div0(a, b):
    """division, whereby division by zero equals zero"""
    # http://stackoverflow.com/a/35696047
    a = np.asanyarray(a)
    b = np.asanyarray(b)
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        c[~np.isfinite(c)] = 0  # -inf inf NaN
    return c


def onehot(y):
    unique = np.unique(y)
    Y = np.zeros((y.shape[0], len(unique)), dtype=np.int8)
    offsets = np.arange(len(y)) * len(unique)
    Y.flat[offsets + y.flat] = 1
    return Y


def batchize(inputs, outputs, batch_size, shuffle=True):
    batch_count = np.ceil(len(inputs) / batch_size).astype(int)

    if shuffle:
        def gen():
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)

            for b in range(batch_count):
                bi = b * batch_size
                batch_indices = indices[bi:bi + batch_size]
                batch_inputs = inputs[batch_indices]
                batch_outputs = outputs[batch_indices]
                yield batch_inputs, batch_outputs

    else:
        def gen():
            for b in range(batch_count):
                bi = b * batch_size
                batch_inputs = inputs[bi:bi + batch_size]
                batch_outputs = outputs[bi:bi + batch_size]
                yield batch_inputs, batch_outputs

    return gen(), batch_count


def mixup(inputs, outputs, batch_size, p=0.1):
    # paper: https://arxiv.org/abs/1710.09412

    if p == 0:
        return batchize(inputs, outputs, batch_size)

    batch_count = np.ceil(len(inputs) / batch_size).astype(int)

    def lerp(a, b, t):
        t = t.reshape([len(t)] + [1] * (a.ndim - t.ndim))
        return (1 - t) * a + t * b

    def gen():
        indices0 = np.arange(len(inputs))
        indices1 = np.arange(len(inputs))
        np.random.shuffle(indices0)
        np.random.shuffle(indices1)

        for b in range(batch_count):
            bi = b * batch_size
            ind0 = indices0[bi:bi + batch_size]
            ind1 = indices1[bi:bi + batch_size]
            ps = np.random.beta(p, p, size=batch_size)
            batch_inputs = lerp(inputs[ind0], inputs[ind1], ps)
            batch_outputs = lerp(outputs[ind0], outputs[ind1], ps)
            yield batch_inputs, batch_outputs

    return gen(), batch_count


# more

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


class Folding:
    # NOTE: this class assumes classes are *exactly* evenly distributed.

    def __init__(self, inputs, outputs, folds):
        # outputs should be one-hot.

        self.folds = int(folds)

        # this temporarily converts one-hot encoding back to integer indices.
        classes = np.argmax(outputs, axis=-1)

        # we need to do stratified k-folds,
        # so let's put them in an order that's easy to split
        # without breaking class distribution.
        # don't worry, they'll get shuffled again in train_batched.
        classes = np.argmax(outputs, axis=-1)
        class_n = np.max(classes) + 1
        sorted_inputs = np.array([inputs[classes == n]
                                  for n in range(class_n)], inputs.dtype)
        sorted_outputs = np.arange(class_n) \
            .repeat(sorted_inputs.shape[1]).reshape(sorted_inputs.shape[:2])

        # now to interleave the classes instead of having them grouped:
        inputs = np.swapaxes(sorted_inputs, 0, 1) \
            .reshape(-1, *sorted_inputs.shape[2:])
        outputs = np.swapaxes(sorted_outputs, 0, 1) \
            .reshape(-1, *sorted_outputs.shape[2:])

        # one final thing: we need to make our outputs one-hot again.
        self.inputs = inputs
        self.outputs = onehot(outputs)

        # now we can do stratified folds simply by contiguous slices!
        self.foldstep = len(self.inputs) // self.folds
        assert len(self.inputs) % self.foldstep == 0, \
            "bad number of folds; cannot be stratified"

    def fold(self, i):
        roll = i * self.foldstep
        split = (self.folds - 1) * self.foldstep

        train_inputs = np.roll(self.inputs, roll, axis=0)[:split]
        valid_inputs = np.roll(self.inputs, roll, axis=0)[split:]

        train_outputs = np.roll(self.outputs, roll, axis=0)[:split]
        valid_outputs = np.roll(self.outputs, roll, axis=0)[split:]

        return train_inputs, train_outputs, valid_inputs, valid_outputs
