from .float import _f, _1, _pi, _inv2
from .optimizer_base import *


class Learner:
    per_batch = False

    def __init__(self, optim, epochs=100, rate=None):
        assert isinstance(optim, Optimizer)
        self.optim = optim
        self.start_rate = rate  # None is okay; it'll use optim.lr instead.
        self.epochs = int(epochs)
        self.reset()

    def reset(self, optim=False):
        self.started = False
        self.epoch = 0
        if optim:
            self.optim.reset()

    @property
    def epoch(self):
        return self._epoch

    @epoch.setter
    def epoch(self, new_epoch):
        self._epoch = int(new_epoch)
        if 0 <= self.epoch <= self.epochs:
            self.rate = self.rate_at(self._epoch)

    @property
    def rate(self):
        return self.optim.lr

    @rate.setter
    def rate(self, new_rate):
        self.optim.lr = new_rate

    def rate_at(self, epoch):
        if self.start_rate is None:
            return self.optim.lr
        return self.start_rate

    def next(self):
        # prepares the next epoch. returns whether or not to continue training.
        if not self.started:
            self.started = True
        self.epoch += 1
        if self.epoch > self.epochs:
            return False
        return True

    def batch(self, progress):  # TODO: rename.
        # interpolates rates between epochs.
        # unlike epochs, we do not store batch number as a state.
        # i.e. calling next() will not respect progress.
        assert 0 <= progress <= 1
        self.rate = self.rate_at(self._epoch + progress)

    def each_epoch(self):  # TODO: rename?
        while self.next():
            yield self.epoch

    @property
    def final_rate(self):
        return self.rate_at(self.epochs - 1e-8)


class AnnealingLearner(Learner):
    def __init__(self, optim, epochs=100, rate=None, halve_every=10):
        self.halve_every = _f(halve_every)
        self.anneal = _f(0.5**(1/self.halve_every))
        super().__init__(optim, epochs, rate)

    def rate_at(self, epoch):
        return super().rate_at(epoch) * self.anneal**epoch


def cosmod(x):
    # plot: https://www.desmos.com/calculator/hlgqmyswy2
    return (_1 + np.cos((x % _1) * _pi)) * _inv2


class SGDR(Learner):
    # Stochastic Gradient Descent with Restarts
    # paper: https://arxiv.org/abs/1608.03983
    # NOTE: this is missing a couple of the proposed features.

    per_batch = True

    def __init__(self, optim, epochs=100, rate=None,
                 restarts=0, restart_decay=0.5, callback=None,
                 expando=0):
        self.restart_epochs = int(epochs)
        self.decay = _f(restart_decay)
        self.restarts = int(restarts)
        self.restart_callback = callback

        # TODO: rename expando to something not insane
        self.expando = expando if expando is not None else lambda i: i
        if type(self.expando) == int:
            inc = self.expando
            self.expando = lambda i: i * inc

        self.splits = []
        epochs = 0
        for i in range(0, self.restarts + 1):
            split = epochs + self.restart_epochs + int(self.expando(i))
            self.splits.append(split)
            epochs = split
        super().__init__(optim, epochs, rate)

    def split_num(self, epoch):
        previous = [0] + self.splits
        for i, split in enumerate(self.splits):
            if epoch - 1 < split:
                sub_epoch = epoch - previous[i]
                next_restart = split - previous[i]
                return i, sub_epoch, next_restart
        raise Exception('this should never happen.')

    def rate_at(self, epoch):
        sr = self.start_rate
        base_rate = sr if sr is not None else self.optim.lr
        restart, sub_epoch, next_restart = self.split_num(max(1, epoch))
        x = _f(sub_epoch - 1) / _f(next_restart)
        return base_rate * self.decay**_f(restart) * cosmod(x)

    def next(self):
        if not super().next():
            return False
        restart, sub_epoch, next_restart = self.split_num(self.epoch)
        if restart > 0 and sub_epoch == 1:
            if self.restart_callback is not None:
                self.restart_callback(restart)
        return True


class TriangularCLR(Learner):
    per_batch = True

    def __init__(self, optim, epochs=400, upper_rate=None, lower_rate=0,
                 period=100, callback=None):
        # NOTE: start_rate is treated as upper_rate
        self.period = int(period)
        assert self.period > 0
        self.callback = callback
        self.lower_rate = _f(lower_rate)
        super().__init__(optim, epochs, upper_rate)

    def _t(self, epoch):
        # NOTE: this could probably be simplified
        offset = self.period / 2
        return np.abs(((epoch - 1 + offset) % self.period) - offset) \
            / offset

    def rate_at(self, epoch):
        sr = self.start_rate
        lr = self.lower_rate
        upper_rate = sr if sr is not None else self.optim.lr
        return self._t(epoch) * (upper_rate - lr) + lr

    def next(self):
        if not super().next():
            return False
        e = self.epoch - 1
        if e > 0 and e % self.period == 0:
            if self.callback is not None:
                self.callback(self.epoch // self.period)
        return True


class SineCLR(TriangularCLR):
    def _t(self, epoch):
        return np.sin(_pi * _inv2 * super()._t(epoch))


class WaveCLR(TriangularCLR):
    def _t(self, epoch):
        return _inv2 * (_1 - np.cos(_pi * super()._t(epoch)))


# more

class PolyLearner(Learner):
    per_batch = True

    def __init__(self, optim, epochs=400, coeffs=(1,)):
        self.coeffs = tuple(coeffs)
        super().__init__(optim, epochs, np.polyval(self.coeffs, 0))

    def rate_at(self, epoch):
        progress = (epoch - 1) / (self.epochs)
        ret = np.polyval(self.coeffs, progress)
        return np.abs(ret)
