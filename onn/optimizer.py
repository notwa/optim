import numpy as np

from .float import _f, _0, _1
from .optimizer_base import *
from .utility import *


def filter_gradients(accum, grads, param):
    # NOTE: this modifies accum in-place.
    # param > 0 acts as a simple one-pole low-pass filter, unity at DC.
    # param < 0 acts as an accumulator with a decay of -param, nonunity at DC.
    # param == 0 simply copies grads into accum.
    if param == 0:
        accum[:] = grads
    elif param == 1:
        pass
    elif param == -1:
        accum += grads
    elif param < 0:
        accum *= -param
        accum += grads
    else:
        accum += (1 - param) * (grads - accum)
    return accum


class Momentum(Optimizer):
    def __init__(self, lr=0.01, mu=0.9, nesterov=False):
        self.mu = _f(mu)
        self.nesterov = bool(nesterov)

        super().__init__(lr)

    def reset(self):
        self.accum = None

        super().reset()

    def compute(self, dW, W):
        if self.accum is None:
            self.accum = np.zeros_like(dW)

        self.accum[:] = self.accum * self.mu + dW
        if self.nesterov:
            return -self.lr * (self.accum * self.mu + dW)
        else:
            return -self.lr * self.accum


class Adadelta(Optimizer):
    # paper: https://arxiv.org/abs/1212.5701

    def __init__(self, lr=1.0, mu=0.95, eps=1e-8):
        self.mu = _f(mu)
        self.eps = _f(eps)

        super().__init__(lr)

    def reset(self):
        self.g = None
        self.x = None

        super().reset()

    def compute(self, dW, W):
        if self.g is None:
            self.g = np.zeros_like(dW)
        if self.x is None:
            self.x = np.zeros_like(dW)

        self.g += (self.mu - 1) * (self.g - np.square(dW))
        delta = dW * np.sqrt(self.x + self.eps) / (np.sqrt(self.g) + self.eps)
        self.x += (self.mu - 1) * (self.x - np.square(delta))
        return -self.lr * delta


class RMSpropCentered(Optimizer):
    # referenced TensorFlow/PyTorch.
    # paper: https://arxiv.org/abs/1308.0850

    def __init__(self, lr=1e-4, aleph=0.95, momentum=0.9, eps=1e-8):
        self.aleph = _f(aleph)
        self.momentum = _f(momentum)
        self.eps = _f(eps)

        super().__init__(lr)

    def reset(self):
        self.g = None
        self.mt = None
        self.vt = None
        self.delta = None

        super().reset()

    def compute(self, dW, W):
        if self.g is None:
            self.g = np.zeros_like(dW)
        if self.mt is None:
            self.mt = np.zeros_like(dW)
        if self.vt is None:
            self.vt = np.zeros_like(dW)
        if self.delta is None:
            self.delta = np.zeros_like(dW)

        self.mt += (1 - self.aleph) * (dW - self.mt)
        self.vt += (1 - self.aleph) * (np.square(dW) - self.vt)

        # PyTorch has the epsilon outside of the sqrt,
        # TensorFlow and the paper have it within.
        # in onn, we generally do it outside, as this seems to work better.
        temp = dW / (np.sqrt(self.vt - np.square(self.mt)) + self.eps)

        # TensorFlow does it this way.
        self.delta[:] = self.momentum * self.delta + self.lr * temp
        return -self.delta
        # PyTorch does it this way.
        # self.delta[:] = self.momentum * self.delta + temp
        # return -self.lr * self.delta
        # they are equivalent only when LR is constant, which it might not be.


class Nadam(Optimizer):
    # paper: https://arxiv.org/abs/1412.6980
    # paper: http://cs229.stanford.edu/proj2015/054_report.pdf
    # TODO: double-check this implementation. also read the damn paper.
    # lifted from:
    # https://github.com/fchollet/keras/blob/5d38b04/keras/optimizers.py#L530
    # https://github.com/jpilaul/IFT6266_project/blob/master/Models/Algo_Momentum.py

    def __init__(self, lr=0.002, b1=0.9, b2=0.999, eps=1e-8):
        self.b1 = _f(b1)  # decay term
        self.b2 = _f(b2)  # decay term
        self.eps = _f(eps)

        super().__init__(lr)

    def reset(self):
        self.mt = None
        self.vt = None
        self.t = 0
        self.sched = 1

        super().reset()

    def compute(self, dW, W):
        self.t += 1

        if self.mt is None:
            self.mt = np.zeros_like(dW)
        if self.vt is None:
            self.vt = np.zeros_like(dW)

        ut0 = self.b1 * (1 - 0.5 * 0.96**(self.t + 0))
        ut1 = self.b1 * (1 - 0.5 * 0.96**(self.t + 1))

        sched0 = self.sched * ut0
        sched1 = self.sched * ut0 * ut1
        self.sched = sched0

        gp = dW / (1 - sched0)

        self.mt += (1 - self.b1) * (dW - self.mt)
        self.vt += (1 - self.b2) * (np.square(dW) - self.vt)

        mtp = self.mt / (1 - sched1)
        vtp = self.vt / (1 - self.b2**self.t)

        mt_bar = (1 - ut0) * gp + ut1 * mtp

        return -self.lr * mt_bar / (np.sqrt(vtp) + self.eps)


class FTML(Optimizer):
    # paper: http://www.cse.ust.hk/~szhengac/papers/icml17.pdf
    # author's implementation: https://github.com/szhengac/optim/commit/923555e

    def __init__(self, lr=0.0025, b1=0.6, b2=0.999, eps=1e-8):
        self.iterations = _0
        self.b1 = _f(b1)  # decay term
        self.b2 = _f(b2)  # decay term
        self.eps = _f(eps)

        super().__init__(lr)

    def reset(self):
        self.dt1 = None
        self.dt = None
        self.vt = None
        self.zt = None
        self.b1_t = _1
        self.b2_t = _1

        super().reset()

    def compute(self, dW, W):
        if self.dt1 is None:
            self.dt1 = np.zeros_like(dW)
        if self.dt is None:
            self.dt = np.zeros_like(dW)
        if self.vt is None:
            self.vt = np.zeros_like(dW)
        if self.zt is None:
            self.zt = np.zeros_like(dW)

        # NOTE: we could probably rewrite these equations to avoid this copy.
        self.dt1[:] = self.dt[:]

        self.b1_t *= self.b1
        self.b2_t *= self.b2

        # hardly an elegant solution.
        lr = max(self.lr, self.eps)

        # same as Adam's vt.
        self.vt[:] = self.b2 * self.vt + (1 - self.b2) * dW * dW

        # you can factor "inner" out of Adam as well.
        inner = np.sqrt(self.vt / (1 - self.b2_t)) + self.eps
        self.dt[:] = (1 - self.b1_t) / lr * inner

        sigma_t = self.dt - self.b1 * self.dt1

        # Adam's mt minus the sigma term.
        self.zt[:] = self.b1 * self.zt + (1 - self.b1) * dW - sigma_t * W

        # subtract by weights to avoid having to override self.update.
        return -self.zt / self.dt - W


class MomentumClip(Optimizer):
    def __init__(self, lr=0.01, mu=0.9, nesterov=False, clip=1.0, debug=False):
        self.mu = _f(mu)
        self.clip = _f(clip)
        self.nesterov = bool(nesterov)
        self.debug = bool(debug)

        super().__init__(lr)

    def reset(self):
        self.accum = None

        super().reset()

    def compute(self, dW, W):
        if self.accum is None:
            self.accum = np.zeros_like(dW)

        total_norm = np.linalg.norm(dW)
        clip_scale = self.clip / (total_norm + 1e-6)
        if clip_scale < 1:
            if self.debug:
                lament("clipping gradients; norm: {:10.5f}".format(total_norm))
            dW *= clip_scale

        self.accum[:] = self.accum * self.mu + dW
        if self.nesterov:
            return -self.lr * (self.accum * self.mu + dW)
        else:
            return -self.lr * self.accum


class AddSign(Optimizer):
    # paper: https://arxiv.org/abs/1709.07417

    def __init__(self, lr=0.01, mu=0.9, alpha=1):
        self.mu = _f(mu)
        self.alpha = _f(alpha)

        super().__init__(lr)

    def reset(self):
        self.accum = None

        super().reset()

    def compute(self, dW, W):
        if self.accum is None:
            self.accum = np.zeros_like(dW)

        self.accum[:] = self.accum * self.mu + dW

        signed = np.sign(dW) * np.sign(self.accum)
        # signed *= decay

        return -self.lr * dW * (self.alpha + signed)


class PowerSign(Optimizer):
    # paper: https://arxiv.org/abs/1709.07417

    def __init__(self, lr=0.01, mu=0.9, alpha=np.e):
        self.mu = _f(mu)
        self.alpha = _f(alpha)
        self.use_exp = np.isclose(self.alpha, _f(np.e))

        super().__init__(lr)

    def reset(self):
        self.accum = None

        super().reset()

    def compute(self, dW, W):
        if self.accum is None:
            self.accum = np.zeros_like(dW)

        self.accum[:] = self.accum * self.mu + dW

        signed = np.sign(dW) * np.sign(self.accum)
        # signed *= decay

        if self.use_exp:
            return -self.lr * dW * np.exp(signed)
        else:
            return -self.lr * dW * np.power(self.alpha, signed)


class Neumann(Optimizer):
    # paper: https://arxiv.org/abs/1712.03298
    # NOTE: this implementation omits resetting as described in the paper.
    #       resetting is totally disabled for now.
    # NOTE: this implementation does not use vanilla SGD for its first epochs.
    #       you can do this yourself if you really want to.
    #       it seems to be enough to use a slow-starting Learner like SineCLR.

    def __init__(self, lr=0.01, delta=1.0,
                 alpha=1e-7, beta=1e-5, gamma=0.99, mu_min=0.5, mu_max=0.9):
        self.delta = _f(delta) # delta-time.
        self.alpha = _f(alpha)  # cubic.
        self.beta = _f(beta)  # repulsive. NOTE: multiplied by len(dW) later.
        self.gamma = _f(gamma)  # EMA, or 1-pole low-pass parameter. same thing.
        # momentum is in the shape of 1 - 1/(1 + t)
        self.mu_min = _f(mu_min)
        self.mu_max = _f(mu_max)
        self.reset_period = 0  # TODO

        super().__init__(lr)

    def reset(self):
        # NOTE: mt and vt are different than the pair in Adam-like optimizers.
        self.mt = None  # momentum accumulator.
        self.vt = None  # weight accumulator.
        self.t = 0

        super().reset()

    def compute(self, dW, W):
        raise Exception("compute() is not available for this Optimizer.")

    def update(self, dW, W):
        if self.mt is None:
            self.mt = np.zeros_like(dW)
        if self.vt is None:
            self.vt = np.zeros_like(dW)

        if self.reset_period > 0 and (self.t - 1) % self.reset_period == 0:
            self.mt = -self.lr * dW
            return

        # momentum quantity:
        mu = _1 - _1/_f(self.t + _1)
        mu = (self.mu_max - self.mu_max) * mu + self.mu_min

        self.t += self.delta

        # change in smoothed weights:
        delta = W - self.vt
        delta_norm_squared = np.square(delta).sum()
        delta_norm = np.sqrt(delta_norm_squared)

        # regularization terms: (push and pull)
        cubic_reg = self.alpha * delta_norm_squared
        repulsive_reg = self.beta * dW.size / delta_norm_squared
        dt = dW + (cubic_reg - repulsive_reg) * (delta / delta_norm)

        # Richardson iteration disguised as plain momentum:
        self.mt = mu * self.mt - self.lr * dt
        # this is only a good approximation for small ||self.lr * self.mt||.

        # update weights and moving average:
        W += mu * self.mt - self.lr * dt  # essentially Nesterov momentum.
        self.vt = W + self.gamma * (self.vt - W)


class Adamax(Optimizer):
    # TODO: paper?

    def __init__(self, lr=0.001, b1=0.9, b2=0.999, debias=False, eps=1e-8):
        self.b1 = _f(b1)  # decay term
        self.b2 = _f(b2)  # decay term
        self.b1_t_default = _f(b1)  # decay term power t
        self.b2_t_default = _f(b2)  # decay term power t
        self.debias = bool(debias)
        self.eps = _f(eps)

        super().__init__(lr)

    def reset(self):
        self.mt = None
        self.vt = None
        self.b1_t = self.b1_t_default
        self.b2_t = self.b2_t_default

        super().reset()

    def compute(self, dW, W):
        if self.mt is None:
            self.mt = np.zeros_like(dW)
        if self.vt is None:
            self.vt = np.zeros_like(dW)
            #self.vt = np.full_like(dW, 0.001)  # NOTE: experimenting.
            #self.vt = np.full_like(dW, self.lr)  # NOTE: experimenting.

        mt = filter_gradients(self.mt, dW, self.b1)
        vt = np.maximum(self.b2 * self.vt, np.abs(dW))

        if self.debias:
            if self.b1_t != 1:
                mt = mt / (1 - self.b1_t)
            if self.b2_t != 1:
                vt = vt / (1 - self.b2_t)

            # decay gain.
            self.b1_t *= self.b1
            self.b2_t *= self.b2

        return -self.lr * mt / (vt + self.eps)


class Adamlike(Optimizer):
    # this generalizes a lot of algorithms that
    # either subsets or supersets the Adam optimizer.
    # refer to the subclasses for details.

    # these defaults match Adam's.
    def __init__(self, lr=0.001, b1=0.9, b2=0.999, power=1/2,
                 debias=True, runmax=False, yogi=False, eps=1e-8):
        self.b1 = _f(b1)  # decay term
        self.b2 = _f(b2)  # decay term
        self.b1_t_default = _f(b1)  # decay term power t
        self.b2_t_default = _f(b2)  # decay term power t
        self.power = _f(power)
        self.debias = bool(debias)
        self.runmax = bool(runmax)
        self.yogi = bool(yogi)
        self.eps = _f(eps)

        super().__init__(lr)

    def reset(self):
        self.mt = None
        self.vt = None
        self.vtmax = None
        self.b1_t = self.b1_t_default
        self.b2_t = self.b2_t_default

        super().reset()

    def compute(self, dW, W):
        if self.mt is None:
            self.mt = np.zeros_like(dW)
        if self.vt is None:
            self.vt = np.zeros_like(dW)
        if self.vtmax is None and self.runmax:
            self.vtmax = np.zeros_like(dW)

        # keep local references of mt and vt to simplify
        # implementing all the variations of Adam later.
        mt = filter_gradients(self.mt, dW, self.b1)
        if self.yogi:
            g2 = np.square(dW)
            vt = self.vt
            vt -= (1 - self.b2) * np.sign(vt - g2) * g2
        else:
            vt = filter_gradients(self.vt, np.square(dW), self.b2)

        if self.runmax:
            self.vtmax[:] = np.maximum(vt, self.vtmax)
            vt = self.vtmax

        if self.debias:
            if self.b1_t != 1:
                mt = mt / (1 - self.b1_t)
            if self.b2_t != 1:
                vt = vt / (1 - self.b2_t)

        if self.power == 0:
            delta = mt
        elif self.power == 1:
            delta = mt / (vt + self.eps)
        elif self.power == 1/2:  # TODO: is this actually faster?
            delta = mt / (np.sqrt(vt) + self.eps)
        elif self.power == 1/3:  # TODO: is this actually faster?
            delta = mt / (np.cbrt(vt) + self.eps)
        else:
            delta = mt / (vt**self.power + self.eps)

        if self.debias:
            # decay gain.
            self.b1_t *= self.b1
            self.b2_t *= self.b2

        return -self.lr * delta


class Adagrad(Adamlike):
    # paper: https://web.stanford.edu/~jduchi/projects/DuchiHaSi11.pdf

    def __init__(self, lr=0.01, eps=1e-8):
        super().__init__(lr=lr, b1=0.0, b2=-1.0, power=1/2,
                         debias=False, runmax=False, eps=eps)

    @property
    def g(self):
        return self.vt

    @g.setter
    def g(self, value):
        self.vt = value


class RMSprop(Adamlike):
    # slides: http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf

    def __init__(self, lr=0.001, mu=0.99, eps=1e-8):
        super().__init__(lr=lr, b1=0.0, b2=mu, power=1/2,
                         debias=False, runmax=False, eps=eps)

    @property
    def mu(self):
        return self.b2

    @mu.setter
    def mu(self, value):
        self.b2 = value

    @property
    def g(self):
        return self.vt

    @g.setter
    def g(self, value):
        self.vt = value


class Adam(Adamlike):
    # paper: https://arxiv.org/abs/1412.6980
    # Adam generalizes RMSprop, and
    # adds a decay term to the regular (non-squared) delta, and performs
    # debiasing to compensate for the filtered deltas starting from zero.

    def __init__(self, lr=0.001, b1=0.9, b2=0.999,
                 debias=True, eps=1e-8):
        super().__init__(lr=lr, b1=b1, b2=b2, power=1/2,
                         debias=debias, runmax=False, yogi=False, eps=eps)


class Yogi(Adamlike):
    # paper: https://papers.nips.cc/paper/8186-adaptive-methods-for-nonconvex-optimization.pdf
    # based on Adam. this changes the filtering for vt.

    def __init__(self, lr=0.01, b1=0.9, b2=0.999,
                 debias=True, eps=1e-3):
        super().__init__(lr=lr, b1=b1, b2=b2, power=1/2,
                         debias=debias, runmax=False, yogi=True, eps=eps)


class AMSgrad(Adamlike):
    # paper: https://openreview.net/forum?id=ryQu7f-RZ
    # based on Adam. this simply adds a running element-wise maximum to vt.

    def __init__(self, lr=0.001, b1=0.9, b2=0.999,
                 debias=True, eps=1e-8):
        super().__init__(lr=lr, b1=b1, b2=b2, power=1/2,
                         debias=debias, runmax=True, yogi=False, eps=eps)


class Padam(Adamlike):
    # paper: https://arxiv.org/abs/1806.06763
    # paper: https://arxiv.org/abs/1808.05671
    # based on AMSgrad. this configures the power of vt to be closer to zero.

    def __init__(self, lr=0.1, b1=0.9, b2=0.999,
                 power=1/8, debias=True, eps=1e-8):
        super().__init__(lr=lr, b1=b1, b2=b2, power=power,
                         debias=debias, runmax=True, yogi=False, eps=eps)
