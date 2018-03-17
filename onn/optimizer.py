import numpy as np

from .float import _f, _0, _1
from .optimizer_base import *
from .utility import *

# some of the the following optimizers are blatantly lifted from tiny-dnn:
# https://github.com/tiny-dnn/tiny-dnn/blob/master/tiny_dnn/optimizers/optimizer.h


class Momentum(Optimizer):
    def __init__(self, lr=0.01, mu=0.9, nesterov=False):
        self.mu = _f(mu)  # momentum
        self.nesterov = bool(nesterov)

        super().__init__(lr)

    def reset(self):
        self.Vprev = None

    def compute(self, dW, W):
        if self.Vprev is None:
            self.Vprev = np.copy(dW)

        V = self.mu * self.Vprev - self.lr * dW
        self.Vprev[:] = V
        if self.nesterov:
            return self.mu * V - self.lr * dW

        return V


class Adagrad(Optimizer):
    def __init__(self, lr=0.01, eps=1e-8):
        self.eps = _f(eps)

        super().__init__(lr)

    def reset(self):
        self.g = None

    def compute(self, dW, W):
        if self.g is None:
            self.g = np.zeros_like(dW)

        self.g += np.square(dW)
        return -self.lr * dW / (np.sqrt(self.g) + self.eps)


class RMSprop(Optimizer):
    # RMSprop generalizes* Adagrad, etc.

    # * RMSprop == Adagrad when
    #   RMSprop.mu == 1

    def __init__(self, lr=1e-4, mu=0.99, eps=1e-8):
        self.mu = _f(mu)  # decay term
        self.eps = _f(eps)

        # one might consider the following equation when specifying mu:
        # mu = e**(-1/t)
        # default: t = -1/ln(0.99) = ~99.5
        # therefore the default of mu=0.99 means
        # an input decays to 1/e its original amplitude over 99.5 batches.
        # (this is from DSP, so how relevant it is in SGD is debatable)

        super().__init__(lr)

    def reset(self):
        self.g = None

    def compute(self, dW, W):
        if self.g is None:
            self.g = np.zeros_like(dW)

        # basically apply a first-order low-pass filter to delta squared,
        self.g += (1 - self.mu) * (np.square(dW) - self.g)

        # and sqrt it to complete the running root-mean-square approximation.
        return -self.lr * dW / (np.sqrt(self.g) + self.eps)


class RMSpropCentered(Optimizer):
    # referenced TensorFlow/PyTorch.
    # paper: https://arxiv.org/pdf/1308.0850v5.pdf

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


class Adam(Optimizer):
    # paper: https://arxiv.org/abs/1412.6980
    # Adam generalizes* RMSprop, and
    # adds a decay term to the regular (non-squared) delta, and performs
    # debiasing to compensate for the filtered deltas starting from zero.

    # * Adam == RMSprop when
    #   Adam.b1 == 0
    #   Adam.b2 == RMSprop.mu

    def __init__(self, lr=0.002, b1=0.9, b2=0.999, eps=1e-8):
        self.b1 = _f(b1)  # decay term
        self.b2 = _f(b2)  # decay term
        self.b1_t_default = _f(b1)  # decay term power t
        self.b2_t_default = _f(b2)  # decay term power t
        self.eps = _f(eps)

        super().__init__(lr)

    def reset(self):
        self.mt = None
        self.vt = None
        self.b1_t = self.b1_t_default
        self.b2_t = self.b2_t_default

    def compute(self, dW, W):
        if self.mt is None:
            self.mt = np.zeros_like(dW)
        if self.vt is None:
            self.vt = np.zeros_like(dW)

        # decay gain
        self.b1_t *= self.b1
        self.b2_t *= self.b2

        # filter
        self.mt += (1 - self.b1) * (dW - self.mt)
        self.vt += (1 - self.b2) * (np.square(dW) - self.vt)

        return -self.lr * (self.mt / (1 - self.b1_t)) \
            / (np.sqrt(self.vt / (1 - self.b2_t)) + self.eps)


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


# more

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


class YellowFin(Optimizer):
    # paper: https://arxiv.org/abs/1706.03471
    # knowyourmeme: http://cs.stanford.edu/~zjian/project/YellowFin/
    # author's implementation:
    # https://github.com/JianGoForIt/YellowFin/blob/master/tuner_utils/yellowfin.py
    # code lifted:
    # https://gist.github.com/botev/f8b32c00eafee222e47393f7f0747666

    def __init__(self, lr=0.1, mu=0.0, beta=0.999, window_size=20,
                 debias=True, clip=1.0):
        self.lr_default = _f(lr)
        self.mu_default = _f(mu)
        self.beta = _f(beta)
        self.window_size = int(window_size)  # curv_win_width
        self.debias_enabled = bool(debias)
        self.clip = _f(clip)

        self.mu = _f(mu)  # momentum
        super().__init__(lr)

    def reset(self):
        self.accum = None

        self.lr = self.lr_default
        self.mu = self.mu_default

        self.step = 0
        self.beta_t = self.beta

        self.curv_win = np.zeros([self.window_size, ], dtype=np.float32)

        self.h_min = None
        self.h_max = None

        self.g_lpf = 0
        # self.g_squared_lpf = 0
        self.g_norm_squared_lpf = 0
        self.g_norm_lpf = 0
        self.h_min_lpf = 0
        self.h_max_lpf = 0
        self.dist_lpf = 0
        self.lr_lpf = 0
        self.mu_lpf = 0

    def get_lr_mu(self):
        p = (np.square(self.dist_avg) * np.square(self.h_min)) \
            / (2 * self.g_var)
        w3 = p * (np.sqrt(0.25 + p / 27.0) - 0.5)
        w = np.power(w3, 1/3)
        y = w - p / (3 * w)
        sqrt_mu1 = y + 1

        sqrt_h_min = np.sqrt(self.h_min)
        sqrt_h_max = np.sqrt(self.h_max)
        sqrt_mu2 = (sqrt_h_max - sqrt_h_min) / (sqrt_h_max + sqrt_h_min)

        sqrt_mu = max(sqrt_mu1, sqrt_mu2)
        if sqrt_mu2 > sqrt_mu1:
            print('note: taking dr calculation. something may have exploded.')

        lr = np.square(1 - sqrt_mu) / self.h_min
        mu = np.square(sqrt_mu)
        return lr, mu

    def compute(self, dW, W):
        if self.accum is None:
            self.accum = np.zeros_like(dW)

        # TODO: prevent allocations everywhere by using [:].
        #       assuming that really works. i haven't actually checked.

        total_norm = np.linalg.norm(dW)
        clip_scale = self.clip / (total_norm + 1e-6)
        if clip_scale < 1:
            # print("clipping gradients; norm: {:10.5f}".format(total_norm))
            dW *= clip_scale

        # fmt = 'W std: {:10.7f}e-3,  dWstd: {:10.7f}e-3,  V std: {:10.7f}e-3'
        # print(fmt.format(np.std(W), np.std(dW) * 100, np.std(V) * 100))

        b = self.beta
        m1b = 1 - self.beta
        debias = 1 / (1 - self.beta_t) if self.debias_enabled else 1

        g = dW
        g_squared = np.square(g)
        g_norm_squared = np.sum(g_squared)
        g_norm = np.sqrt(g_norm_squared)

        self.curv_win[self.step % self.window_size] = g_norm_squared
        valid_window = self.curv_win[:min(self.window_size, self.step + 1)]
        h_min_t = np.min(valid_window)
        h_max_t = np.max(valid_window)

        self.g_lpf = b * self.g_lpf + m1b * g
        # self.g_squared_lpf = b * self.g_squared_lpf + m1b * g_squared
        self.g_norm_squared_lpf = b * self.g_norm_squared_lpf \
            + m1b * g_norm_squared
        self.g_norm_lpf = b * self.g_norm_lpf + m1b * g_norm
        self.h_min_lpf = b * self.h_min_lpf + m1b * h_min_t
        self.h_max_lpf = b * self.h_max_lpf + m1b * h_max_t

        g_avg = debias * self.g_lpf
        # g_squared_avg = debias * self.g_squared_lpf
        g_norm_squared_avg = debias * self.g_norm_squared_lpf
        g_norm_avg = debias * self.g_norm_lpf
        self.h_min = debias * self.h_min_lpf
        self.h_max = debias * self.h_max_lpf
        assert self.h_max >= self.h_min

        dist = g_norm_avg / g_norm_squared_avg

        self.dist_lpf = b * self.dist_lpf + m1b * dist

        self.dist_avg = debias * self.dist_lpf

        self.g_var = g_norm_squared_avg - np.sum(np.square(g_avg))
        # equivalently:
        # self.g_var = np.sum(np.abs(g_squared_avg - np.square(g_avg)))

        if self.step > 0:
            lr_for_real, mu_for_real = self.get_lr_mu()
            self.mu_lpf = b * self.mu_lpf + m1b * mu_for_real
            self.lr_lpf = b * self.lr_lpf + m1b * lr_for_real
            self.mu = debias * self.mu_lpf
            self.lr = debias * self.lr_lpf

        self.accum[:] = self.accum * self.mu - self.lr * dW
        V = self.accum

        self.step += 1
        self.beta_t *= self.beta
        return V


class AddSign(Optimizer):
    # paper: https://arxiv.org/abs/1709.07417

    def __init__(self, lr=0.01, mu=0.9, alpha=1):
        self.mu = _f(mu)
        self.alpha = _f(alpha)

        super().__init__(lr)

    def reset(self):
        self.accum = None

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
    # NOTE: this implementation is missing resetting as described in the paper.
    #       resetting is totally disabled for now.
    # NOTE: this implementation does not use vanilla SGD for its first epochs.
    #       you should do this yourself if you need it.
    #       it seems like using a Learner like SineCLR makes this unnecessary.

    def __init__(self, lr=0.01):
        self.alpha = _f(1e-7)  # cubic.
        self.beta = _f(1e-5)  # repulsive. NOTE: multiplied by len(dW) later.
        self.gamma = _f(0.99)  # EMA, or 1-pole low-pass parameter. same thing.
        # momentum is ∝ (in the shape of) 1 - 1/(1 + t)
        self.mu_min = _f(0.5)
        self.mu_max = _f(0.9)
        self.reset_period = 0  # TODO

        super().__init__(lr)

    def reset(self):
        # NOTE: mt and vt are different than the pair in Adam-like optimizers.
        self.mt = None  # momentum accumulator.
        self.vt = None  # weight accumulator.
        self.t = 0

    def compute(self, dW, W):
        raise Exception("compute() is not available for this Optimizer.")

    def update(self, dW, W):
        self.t += 1

        if self.mt is None:
            self.mt = np.zeros_like(dW)
        if self.vt is None:
            self.vt = np.zeros_like(dW)

        if self.reset_period > 0 and (self.t - 1) % self.reset_period == 0:
            self.mt = -self.lr * dW
            return

        # momentum quantity:
        mu = _1 - _1/_f(self.t)  # the + 1 is implicit.
        mu = (mu + self.mu_min) * (self.mu_max - self.mu_min)

        # smoothed change in weights:
        delta = W - self.vt
        delta_norm_squared = np.square(delta).sum()
        delta_norm = np.sqrt(delta_norm_squared)

        # regularization terms: (push and pull)
        cubic_reg = self.alpha * delta_norm_squared
        repulsive_reg = self.beta * dW.size / delta_norm_squared
        dt = dW + (cubic_reg - repulsive_reg) * (delta / delta_norm)

        # plain momentum:
        self.mt = mu * self.mt - self.lr * dt

        # weights and accumulator:
        W += mu * self.mt - self.lr * dt
        self.vt = W + self.gamma * (self.vt - W)


class AMSgrad(Optimizer):
    # paper: https://openreview.net/forum?id=ryQu7f-RZ
    # based on Adam. this simply adds a running element-wise maximum to vt.

    def __init__(self, lr=0.002, b1=0.9, b2=0.999, eps=1e-8, debias=True):
        self.b1 = _f(b1)  # decay term
        self.b2 = _f(b2)  # decay term
        self.b1_t_default = _f(b1)  # decay term power t
        self.b2_t_default = _f(b2)  # decay term power t
        self.eps = _f(eps)
        self.debias = bool(debias)

        super().__init__(lr)

    def reset(self):
        self.mt = None
        self.vt = None
        self.vtmax = None
        self.b1_t = self.b1_t_default
        self.b2_t = self.b2_t_default

    def compute(self, dW, W):
        if self.mt is None:
            self.mt = np.zeros_like(dW)
        if self.vt is None:
            self.vt = np.zeros_like(dW)
        if self.vtmax is None:
            self.vtmax = np.zeros_like(dW)

        # filter
        self.mt += (1 - self.b1) * (dW - self.mt)
        self.vt += (1 - self.b2) * (np.square(dW) - self.vt)

        self.vtmax = np.maximum(self.vtmax, self.vt)

        if self.debias:
            ret = -self.lr * (self.mt / (1 - self.b1_t)) \
                / (np.sqrt(self.vtmax / (1 - self.b2_t)) + self.eps)
        else:
            ret = -self.lr * self.mt / (np.sqrt(self.vtmax) + self.eps)

        # decay gain
        self.b1_t *= self.b1
        self.b2_t *= self.b2

        return ret
