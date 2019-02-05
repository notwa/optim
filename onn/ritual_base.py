import types
import numpy as np
from collections import namedtuple

from .float import _f, _0
from .utility import batchize

Losses = namedtuple("Losses", ["avg_loss", "avg_mloss", "losses", "mlosses"])


class Ritual:  # i'm just making up names at this point.
    def __init__(self, learner=None):
        self.learner = learner if learner is not None else Learner(Optimizer())
        self.model = None

    def reset(self):
        self.learner.reset(optim=True)
        self.en = 0
        self.bn = 0

    def prepare(self, model):
        self.en = 0
        self.bn = 0
        self.model = model

    def _learn(self, inputs, outputs):
        error, predicted = self.model.forward(inputs, outputs)
        self.model.backward(predicted, outputs)
        self.model.regulate()
        return error, predicted

    def _update(self):
        optim = self.learner.optim
        optim.model = self.model
        optim.update(self.model.dW, self.model.W)

    def _measure(self, predicted, outputs):
        loss = self.model.loss.forward(predicted, outputs)
        if np.isnan(loss):
            raise Exception("nan")
        self.losses.append(loss)
        self.cumsum_loss += loss

        mloss = self.model.mloss.forward(predicted, outputs)
        if np.isnan(mloss):
            raise Exception("nan")
        self.mlosses.append(mloss)
        self.cumsum_mloss += mloss

    def _train_batch_new(self, inputs, outputs, b, batch_count):
        if self.learner.per_batch:
            self.learner.batch(b / batch_count)

        error, predicted = self.model.forward(inputs, outputs)
        error += self.model.regulate_forward()
        self.model.backward(predicted, outputs)
        self.model.regulate()

        optim = self.learner.optim
        optim.model = self.model
        optim.update(self.model.dW, self.model.W)

        return predicted

    def _train_batch(self, batch_inputs, batch_outputs, b, batch_count,
                     test_only=False, loss_logging=False, mloss_logging=True):
        if not test_only and self.learner.per_batch:
            self.learner.batch(b / batch_count)

        if test_only:
            predicted = self.model.evaluate(batch_inputs, deterministic=True)
        else:
            error, predicted = self._learn(batch_inputs, batch_outputs)
            self.model.regulate_forward()
            self._update()

        if loss_logging:
            batch_loss = self.model.loss.forward(predicted, batch_outputs)
            if np.isnan(batch_loss):
                raise Exception("nan")
            self.losses.append(batch_loss)
            self.cumsum_loss += batch_loss

        if mloss_logging:
            # NOTE: this can use the non-deterministic predictions. fixme?
            batch_mloss = self.model.mloss.forward(predicted, batch_outputs)
            if np.isnan(batch_mloss):
                raise Exception("nan")
            self.mlosses.append(batch_mloss)
            self.cumsum_mloss += batch_mloss

    def train(self, batch_gen, batch_count, clear_grad=True):
        assert self.model is not None, "call prepare(model) before training"
        self.en += 1

        self.cumsum_loss, self.cumsum_mloss = _0, _0
        self.losses, self.mlosses = [], []

        for b, (inputs, outputs) in enumerate(batch_gen):
            self.bn += 1
            if clear_grad:
                self.model.clear_grad()
            predicted = self._train_batch_new(inputs, outputs, b, batch_count)
            self._measure(predicted, outputs)

        avg_mloss = self.cumsum_mloss / _f(batch_count)
        avg_loss = self.cumsum_loss / _f(batch_count)
        return Losses(avg_loss, avg_mloss, self.losses, self.mlosses)

    def train_batched(self, inputs_or_generator, outputs_or_batch_count,
                      batch_size=None,
                      return_losses=False, test_only=False, shuffle=True,
                      clear_grad=True):
        assert isinstance(return_losses, bool) or return_losses == 'both'
        assert self.model is not None

        gen = isinstance(inputs_or_generator, types.GeneratorType)
        if gen:
            generator = inputs_or_generator
            batch_count = outputs_or_batch_count
            assert isinstance(batch_count, int), type(batch_count)
        else:
            inputs = inputs_or_generator
            outputs = outputs_or_batch_count

        if not test_only:
            self.en += 1

        if shuffle:
            if gen:
                raise Exception(
                    "shuffling is incompatibile with using a generator.")
            indices = np.arange(inputs.shape[0])
            np.random.shuffle(indices)
            inputs = inputs[indices]
            outputs = outputs[indices]

        self.cumsum_loss, self.cumsum_mloss = _0, _0
        self.losses, self.mlosses = [], []

        if not gen:
            batch_count = inputs.shape[0] // batch_size
            # TODO: lift this restriction
            assert inputs.shape[0] % batch_size == 0, \
                "inputs is not evenly divisible by batch_size"

        prev_batch_size = None
        for b in range(batch_count):
            if not test_only:
                self.bn += 1

            if gen:
                batch_inputs, batch_outputs = next(generator)
                batch_size = batch_inputs.shape[0]
                # TODO: lift this restriction
                fmt = "non-constant batch size (got {}, expected {})"
                assert (batch_size == prev_batch_size
                        or prev_batch_size is None), \
                    fmt.format(batch_size, prev_batch_size)
            else:
                bi = b * batch_size
                batch_inputs = inputs[bi:bi+batch_size]
                batch_outputs = outputs[bi:bi+batch_size]

            if clear_grad:
                self.model.clear_grad()
            self._train_batch(batch_inputs, batch_outputs, b, batch_count,
                              test_only, return_losses == 'both',
                              return_losses)

            prev_batch_size = batch_size

        avg_mloss = self.cumsum_mloss / _f(batch_count)
        if return_losses == 'both':
            avg_loss = self.cumsum_loss / _f(batch_count)
            return avg_loss, avg_mloss, self.losses, self.mlosses
        elif return_losses:
            return avg_mloss, self.mlosses
        return avg_mloss

    def test_batched(self, inputs, outputs, batch_size=None):
        assert self.model is not None, "call prepare(model) before testing"

        if batch_size is None:
            batch_size = len(inputs)

        self.cumsum_loss, self.cumsum_mloss = _0, _0
        self.losses, self.mlosses = [], []

        batch_gen, batch_count = batchize(inputs, outputs, batch_size,
                                          shuffle=False)

        for inputs, outputs in batch_gen:
            predicted = self.model.evaluate(inputs)
            self._measure(predicted, outputs)

        avg_mloss = self.cumsum_mloss / _f(batch_count)
        avg_loss = self.cumsum_loss / _f(batch_count)
        return Losses(avg_loss, avg_mloss, self.losses, self.mlosses)
