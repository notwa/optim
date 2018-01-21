import types
import numpy as np

from .floats import *

class Ritual: # i'm just making up names at this point.
    def __init__(self, learner=None):
        self.learner = learner if learner is not None else Learner(Optimizer())
        self.model = None

    def reset(self):
        self.learner.reset(optim=True)
        self.en = 0
        self.bn = 0

    def learn(self, inputs, outputs):
        error, predicted = self.model.forward(inputs, outputs)
        self.model.backward(predicted, outputs)
        self.model.regulate()
        return error, predicted

    def update(self):
        optim = self.learner.optim
        optim.model = self.model
        optim.update(self.model.dW, self.model.W)

    def prepare(self, model):
        self.en = 0
        self.bn = 0
        self.model = model

    def _train_batch(self, batch_inputs, batch_outputs, b, batch_count,
                     test_only=False, loss_logging=False, mloss_logging=True):
        if not test_only and self.learner.per_batch:
            self.learner.batch(b / batch_count)

        if test_only:
            predicted = self.model.evaluate(batch_inputs, deterministic=True)
        else:
            error, predicted = self.learn(batch_inputs, batch_outputs)
            self.model.regulate_forward()
            self.update()

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
                raise Exception("shuffling is incompatibile with using a generator.")
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
                assert batch_size == prev_batch_size or prev_batch_size is None, \
                  "non-constant batch size (got {}, expected {})".format(batch_size, prev_batch_size)
            else:
                bi = b * batch_size
                batch_inputs  = inputs[ bi:bi+batch_size]
                batch_outputs = outputs[bi:bi+batch_size]

            if clear_grad:
                self.model.clear_grad()
            self._train_batch(batch_inputs, batch_outputs, b, batch_count,
                              test_only, return_losses=='both', return_losses)

            prev_batch_size = batch_size

        avg_mloss = self.cumsum_mloss / _f(batch_count)
        if return_losses == 'both':
            avg_loss = self.cumsum_loss / _f(batch_count)
            return avg_loss, avg_mloss, self.losses, self.mlosses
        elif return_losses:
            return avg_mloss, self.mlosses
        return avg_mloss

    def test_batched(self, inputs, outputs, *args, **kwargs):
        return self.train_batched(inputs, outputs, *args,
                                  test_only=True, **kwargs)

    def train_batched_gen(self, generator, batch_count, *args, **kwargs):
        return self.train_batched(generator, batch_count, *args,
                                  shuffle=False, **kwargs)
