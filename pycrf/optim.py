"""Defines command line interface to torch optimizer classes."""

from abc import ABC, abstractstaticmethod, abstractclassmethod
import argparse
import sys
from typing import Iterable, Dict, Type, Union
from warnings import warn

import torch
import numpy as np


class CLOptim(ABC):
    """Command line optimizer interface."""

    @staticmethod
    @abstractstaticmethod
    def cl_opts(parser: argparse.ArgumentParser, require=True) -> None:
        """Add command line options specific to this optimizer."""
        pass

    @classmethod
    @abstractclassmethod
    def cl_init(cls, params: Iterable, opts: argparse.Namespace):
        """Initialize optimizer from command line options."""
        pass

    def epoch_update(self, loss: float) -> None:
        """Do any updating (such as lr update) after each epoch."""
        pass


class AdaGrad(torch.optim.Adagrad, CLOptim):
    """Wraps torch AdaGrad optimizer."""

    @staticmethod
    def cl_opts(parser: argparse.ArgumentParser, require=True) -> None:
        """Add command line options specific to this optimizer."""
        group = parser.add_argument_group("AdaGrad options")
        group.add_argument(
            "--lr",
            type=float,
            default=0.01,
            help="""Learning rate. Default is 0.01."""
        )
        group.add_argument(
            "--weight-decay",
            type=float,
            default=0.,
            help="""L2 penalty. Default is 0."""
        )
        group.add_argument(
            "--lr-decay",
            type=float,
            default=0.,
            help="""Learning rate decay. Default is 0."""
        )

    @classmethod
    def cl_init(cls, params: Iterable, opts: argparse.Namespace):
        """Initialize this optimizer from the command line."""
        return cls(params,
                   lr=opts.lr,
                   lr_decay=opts.lr_decay,
                   weight_decay=opts.weight_decay)


class AdaDelta(torch.optim.Adadelta, CLOptim):
    """Wraps torch AdaDelta optimizer."""

    @staticmethod
    def cl_opts(parser: argparse.ArgumentParser, require=True) -> None:
        """Add command line options specific to this optimizer."""
        group = parser.add_argument_group("AdaDelta options")
        group.add_argument(
            "--lr",
            type=float,
            default=1.,
            help="""Learning rate. Default is 1.0."""
        )
        group.add_argument(
            "--eps",
            type=float,
            default=1e-6,
            help="""Term added to denominator for numerical stability. Default
            is 1e-6."""
        )
        group.add_argument(
            "--weight-decay",
            type=float,
            default=0.,
            help="""L2 penalty. Default is 0."""
        )
        group.add_argument(
            "--rho",
            type=float,
            default=0.9,
            help="""Coefficient used for computing running average of squared
            gradients. Default is 0.9."""
        )

    @classmethod
    def cl_init(cls, params: Iterable, opts: argparse.Namespace):
        """Initialize this optimizer from the command line."""
        return cls(params,
                   lr=opts.lr,
                   eps=opts.eps,
                   rho=opts.rho,
                   weight_decay=opts.weight_decay)


class RMSProp(torch.optim.RMSprop, CLOptim):
    """Wraps torch RMSProp optimizer."""

    @staticmethod
    def cl_opts(parser: argparse.ArgumentParser, require=True) -> None:
        """Add command line options specific to this optimizer."""
        group = parser.add_argument_group("RMSProp options")
        group.add_argument(
            "--lr",
            type=float,
            default=0.01,
            help="""Learning rate. Default is 0.01."""
        )
        group.add_argument(
            "--alpha",
            type=float,
            default=0.99,
            help="""Smoothing constant. Default is 0.99."""
        )
        group.add_argument(
            "--eps",
            type=float,
            default=1e-8,
            help="""Term added to denominator for numerical stability. Default
            is 1e-8."""
        )
        group.add_argument(
            "--centered",
            action="store_true",
            help="""Compute the centered RMSProp, where the gradient is
            normalized by an estimation of its variance."""
        )
        group.add_argument(
            "--momentum",
            type=float,
            default=0.,
            help="""Momentum factor. Default is 0."""
        )
        group.add_argument(
            "--weight-decay",
            type=float,
            default=0.,
            help="""L2 penalty. Default is 0."""
        )

    @classmethod
    def cl_init(cls, params: Iterable, opts: argparse.Namespace):
        """Initialize this optimizer from the command line."""
        return cls(params,
                   lr=opts.lr,
                   alpha=opts.alpha,
                   eps=opts.eps,
                   centered=opts.centered,
                   momentum=opts.momentum,
                   weight_decay=opts.weight_decay)


class Adam(torch.optim.Adam, CLOptim):
    """Wraps torch Adam optimizer."""

    @staticmethod
    def cl_opts(parser: argparse.ArgumentParser, require=True) -> None:
        """Add command line options specific to Adam."""
        group = parser.add_argument_group("Adam options")
        group.add_argument(
            "--lr",
            type=float,
            default=0.001,
            help="""Learning rate. Default is 0.001."""
        )
        group.add_argument(
            "--beta1",
            type=float,
            default=0.9,
            help="""Coefficient used for computing running averages of gradient
            and its square. Default is 0.9"""
        )
        group.add_argument(
            "--beta2",
            type=float,
            default=0.999,
            help="""Coefficient used for computing running averages of gradient
            and its square. Default is 0.999"""
        )
        group.add_argument(
            "--eps",
            type=float,
            default=1e-8,
            help="""Term added to denominator to add numerical stabiliary.
            Default is 1e-8."""
        )
        group.add_argument(
            "--amsgrad",
            action="store_true",
            help="""Use the AMSGrad variant."""
        )
        group.add_argument(
            "--weight-decay",
            type=float,
            default=0.,
            help="""L2 penalty. Default is 0."""
        )

    @classmethod
    def cl_init(cls, params: Iterable, opts: argparse.Namespace):
        """Initialize Adam optimizer from command line options."""
        return cls(params,
                   lr=opts.lr,
                   betas=(opts.beta1, opts.beta2),
                   eps=opts.eps,
                   amsgrad=opts.amsgrad,
                   weight_decay=opts.weight_decay)


class SparseAdam(torch.optim.SparseAdam, CLOptim):
    """Wraps torch SparseAdam optimizer."""

    @staticmethod
    def cl_opts(parser: argparse.ArgumentParser, require=True) -> None:
        """Add command line options specific to Adam."""
        group = parser.add_argument_group("SparseAdam options")
        group.add_argument(
            "--lr",
            type=float,
            default=0.001,
            help="""Learning rate. Default is 0.001."""
        )
        group.add_argument(
            "--beta1",
            type=float,
            default=0.9,
            help="""Coefficient used for computing running averages of gradient
            and its square. Default is 0.9"""
        )
        group.add_argument(
            "--beta2",
            type=float,
            default=0.999,
            help="""Coefficient used for computing running averages of gradient
            and its square. Default is 0.999"""
        )
        group.add_argument(
            "--eps",
            type=float,
            default=1e-8,
            help="""Term added to denominator to add numerical stabiliary.
            Default is 1e-8."""
        )

    @classmethod
    def cl_init(cls, params: Iterable, opts: argparse.Namespace):
        """Initialize Adam optimizer from command line options."""
        return cls(params,
                   lr=opts.lr,
                   betas=(opts.beta1, opts.beta2),
                   eps=opts.eps)


class SGD(torch.optim.SGD, CLOptim):
    """Wraps torch stochastic gradient descent optimizer."""

    def __init__(self,
                 params: Iterable,
                 decay_rate: float = 0.,
                 decay_start: int = 0,
                 conditional_decay: bool = False,
                 cyclic: bool = False,
                 cycle_len: int = 10,
                 cycle_mult: int = None,
                 lr: float = None,
                 **kwargs) -> None:
        self.decay_rate = decay_rate
        self.decay_start = decay_start
        self.conditional_decay = conditional_decay
        self.initial_lr: float = lr
        self.epoch: int = 0
        self.last_loss: Union[None, float] = None
        self.cyclic = cyclic
        self.cycle_len = cycle_len
        self.cycle_mult = cycle_mult
        self._cycle_fact: float = 1.
        self._cycle_counter: int = 1

        kwargs["lr"] = lr
        super(SGD, self).__init__(params, **kwargs)

    @property
    def lr(self):
        """Get the learning rate."""
        return self.param_groups[0]["lr"]

    @lr.setter
    def lr(self, value: float) -> None:
        """Set the learning rate."""
        for param_group in self.param_groups:
            param_group["lr"] = value

    @staticmethod
    def cl_opts(parser: argparse.ArgumentParser, require=True) -> None:
        """Add command line options specific to SGD."""
        group = parser.add_argument_group("SGD options")
        group.add_argument(
            "--lr",
            type=float,
            default=0.01,
            help="""Learning rate"""
        )
        group.add_argument(
            "--momentum",
            type=float,
            default=0.,
            help="""Momentum factor. Default is 0, but a sensible value might
            be 0.9."""
        )
        group.add_argument(
            "--nesterov",
            action="store_true",
            help="""Enable Nesterov momentum."""
        )
        group.add_argument(
            "--decay-rate",
            type=float,
            default=0.,
            help="""Learning rate decay factor. Default is 0, but a sensible
            value might be 0.05."""
        )
        group.add_argument(
            "--weight-decay",
            type=float,
            default=0.,
            help="""L2 penalty. Default is 0."""
        )
        group.add_argument(
            "--decay-start",
            type=int,
            default=0,
            help="""Epoch to start decay at if loss doesn't decrease."""
        )
        group.add_argument(
            "--conditional-decay",
            action="store_true",
            help="""If set, learning rate decay will only happen when the loss
            does not decrease from the previous round."""
        )
        group.add_argument(
            "--cycle_len",
            type=int,
            help="""Cycle length (in epochs) for a cyclic learning rate annealing
            schedule."""
        )
        group.add_argument(
            "--cycle_mult",
            type=float,
            help="""Factor by which the cyclic annealing schedule is slowed."""
        )
        return None

    @classmethod
    def cl_init(cls, params: Iterable, opts: argparse.Namespace):
        """Initialize SGD from command line options."""
        kwargs = {}
        if opts.cycle_len:
            kwargs["cycle_len"] = opts.cycle_len
            kwargs["cyclic"] = True
            if opts.cycle_mult:
                kwargs["cycle_mult"] = opts.cycle_mult
        elif opts.cycle_mult:
            warn("Unused option: '--cycle_mult'. If you want to use a cyclic learning "
                 "rate schedule, you must specify the option '--cycle_len' as well.")
            sys.stderr.flush()
        return cls(params,
                   decay_rate=opts.decay_rate,
                   lr=opts.lr,
                   momentum=opts.momentum,
                   nesterov=opts.nesterov,
                   weight_decay=opts.weight_decay,
                   decay_start=opts.decay_start,
                   conditional_decay=opts.conditional_decay,
                   **kwargs)

    def _vanilla_decay(self, loss: float) -> None:
        # pylint: disable=attribute-defined-outside-init
        """
        Perform 'vanilla' LR decay.

        This just follows a simple annealing schedule in which the LR is decreased
        by a factor at the end of each epoch.
        """
        # Check if we should decrease the learning rate.
        if self.decay_rate > 0 and self.decay_start <= self.epoch:
            last_loss_less: bool = self.last_loss is not None and self.last_loss <= loss
            # Decrease the learning rate if `self.conditional_decay = False` or
            # if `self.conditional_decay = True` and the loss has not decreased
            # from the previous round.
            if not self.conditional_decay or last_loss_less:
                self.lr = self.initial_lr / (1 + self.decay_rate * self.epoch)
                print(f"Updating learning rate to {self.lr}", flush=True)

        # Update the latest loss.
        self.last_loss = loss

    def _cyclic_decay(self) -> None:
        # pylint: disable=attribute-defined-outside-init
        """
        Perform cyclic LR decay.

        See http://arxiv.org/abs/1704.00109.
        """
        updated_cycle_len = int(self._cycle_fact * self.cycle_len)
        if self.cycle_mult and \
                self.epoch > 1 and \
                (self._cycle_counter) % updated_cycle_len == 0:
            # Adjust the cycle length.
            self._cycle_fact *= self.cycle_mult
            self._cycle_counter = 0
        self.lr = \
            (self.initial_lr / 2) * \
            (
                np.cos(
                    np.pi * (
                        (self._cycle_counter) % updated_cycle_len
                    ) / updated_cycle_len
                )
                + 1
            )
        self._cycle_counter += 1
        print(f"Updating learning rate to {self.lr}", flush=True)

    def epoch_update(self, loss: float) -> None:
        # pylint: disable=invalid-name,attribute-defined-outside-init
        """Update the learning rate."""
        self.epoch += 1

        if not self.cyclic:
            self._vanilla_decay(loss)
        else:
            self._cyclic_decay()


OPTIM_ALIASES: Dict[str, Type[CLOptim]] = {
    "SGD": SGD,
    "Adam": Adam,
    "SparseAdam": SparseAdam,
    "AdaDelta": AdaDelta,
    "AdaGrad": AdaGrad,
    "RMSProp": RMSProp,
}
