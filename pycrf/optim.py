"""Defines command line interface to torch optimizer classes."""

from abc import ABC, abstractstaticmethod, abstractclassmethod
import argparse
import sys
from typing import Dict, Type, Tuple, List
from warnings import warn

import torch
import numpy as np


class CLOptim(ABC, torch.optim.Optimizer):
    """Command line optimizer interface."""

    @property
    def lr(self):
        """Get the learning rate."""
        return tuple(x["lr"] for x in self.param_groups)

    @lr.setter
    def lr(self, values: Tuple[float]) -> None:
        """Set the learning rate."""
        for param_group, value in zip(self.param_groups, values):
            param_group["lr"] = value

    @staticmethod
    @abstractstaticmethod
    def cl_opts(parser: argparse.ArgumentParser, require=True) -> None:
        """Add command line options specific to this optimizer."""
        raise NotImplementedError()

    @classmethod
    @abstractclassmethod
    def cl_init(cls, params: List[dict], opts: argparse.Namespace):
        """Initialize optimizer from command line options."""
        raise NotImplementedError()

    def iteration_update(self, i: int) -> None:
        """Do any updating (such as lr update) after an iteration (mini-batch)."""
        pass

    def epoch_update(self, loss: float) -> None:
        """Do any updating (such as lr update) after each epoch."""
        pass

    def epoch_prepare(self, training_size: int, batch_size: int) -> None:
        """Prepare optimizer for an epoch of training."""
        pass

    @staticmethod
    def update_param_groups(params: List[dict],
                            opts: argparse.Namespace) -> None:
        """Add learning rate to param groups."""
        lrs: Tuple[float, ...]
        if len(params) > 1:
            lrs = (opts.lr_word_emb if opts.lr_word_emb else opts.lr, opts.lr)
        else:
            lrs = (opts.lr,)

        for param_group, lr in zip(params, lrs):
            param_group["lr"] = lr


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
            "--lr-word-emb",
            type=float,
            help="""Learning rate for word embeddings, only. A sensible value
            would be 3-10x smaller than the default learning rate."""
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
    def cl_init(cls, params: List[dict], opts: argparse.Namespace):
        """Initialize this optimizer from the command line."""
        cls.update_param_groups(params, opts)
        return cls(params,
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
            "--lr-word-emb",
            type=float,
            help="""Learning rate for word embeddings, only. A sensible value
            would be 3-10x smaller than the default learning rate."""
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
    def cl_init(cls, params: List[dict], opts: argparse.Namespace):
        """Initialize this optimizer from the command line."""
        cls.update_param_groups(params, opts)
        return cls(params,
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
            "--lr-word-emb",
            type=float,
            help="""Learning rate for word embeddings, only. A sensible value
            would be 3-10x smaller than the default learning rate."""
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
    def cl_init(cls, params: List[dict], opts: argparse.Namespace):
        """Initialize this optimizer from the command line."""
        cls.update_param_groups(params, opts)
        return cls(params,
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
            "--lr-word-emb",
            type=float,
            help="""Learning rate for word embeddings, only. A sensible value
            would be 3-10x smaller than the default learning rate."""
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
    def cl_init(cls, params: List[dict], opts: argparse.Namespace):
        """Initialize Adam optimizer from command line options."""
        cls.update_param_groups(params, opts)
        return cls(params,
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
            "--lr-word-emb",
            type=float,
            help="""Learning rate for word embeddings, only. A sensible value
            would be 3-10x smaller than the default learning rate."""
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
    def cl_init(cls, params: List[dict], opts: argparse.Namespace):
        """Initialize Adam optimizer from command line options."""
        cls.update_param_groups(params, opts)
        return cls(params,
                   betas=(opts.beta1, opts.beta2),
                   eps=opts.eps)


class SGD(torch.optim.SGD, CLOptim):
    """Wraps torch stochastic gradient descent optimizer."""

    def __init__(self,
                 params: List[dict],
                 cycle_len: int = None,
                 cycle_mult: float = 1.,
                 **kwargs) -> None:
        self.initial_lr = tuple(x["lr"] for x in params)
        self.epoch: int = 0
        self.cycle_len = self._updated_cycle_len = cycle_len
        self.cycle_mult = cycle_mult
        self._cycle_fact: float = 1.
        self._cycle_counter: int = 0
        self._training_size: int = 0
        self._batch_size: int = 0

        super(SGD, self).__init__(params, **kwargs)

    @staticmethod
    def cl_opts(parser: argparse.ArgumentParser, require=True) -> None:
        """Add command line options specific to SGD."""
        group = parser.add_argument_group("SGD options")
        group.add_argument(
            "--lr",
            type=float,
            default=0.01,
            help="""Learning rate. Default is 0.01. To add cosine annealing (with
            restarts) to the learning rate, use the cycle_len option as well.
            If you don't want resets, set cycle_len to the number of epochs."""
        )
        group.add_argument(
            "--lr-word-emb",
            type=float,
            help="""Learning rate for word embeddings, only. A sensible value
            would be 3-10x smaller than the default learning rate."""
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
            "--weight-decay",
            type=float,
            default=0.,
            help="""L2 penalty. Default is 0."""
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

    @classmethod
    def cl_init(cls, params: List[dict], opts: argparse.Namespace):
        """Initialize SGD from command line options."""
        if not opts.cycle_len and opts.cycle_mult:
            warn("Unused option: '--cycle_mult'. If you want to use a cyclic learning "
                 "rate schedule, you must specify the option '--cycle_len' as well.")
            sys.stderr.flush()

        cls.update_param_groups(params, opts)

        return cls(params,
                   cycle_len=opts.cycle_len,
                   cycle_mult=opts.cycle_mult or 1.,
                   momentum=opts.momentum,
                   nesterov=opts.nesterov,
                   weight_decay=opts.weight_decay)

    def _cyclic_decay(self, i: int) -> None:
        # pylint: disable=attribute-defined-outside-init
        """
        Perform cyclic LR decay.

        See http://arxiv.org/abs/1704.00109.
        """
        offset: float
        if self._training_size:
            offset = i / self._training_size
        else:
            offset = 0.

        # Update learning rates according to this monotonic function.
        # See Eq. (2) in Huang et al. 2017.
        new_lr = tuple([
            (x / 2) * \
            (
                np.cos(
                    np.pi *
                    ((self._cycle_counter + offset) % self._updated_cycle_len) /  # type: ignore
                    self._updated_cycle_len
                )
                + 1.
            ) for x in self.initial_lr
        ])
        self.lr = new_lr

    def iteration_update(self, i: int) -> None:
        """Update learning rate according to cyclic schedule."""
        if self.cycle_len:
            self._cyclic_decay(i)

    def epoch_update(self, loss: float) -> None:
        # pylint: disable=invalid-name,attribute-defined-outside-init
        """Update the learning rate."""
        self.epoch += 1
        if not self.cycle_len:
            return None

        self._cycle_counter += 1

        if self._cycle_counter % self._updated_cycle_len == 0:  # type: ignore
            # Adjust the cycle length.
            self._cycle_fact *= self.cycle_mult
            self._cycle_counter = 0
            self._updated_cycle_len = int(self._cycle_fact * self.cycle_len)

        print(f"Learning rate set to {self.lr}", flush=True)
        return None

    def epoch_prepare(self, training_size: int, batch_size: int) -> None:
        """Update training set size and batch size."""
        self._training_size = training_size
        self._batch_size = batch_size


OPTIM_ALIASES: Dict[str, Type[CLOptim]] = {
    "SGD": SGD,
    "Adam": Adam,
    "SparseAdam": SparseAdam,
    "AdaDelta": AdaDelta,
    "AdaGrad": AdaGrad,
    "RMSProp": RMSProp,
}
