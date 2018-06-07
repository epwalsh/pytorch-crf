"""Defines command line interface to torch optimizer classes."""

from abc import ABC, abstractstaticmethod, abstractclassmethod
import argparse
from typing import Iterable

import torch


class CLOptim(ABC):
    """Command line optimizer interface."""

    @staticmethod
    @abstractstaticmethod
    def cl_opts(parser: argparse.ArgumentParser) -> None:
        """Add command line options specific to this optimizer."""
        pass

    @classmethod
    @abstractclassmethod
    def cl_init(cls, params: Iterable, opts: argparse.Namespace):
        """Initialize optimizer from command line options."""
        pass

    def epoch_update(self) -> None:
        """Do any updating (such as lr update) after each epoch."""
        pass


class AdaGrad(torch.optim.Adagrad, CLOptim):
    """Wraps torch AdaGrad optimizer."""

    @staticmethod
    def cl_opts(parser: argparse.ArgumentParser) -> None:
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
    def cl_opts(parser: argparse.ArgumentParser) -> None:
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
    def cl_opts(parser: argparse.ArgumentParser) -> None:
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
    def cl_opts(parser: argparse.ArgumentParser) -> None:
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


class SGD(torch.optim.SGD, CLOptim):
    """Wraps torch stochastic gradient descent optimizer."""

    def __init__(self,
                 params: Iterable,
                 decay_rate: float = 0.,
                 **kwargs) -> None:
        super(SGD, self).__init__(params, **kwargs)
        self.decay_rate = decay_rate
        self.initial_lr: float = kwargs["lr"]
        self.epoch = 0

    @staticmethod
    def cl_opts(parser: argparse.ArgumentParser) -> None:
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
        return None

    @classmethod
    def cl_init(cls, params: Iterable, opts: argparse.Namespace):
        """Initialize SGD from command line options."""
        return cls(params,
                   decay_rate=opts.decay_rate,
                   lr=opts.lr,
                   momentum=opts.momentum,
                   nesterov=opts.nesterov,
                   weight_decay=opts.weight_decay)

    def epoch_update(self) -> None:
        # pylint: disable=invalid-name,attribute-defined-outside-init
        """Update the learning rate."""
        self.epoch += 1
        if self.decay_rate > 0:
            self.lr = self.initial_lr / (1 + self.decay_rate * self.epoch)
            print(f"Updating learning rate to {self.lr}", flush=True)


OPTIM_ALIASES = {
    "SGD": SGD,
    "Adam": Adam,
    "AdaDelta": AdaDelta,
    "AdaGrad": AdaGrad,
    "RMSProp": RMSProp,
}
