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

    def epoch_update(self):
        """Do any updating (such as lr update) after each epoch."""
        pass


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
        return None

    @classmethod
    def cl_init(cls, params: Iterable, opts: argparse.Namespace):
        """Initialize SGD from command line options."""
        return cls(params,
                   decay_rate=opts.decay_rate,
                   lr=opts.lr,
                   momentum=opts.momentum,
                   nesterov=opts.nesterov)

    def epoch_update(self):
        # pylint: disable=invalid-name,attribute-defined-outside-init
        """Update the learning rate."""
        self.epoch += 1
        if self.decay_rate > 0:
            self.lr = self.initial_lr / (1 + self.decay_rate * self.epoch)
            print(f"Updating learning rate to {self.lr}", flush=True)


OPTIM_ALIASES = {
    "SGD": SGD,
}
