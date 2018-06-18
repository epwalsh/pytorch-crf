"""Defines command-line options."""

import argparse
from typing import Type, Dict

import torch

from .modules import CharLSTM, CharCNN
from .optim import OPTIM_ALIASES


MODEL_ALIASES: Dict[str, Type[torch.nn.Module]] = {
    "lstm": CharLSTM,
    "cnn": CharCNN,
}


def help_opts(parser: argparse.ArgumentParser) -> None:
    """Add help options."""
    group = parser.add_argument_group("Reference options")
    group.add_argument(
        "-h", "--help",
        action="store_true",
        help="""Display this help message and exit."""
    )


def base_opts(parser: argparse.ArgumentParser) -> None:
    """
    Add base command-line options.

    These options are are available regardless of task or the model being
    trained.
    """
    group = parser.add_argument_group("Base options")
    group.add_argument(
        "--verbose",
        action="store_true",
        help="""Control the amount of output printed to the terminal."""
    )
    group.add_argument(
        "--cuda",
        action="store_true",
        help="""Use a cuda device."""
    )
    group.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="""GPU device ID to use."""
    )


def train_opts(parser: argparse.ArgumentParser, require: bool = True) -> None:
    """Add options specific to a training task."""
    group = parser.add_argument_group("Training options")
    group.add_argument(
        "--char-features",
        default="lstm",
        type=str,
        choices=list(MODEL_ALIASES.keys()),
        help="""The character-level feature generation layer to use."""
    )
    group.add_argument(
        "--train",
        type=str,
        required=require,
        nargs="+",
        help="""Path(s) to the training dataset(s)."""
    )
    group.add_argument(
        "--validation",
        type=str,
        nargs="+",
        default=[],
        help="""Path(s) to the validation dataset(s)."""
    )
    group.add_argument(
        "-o", "--out",
        type=str,
        required=False,
        help="""Path to the file where the trained model should be saved."""
    )
    group.add_argument(
        "--vectors",
        type=str,
        help="""Path to vector cache."""
    )
    group.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="""The maximum number of epochs to train for."""
    )
    group.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="""Log progress after processing this many batches."""
    )
    group.add_argument(
        "--optim",
        type=str,
        default="SGD",
        choices=list(OPTIM_ALIASES.keys()),
        help="""The optimizer to use."""
    )
    group.add_argument(
        "--max-grad",
        type=float,
        help="""Clip gradient components that exceed this in absolute value."""
    )
    group.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="""Mini batch size. Default is 1."""
    )
    group.add_argument(
        "--results",
        type=str,
        help="""YAML file to append results to."""
    )
    group.add_argument(
        "--dropout",
        type=float,
        default=0.,
        help="""Dropout probability."""
    )
