"""Defines command-line options."""

import argparse

from .modules import LSTMCRF
from .optim import OPTIM_ALIASES


MODEL_ALIASES = {
    "lstm_crf": LSTMCRF,
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
        "-m", "--model",
        default="lstm_crf",
        type=str,
        choices=list(MODEL_ALIASES.keys()),
        help="""The model to use."""
    )
    group.add_argument(
        "--cuda",
        action="store_true",
        help="""Use a cuda device."""
    )


def train_opts(parser: argparse.ArgumentParser, require: bool = True) -> None:
    """Add options specific to a training task."""
    group = parser.add_argument_group("Training options")
    group.add_argument(
        "--train",
        type=str,
        required=require,
        nargs="+",
        help="""Path(s) to the training dataset(s)."""
    )
    group.add_argument(
        "--valid",
        type=str,
        nargs="+",
        default=[],
        help="""Path(s) to the validation dataset(s)."""
    )
    group.add_argument(
        "-o", "--out",
        type=str,
        required=require,
        help="""Path to the file where the trained model should be saved."""
    )
    group.add_argument(
        "--labels",
        type=str,
        nargs="+",
        required=require,
        help="""The target labels to use."""
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
