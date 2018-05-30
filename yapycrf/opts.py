"""Defines command-line options."""

import argparse


def help_opts(parser: argparse.ArgumentParser) -> None:
    """Help options."""
    group = parser.add_argument_group("Reference options")
    group.add_argument(
        "-h", "--help",
        action="store_true",
        help="""Display this help message and exit."""
    )


def base_opts(parser: argparse.ArgumentParser) -> None:
    """
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
        default="bilstm_crf",
        type=str,
        choices=["bilstm_crf"],
        help="""The model to use."""
    )


def train_opts(parser: argparse.ArgumentParser, require: bool = True) -> None:
    """Options specific to a training task."""
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
