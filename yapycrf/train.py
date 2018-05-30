"""Command-line interface for training a model."""

import argparse
from typing import List

from .opts import help_opts, base_opts, train_opts


def main(args: List[str] = None) -> None:
    """
    Function to be run when script is invoked.

    Parameters
    ----------
    args : List[str], optional (default: None)
        A list of args to be passed to the ``ArgumentParser``. This is only
        used for testing.

    Returns
    -------
    None

    """
    parser = argparse.ArgumentParser(add_help=False)
    help_opts(parser)

    # Parse initial option to check for 'help' flag.
    initial_opts, _ = parser.parse_known_args(args=args)

    # Add base options and parse again.
    base_opts(parser)
    train_opts(parser, require=not initial_opts.help)
    initial_opts, _ = parser.parse_known_args(args=args)

    # Check if we should display the help message and exit.
    if initial_opts.help:
        parser.print_help()


if __name__ == "__main__":
    main()
