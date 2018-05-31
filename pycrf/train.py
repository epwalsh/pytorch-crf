"""
Train models from the command line.
-----------------------------------

The training script itself is invoked with:

.. code-block:: bash

  python -m pycrf.train

You can see all of the options available for training a specific model with:

.. code-block:: bash

  python -m pycrf.train --help --model MODEL_NAME

"""

import argparse
from typing import List

from .io import Vocab, Dataset
from .opts import help_opts, base_opts, train_opts, MODEL_ALIASES


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

    # Add model-specific options.
    model_class = MODEL_ALIASES[initial_opts.model]
    model_class.cl_opts(parser)

    # Check if we should display the help message and exit.
    if initial_opts.help:
        parser.print_help()
        return

    # Parse the args again.
    opts = parser.parse_args(args=args)

    # Initialize the vocab and dataset.
    vocab = Vocab(opts.labels, cache=opts.vectors)
    dataset = Dataset()
    for fname in opts.train:
        dataset.load_file(fname, vocab)
    print("Loaded {:d} sentences for training".format(len(dataset)),
          flush=True)

    # Initialize the model.
    model = model_class.cl_init(opts, vocab)
    print(model, flush=True)


if __name__ == "__main__":
    main()
