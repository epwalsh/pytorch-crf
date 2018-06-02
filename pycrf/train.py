"""
Train models from the command line.

The training script itself is invoked with:

.. code-block:: bash

  python -m pycrf.train

You can see all of the options available for training a specific model with:

.. code-block:: bash

  python -m pycrf.train --help --model MODEL_NAME

"""

import argparse
from typing import List

import torch

from .io import Vocab, Dataset
from .opts import help_opts, base_opts, train_opts, MODEL_ALIASES


def train(opts: argparse.Namespace,
          model: torch.nn.Module,
          dataset_train: Dataset,
          dataset_valid: Dataset = None) -> None:
    """Train a model on the given dataset."""
    # Initialize optimizer.
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)

    # Loop through epochs.
    for epoch in range(opts.epochs):
        print("Epoch {:d}".format(epoch + 1))
        i = 0
        total_loss = 0.
        running_loss = 0.
        for src, tgt in dataset_train:
            # Zero out the gradient.
            model.zero_grad()

            # Compute the loss.
            loss = model(*src, tgt)
            total_loss += loss
            running_loss += loss

            # Compute the gradient.
            loss.backward()

            # Take a step.
            optimizer.step()

            if (i + 1) % opts.log_interval == 0:
                loss_report = running_loss / opts.log_interval
                progress = 100 * (i + 1) / len(dataset_train)
                print("[{:5.2f}%] {:f}"
                      .format(progress, loss_report))
                running_loss = 0

            i += 1

        print("Loss: {:f}".format(total_loss))

        # Evaluate on validation set.
        if dataset_valid:
            pass


def main(args: List[str] = None) -> None:
    """
    Parse command-line options and train model.

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
    device = torch.device("cuda" if opts.cuda else "cpu")

    # Initialize the vocab and datasets.
    print("Loading datasets")
    vocab = Vocab(opts.labels, cache=opts.vectors)
    dataset_train = Dataset()
    dataset_valid = Dataset()
    for fname in opts.train:
        dataset_train.load_file(fname, vocab, device=device)
    print("Loaded {:d} sentences for training".format(len(dataset_train)),
          flush=True)
    if opts.valid:
        for fname in opts.valid:
            dataset_valid.load_file(fname, vocab, device=device)
        print("Loaded {:d} sentences for validation"
              .format(len(dataset_valid)))

    # Initialize the model.
    model = model_class.cl_init(opts, vocab).to(device)
    print(model, flush=True)

    # Train model.
    train(opts, model, dataset_train, dataset_valid)


if __name__ == "__main__":
    main()
