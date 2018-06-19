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
import torchtext

from .eval import ModelStats
from .io import Vocab, Dataset
from .logging import Logger
from .modules import LSTMCRF
from .optim import OPTIM_ALIASES
from .opts import help_opts, base_opts, train_opts, MODEL_ALIASES


def train(opts: argparse.Namespace,
          model: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          dataset_train: Dataset,
          dataset_valid: Dataset = None) -> None:
    """
    Train a model on the given dataset.

    Parameters
    ----------
    opts : argparse.Namespace
        The command-line options.

    model : torch.nn.Module
        The model to train.

    optimizer : torch.optim.Optimizer
        The optimizer to train with.

    dataset_train : Dataset
        The training dataset.

    dataset_valid : Dataset, optional
        A validation dataset which will be evaluated at the end of each epoch.

    Returns
    -------
    None

    """
    # Initialize logger.
    n_examples = len(dataset_train)
    logger = Logger(n_examples,
                    log_interval=opts.log_interval,
                    verbose=opts.verbose,
                    results_file=opts.results,
                    log_dir=opts.log_dir)

    # ==========================================================================
    # Loop through epochs.
    # ==========================================================================

    for epoch in range(opts.epochs):
        logger.start_epoch()

        # Shuffle dataset.
        dataset_train.shuffle()

        # Set model to train mode.
        model.train()

        # ======================================================================
        # Loop through all mini-batches.
        # ======================================================================

        iteration = 0
        while iteration < len(dataset_train):
            # Zero out the gradient.
            model.zero_grad()

            # ==================================================================
            # Loop through sentences in mini-batch.
            # ==================================================================

            batch_loss: torch.Tensor = 0.
            for _ in range(min([opts.batch_size, n_examples - iteration])):
                src, tgt = dataset_train[iteration]

                # Compute the loss.
                loss = model(*src, tgt)
                batch_loss += loss

                logger.update(iteration, loss, model.named_parameters())
                iteration += 1

            # ==================================================================
            # End mini-batch.
            # ==================================================================

            # Compute the gradient.
            batch_loss.backward()

            # Clip gradients.
            if opts.max_grad is not None:
                torch.nn.utils.clip_grad_value_(
                    model.parameters(), opts.max_grad)

            # Take a step.
            optimizer.step()

        # ======================================================================
        # End all mini-batches.
        # ======================================================================

        # Log the loss and duration of the epoch.
        logger.end_epoch()

        # Update optimizer.
        optimizer.epoch_update(logger.epoch_loss)

        # Gather eval stats.
        eval_stats = ModelStats(model.vocab.labels_itos, epoch)

        # Evaluate on validation set.
        if dataset_valid:
            # Put model into evaluation mode.
            model.eval()
            for src, tgt in dataset_valid:
                labs = list(tgt.cpu().numpy())
                preds = model.predict(*src)[0][0]
                eval_stats.update(labs, preds)

        logger.append_eval_stats(eval_stats, validation=bool(dataset_valid))

    # ==========================================================================
    # End epochs.
    # ==========================================================================

    # Log results.
    logger.end_train(validation=bool(dataset_valid))


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

    # Add optimizer-specific options.
    optim_class = OPTIM_ALIASES[initial_opts.optim]
    optim_class.cl_opts(parser)

    # Add model-specific options.
    LSTMCRF.cl_opts(parser)
    char_feats_class = MODEL_ALIASES[initial_opts.char_features]
    char_feats_class.cl_opts(parser)

    # Check if we should display the help message and exit.
    if initial_opts.help:
        parser.print_help()
        return

    # Parse the args again.
    opts = parser.parse_args(args=args)

    # Set the device to train on.
    if opts.cuda:
        device = torch.device("cuda", opts.gpu_id)
    else:
        if torch.cuda.is_available():
            print("Warning: CUDA is available, but you have not used the --cuda flag")
        device = torch.device("cpu")

    # Load pretrained word embeddings and initialize vocab.
    glove = torchtext.vocab.GloVe(name="6B", dim=opts.word_vec_dim, cache=opts.vectors)
    vocab = Vocab(glove.stoi, glove.itos)

    # Load datasets.
    print("Loading datasets", flush=True)
    dataset_train = Dataset()
    dataset_valid = Dataset()
    for fname in opts.train:
        dataset_train.load_file(fname, vocab, device=device)
    print("Loaded {:d} sentences for training".format(len(dataset_train)),
          flush=True)
    if opts.validation:
        for fname in opts.validation:
            dataset_valid.load_file(fname, vocab, device=device)
        print("Loaded {:d} sentences for validation"
              .format(len(dataset_valid)))
    print("Training on labels:", list(vocab.labels_stoi.keys()))

    # Initialize the model.
    char_feats_layer = char_feats_class.cl_init(opts, vocab).to(device)
    model = LSTMCRF.cl_init(opts, vocab, char_feats_layer, glove.vectors).to(device)
    print(model, flush=True)

    # Initialize the optimizer.
    optimizer = optim_class.cl_init(
        filter(lambda p: p.requires_grad, model.parameters()), opts)

    # Train model.
    try:
        train(opts, model, optimizer, dataset_train, dataset_valid)
    except KeyboardInterrupt:
        print("Exiting early")


if __name__ == "__main__":
    main()
