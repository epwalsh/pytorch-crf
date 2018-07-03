"""
Train models from the command line.

The training script itself is invoked with:

.. code-block:: bash

  python -m pycrf.train

You can see all of the options available for a specific character-level
feature model or optimizer with:

.. code-block:: bash

  python -m pycrf.train --help --char-features MODEL_NAME --optim OPTIM_NAME

"""

import argparse
from typing import List

import torch

from .eval import ModelStats
from .io import Vocab, Dataset
from .io.vectors import load_pretrained
from .logging import Logger
from .modules import LSTMCRF
from .optim import OPTIM_ALIASES
from .opts import train_opts, MODEL_ALIASES, get_parser, parse_all, get_device
from .utils import _parse_data_path


def _get_checkpoint_path(path: str, epoch: int) -> str:
    return f"{path}_epoch_{epoch+1:03d}.pt"


def save_checkpoint(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    path: str,
                    epoch: int) -> None:
    """Save a training checkpoint."""
    checkpoint_path = _get_checkpoint_path(path, epoch)
    print(f"Saving checkpoint to {checkpoint_path}")
    torch.save({
        "epoch": epoch + 1,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, checkpoint_path)


def load_checkpoint(model: torch.nn.Module,
                    path: str,
                    epoch: int,
                    optimizer: torch.optim.Optimizer = None) -> None:
    """Load model weights from checkpoint."""
    checkpoint_path = _get_checkpoint_path(path, epoch)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])


def save_model(model: torch.nn.Module, path: str, epoch: int) -> None:
    """Save the model with best epoch weights."""
    model_path = f"{path}.pt"
    print(f"Saving best model to {model_path}")
    load_checkpoint(model, path, epoch)
    torch.save(model, model_path)


def train(opts: argparse.Namespace,
          model: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          dataset_train: Dataset,
          dataset_valid: Dataset = None) -> None:
    # pylint: disable=too-many-branches
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

    # Load checkpoint if given.
    if opts.checkpoint:
        checkpoint = torch.load(opts.checkpoint)
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print(f"Loaded model checkpoint {opts.checkpoint}")
    else:
        start_epoch = 0

    # ==========================================================================
    # Loop through epochs.
    # ==========================================================================

    for epoch in range(start_epoch, opts.epochs):
        try:
            logger.start_epoch(epoch)

            # Shuffle dataset.
            dataset_train.shuffle()

            # Set model to train mode.
            model.train()

            # ==================================================================
            # Loop through all mini-batches.
            # ==================================================================

            iteration = 0
            while iteration < len(dataset_train):
                # Zero out the gradient.
                model.zero_grad()

                # ==============================================================
                # Loop through sentences in mini-batch.
                # ==============================================================

                batch_loss: torch.Tensor = 0.
                for _ in range(min([opts.batch_size, n_examples - iteration])):
                    src, tgt = dataset_train[iteration]

                    # Compute the loss.
                    loss = model(*src, tgt)
                    batch_loss += loss

                    logger.update(epoch, iteration, loss, model.named_parameters())
                    iteration += 1

                # ==============================================================
                # End mini-batch.
                # ==============================================================

                # Compute the gradient.
                batch_loss.backward()

                # Clip gradients.
                if opts.max_grad is not None:
                    torch.nn.utils.clip_grad_value_(
                        model.parameters(), opts.max_grad)

                # Take a step.
                optimizer.step()

            # ==================================================================
            # End all mini-batches.
            # ==================================================================

            # Log the loss and duration of the epoch.
            logger.end_epoch()

            # Update optimizer.
            optimizer.epoch_update(logger.epoch_loss)

            # Gather eval stats.
            eval_stats = ModelStats(model.vocab.labels_itos, epoch=epoch)

            # Evaluate on validation set.
            if dataset_valid:
                # Put model into evaluation mode.
                model.eval()
                for src, tgt in dataset_valid:
                    labs = list(tgt.cpu().numpy())
                    preds = model.predict(*src)[0][0]
                    eval_stats.update(labs, preds)

            logger.append_eval_stats(eval_stats, validation=bool(dataset_valid))

        except KeyboardInterrupt:
            print("Exiting early...")
            break
        finally:
            # Save checkpoint.
            if opts.out:
                save_checkpoint(model, optimizer, opts.out, epoch)

    # ==========================================================================
    # End epochs.
    # ==========================================================================

    # Log results.
    best_epoch = logger.end_train(validation=bool(dataset_valid))

    # Save best model.
    if opts.out:
        save_model(model, opts.out, best_epoch)


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
    initial_opts, parser = get_parser(args, train_opts)

    # Add optimizer-specific options.
    optim_class = OPTIM_ALIASES[initial_opts.optim]
    optim_class.cl_opts(parser, require=not initial_opts.help)

    # Add model-specific options.
    LSTMCRF.cl_opts(parser, require=not initial_opts.help)
    char_feats_class = MODEL_ALIASES[initial_opts.char_features]
    char_feats_class.cl_opts(parser, require=not initial_opts.help)

    opts = parse_all(args, initial_opts, parser)
    if not opts:
        return

    # Ensure we were given a path to a training object or training text files
    # and pretrained word vectors text file.
    if not opts.train_object and not opts.train and not opts.word_vectors:
        missing = "train" if not opts.train else "word-vectors"
        parser.print_usage()
        print(f"train.py: error: missing required argument --{missing}", flush=True)
        return

    # Get the device to train on.
    device = get_device(opts)

    dataset_train = Dataset()
    dataset_valid = Dataset(is_test=True)
    vocab: Vocab
    pretrained_vecs: torch.Tensor

    if opts.train_object:
        print(f"Loading training state object from {opts.train_object}", flush=True)
        train_object = torch.load(opts.train_object)
        vocab = train_object["vocab"]
        pretrained_vecs = train_object["word_vectors"]
        dataset_train = train_object["train"]
        dataset_valid = train_object["validation"]
    else:
        # Load pretrained word embeddings and initialize vocab.
        print(f"Loading pretrained word vectors from {opts.word_vectors}", flush=True)
        pretrained_vecs, terms_itos, terms_stoi = load_pretrained(opts.word_vectors)
        vocab = Vocab(terms_stoi, terms_itos, default_context=opts.default_context)

        # Load datasets.
        print("Loading datasets", flush=True)
        for fname in opts.train:
            context, path = _parse_data_path(fname)
            dataset_train.load_file(path, vocab, device=device, sent_context=context)

        print(f"Loaded {len(dataset_train):d} sentences for training", flush=True)

        if opts.validation:
            for fname in opts.validation:
                context, path = _parse_data_path(fname)
                dataset_valid.load_file(path, vocab, device=device, sent_context=context)

            print(f"Loaded {len(dataset_valid):d} sentences for validation", flush=True)

    print("Found the following labels:", list(vocab.labels_stoi.keys()))
    if len(vocab.sent_context_stoi) > 1:
        print("Found the following sentence contexts:", list(vocab.sent_context_stoi))

    # Initialize the model.
    char_feats_layer = char_feats_class.cl_init(opts, vocab).to(device)
    model = LSTMCRF.cl_init(opts, vocab, char_feats_layer, pretrained_vecs).to(device)
    print(model, flush=True)

    # Initialize the optimizer.
    optimizer = optim_class.cl_init(filter(lambda p: p.requires_grad, model.parameters()), opts)

    # Train model.
    train(opts, model, optimizer, dataset_train, dataset_valid)


if __name__ == "__main__":
    main()
