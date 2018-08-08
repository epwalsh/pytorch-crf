# pylint: disable=too-many-statements,too-many-branches

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

import sys
from typing import List, Tuple, Generator

import torch
import numpy as np

from .eval import ModelStats
from .exceptions import LearnerInitializationError, ArgParsingError
from .io import Vocab, Dataset
from .io.vectors import load_pretrained
from .logging import TrainLogger, LRFinderLogger
from .modules import LSTMCRF
from .optim import CLOptim, OPTIM_ALIASES, SGD
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
    print(f"Saving checkpoint to {checkpoint_path}", flush=True)
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
    print(f"Saving best model to {model_path}", flush=True)
    load_checkpoint(model, path, epoch)
    torch.save(model, model_path)
    print("Done!", flush=True)


class Learner:
    """Serves as an interface to the command-line interface."""

    def __init__(self, args: List[str] = None) -> None:
        initial_opts, parser = get_parser(args, train_opts)

        # Add optimizer-specific options.
        optim_class = OPTIM_ALIASES[initial_opts.optim]
        optim_class.cl_opts(parser, require=not initial_opts.help)

        # Add model-specific options.
        LSTMCRF.cl_opts(parser, require=not initial_opts.help)
        char_feats_class = MODEL_ALIASES[initial_opts.char_features]
        char_feats_class.cl_opts(parser, require=not initial_opts.help)

        opts = parse_all(args, initial_opts, parser)

        # Ensure we were given a path to a training object or training text files
        # and pretrained word vectors text file.
        if not opts.train_object:
            missing = []
            if not opts.train:
                missing.append("--train (or --train-object)")
            if not opts.word_vectors:
                missing.append("--word-vectors")
            if missing:
                err_msg = f"the following arguments are required: {', '.join(missing)}"
                raise ArgParsingError(err_msg)

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

        self._opts = opts
        self.vocab = vocab
        self.pretrained_word_vecs = pretrained_vecs
        self.model = model
        self.char_feats_class = char_feats_class
        self.optimizer_class = optim_class
        self.dataset_train = dataset_train
        self.dataset_valid = dataset_valid

    def reset_model(self) -> None:
        """Initialize model again to completely reset params."""
        device = get_device(self._opts)
        char_feats_layer = self.char_feats_class.cl_init(self._opts, self.vocab).to(device)
        self.model = LSTMCRF.cl_init(self._opts, self.vocab, char_feats_layer,
                                     self.pretrained_word_vecs).to(device)

    def save_train_state(self, path: str) -> None:
        """Save training set state object."""
        train_object = {
            "vocab": self.vocab,
            "word_vectors": self.pretrained_word_vecs,
            "train": self.dataset_train,
            "validation": self.dataset_valid,
        }
        torch.save(train_object, path)

    @classmethod
    def build(cls, **options):
        """Initialize a learner object."""
        args: List[str] = []
        for opt_name, opt_value in options.items():
            opt_name = "--" + opt_name.replace("_", "-")
            args.append(opt_name)
            if isinstance(opt_value, (list, tuple)):
                args.extend(opt_value)
            elif not isinstance(opt_value, bool):
                args.append(str(opt_value))

        try:
            return cls(args=args)
        except ArgParsingError as exc:
            raise LearnerInitializationError(
                missing_args=exc.missing_args,
                unknown_args=exc.unknown_args)

    def __getattr__(self, name):
        return getattr(self._opts, name)

    def __setattr__(self, name, value):
        if name in self.__dict__ or \
                "_opts" not in self.__dict__ or \
                name not in self._opts.__dict__:
            super().__setattr__(name, value)
        else:
            self._opts.__dict__[name] = value

    def fit_epoch(self,
                  optimizer: CLOptim,
                  dataset: Dataset,
                  epoch: int = 0,
                  logger: TrainLogger = None,
                  n: int = None) -> Generator[float, None, None]:
        """Train for one epoch."""
        # Prepare optimizer for epoch.
        optimizer.epoch_prepare(len(dataset), self._opts.batch_size)

        # Set model to train mode.
        self.model.train()

        # ==================================================================
        # Loop through all mini-batches.
        # ==================================================================

        n_examples = n or len(dataset)
        example_num = 0
        while example_num < n_examples:
            # Zero out the gradient.
            self.model.zero_grad()

            # ==============================================================
            # Loop through sentences in mini-batch.
            # ==============================================================

            batch_loss: torch.Tensor = 0.
            for _ in range(min([self._opts.batch_size, n_examples - example_num])):
                src, tgt = dataset[example_num]

                # Compute the loss.
                loss = self.model(*src, tgt)
                batch_loss += loss

                if logger:
                    logger.update(epoch, example_num, loss,
                                  self.model.named_parameters(),
                                  optimizer.lr)
                example_num += 1

            # Compute the gradient.
            batch_loss.backward()

            # Clip gradients.
            if self._opts.max_grad is not None:
                torch.nn.utils.clip_grad_value_(
                    self.model.parameters(), self._opts.max_grad)

            # Take a step.
            optimizer.step()

            # Update the optimizer.
            optimizer.iteration_update(example_num)

            yield batch_loss.item()

    def fit(self) -> None:
        """Train model."""
        # Initialize the optimizer.
        optimizer = self.optimizer_class.cl_init(self.model.get_trainable_params(), self._opts)

        # Initialize logger.
        logger = TrainLogger(len(self.dataset_train),
                             log_interval=self._opts.log_interval,
                             verbose=self._opts.verbose,
                             results_file=self._opts.results,
                             log_dir=self._opts.log_dir)

        # Load checkpoint if given.
        if self._opts.checkpoint:
            checkpoint = torch.load(self._opts.checkpoint)
            start_epoch = checkpoint["epoch"]
            self.model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(f"Loaded model checkpoint {self._opts.checkpoint}")
        else:
            start_epoch = 0

        # ==========================================================================
        # Loop through epochs.
        # ==========================================================================

        for epoch in range(start_epoch, self._opts.epochs):
            try:
                logger.start_epoch(epoch)

                # Record learning rate and other metrics.
                metrics = {f"lr/group{i}": x for i, x in enumerate(optimizer.lr)}
                logger.record(metrics, epoch + 1)

                # Shuffle dataset.
                self.dataset_train.shuffle()

                # Fit to training set for one epoch.
                for _ in self.fit_epoch(optimizer, self.dataset_train, logger=logger, epoch=epoch):
                    pass

                # Log the loss and duration of the epoch.
                logger.end_epoch()

                # Update optimizer.
                optimizer.epoch_update(logger.epoch_loss)

                # Gather eval stats.
                eval_stats = ModelStats(self.model.vocab.labels_itos, epoch=epoch)

                # Evaluate on validation set.
                if self.dataset_valid:
                    # Put model into evaluation mode.
                    self.model.eval()
                    for src, tgt in self.dataset_valid:
                        labs = list(tgt.cpu().numpy())
                        preds = self.model.predict(*src)[0][0]
                        eval_stats.update(labs, preds)

                # Record loss and validation set metrics.
                logger.append_eval_stats(eval_stats, validation=bool(self.dataset_valid))

            except KeyboardInterrupt:
                print("Exiting early...")
                break
            finally:
                # Save checkpoint.
                if self._opts.out:
                    save_checkpoint(self.model, optimizer, self._opts.out, epoch)

        # ==========================================================================
        # End epochs.
        # ==========================================================================

        # Log results.
        best_epoch = logger.end_train(validation=bool(self.dataset_valid))

        # Save best model.
        if self._opts.out:
            save_model(self.model, self._opts.out, best_epoch)

    def find_lr(self,
                bounds: Tuple[float, float] = (1e-5, 1.),
                iterations: int = 100,
                smoothing: float = 0.) -> Tuple[np.array, np.array]:
        """
        Find best learning rate.

        Parameters
        ----------
        bounds : Tuple[float, float]
            The minimum and maximum learning to try.

        Returns
        -------
        Tuple[np.array, np.array]
            An array of learnings rates and an array of corresponding values
            of the loss on the validation set.

        """
        assert iterations * self._opts.batch_size <= len(self.dataset_train)
        assert smoothing < 1.

        # Construst sequence of learning rates to test.
        min_lr, max_lr = bounds
        start_seq = np.log(min_lr)
        stop_seq = np.log(max_lr)
        lrs = np.exp(np.arange(start_seq, stop_seq, step=0.1))

        logger = LRFinderLogger(len(lrs))

        # Only going to fit the last layer for tuning the lr.
        params = self.model.get_trainable_params()[-1:]

        losses: List[float] = []
        for lr in lrs:
            params[0]["lr"] = lr
            optimizer = SGD(params)

            # Shuffle dataset.
            self.dataset_train.shuffle()

            # Fit for specifiec number of iterations.
            loss: float = 0.
            for batch_loss in self.fit_epoch(optimizer, self.dataset_train,
                                             n=(iterations * self._opts.batch_size)):
                loss += batch_loss

            # Evaluate validation set.
            losses.append(loss)

            # Log progress.
            logger.update(lr, loss)

        # Apply exponential smoothing.
        if smoothing:
            for i in range(1, len(losses)):
                losses[i] = (1 - smoothing) * losses[i] + smoothing * losses[i-1]

        return lrs, np.array(losses)


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
    try:
        learner = Learner(args=args)
        learner.fit()
    except ArgParsingError as exc:
        if exc.message:
            sys.exit(f"Error: {exc.message}")


if __name__ == "__main__":
    main()
