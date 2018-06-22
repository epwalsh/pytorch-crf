"""Label a test dataset from the command line."""

import argparse
from typing import List

import torch

from .eval import ModelStats
from .io import Dataset
from .opts import label_opts, get_parser, parse_all, get_device


def label_data(opts: argparse.Namespace,
               model: torch.nn.Module,
               device: torch.device) -> None:
    """
    Label a dataset.

    Parameters
    ----------
    opts : argparse.Namespace
        The command-line options.

    model : torch.nn.Module
        The model to train.

    Returns
    -------
    None

    """
    model.eval()

    eval_stats = ModelStats(model.vocab.labels_itos)
    try:
        cursor = Dataset.read_file(opts.data, model.vocab, device=device)
        for src, tgt, raw_src, raw_tgt in cursor:
            labs = list(tgt.cpu().numpy())
            preds = model.predict(*src)[0][0]
            eval_stats.update(labs, preds)
            if opts.verbose:
                pred_labs = [model.vocab.labels_itos[x] for x in preds]
                padding = [max([len(x), len(y), len(z)])
                           for x, y, z in zip(raw_src, raw_tgt, pred_labs)]
                print("Tokens: ", end="")
                for index, token in enumerate(raw_src):
                    print(f"{token:{padding[index]}s}", end=" ")
                print("\nLabels: ", end="")
                for index, label in enumerate(raw_tgt):
                    print(f"{label:{padding[index]}s}", end=" ")
                print("\nPreds:  ", end="")
                for index, label in enumerate(pred_labs):
                    print(f"{label:{padding[index]}s}", end=" ")
                print("\n", flush=True)
    except KeyboardInterrupt:
        print("Exiting early...")

    print(eval_stats, flush=True)


def main(args: List[str] = None) -> None:
    """
    Parse command-line options and label dataset.

    Parameters
    ----------
    args : List[str], optional (default: None)
        A list of args to be passed to the ``ArgumentParser``. This is only
        used for testing.

    Returns
    -------
    None

    """
    initial_opts, parser = get_parser(args, label_opts)
    opts = parse_all(args, initial_opts, parser)
    if not opts:
        return

    # Get the device to train on.
    device = get_device(opts)

    # Load model.
    print(f"Loading model from {opts.model}", flush=True)
    model = torch.load(opts.model, map_location=lambda storage, loc: storage).to(device)

    # Label the dataset.
    label_data(opts, model, device)


if __name__ == "__main__":
    main()
