"""Train logging."""

import time
from typing import List

import numpy as np
import torch
import tensorflow as tf

from .eval import ModelStats


class Logger:
    """
    Logs progress during training.

    Parameters
    ----------
    n_examples : int
        The number of training examples.

    log_interval : int
        The number of iterations before logging progress.

    verbose : int
        Controls the verbosity and frequency of logs.

    results_file : str
        File to append final results to.

    """

    def __init__(self,
                 n_examples: int,
                 log_interval: int = 100,
                 verbose: bool = True,
                 log_dir: str = None,
                 results_file: str = None) -> None:
        self.n_examples = n_examples
        self.log_interval = log_interval
        self.verbose = verbose
        self.results_file = results_file
        self.eval_stats: List[ModelStats] = []

        self.train_start_time: float = time.time()
        self.epoch_start_time: float = None
        self.running_time: float = None

        self.epoch_loss = 0.
        self.running_loss = 0.

        if log_dir is not None:
            self.writer = tf.summary.FileWriter(log_dir)
        else:
            self.writer = None

    @property
    def current_epoch(self):
        """Return current epoch number."""
        return len(self.eval_stats)

    def scalar_summary(self, tag: str, value: float, step: int) -> None:
        """Log a scalar variable."""
        if self.writer is None:
            return None
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)
        return None

    def histo_summary(self,
                      tag: str,
                      values: np.ndarray,
                      step: int,
                      bins: int = 1000) -> None:
        # pylint: disable=no-member
        """Log a histogram of the tensor of values."""
        if self.writer is None:
            return None

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for count in counts:
            hist.bucket.append(count)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()

        return None

    def start_epoch(self) -> None:
        """Log start of epoch."""
        print("\nEpoch {:d}".format(self.current_epoch + 1))
        print("==================================================", flush=True)
        self.epoch_loss = 0.
        self.running_loss = 0.
        self.epoch_start_time = self.running_time = time.time()

    def end_epoch(self) -> None:
        """Log end of epoch."""
        epoch_duration = time.time() - self.epoch_start_time
        print("Loss: {:f}, duration: {:.0f} seconds"
              .format(self.epoch_loss, epoch_duration), flush=True)

    def end_train(self, validation: bool = False) -> None:
        """End round of training."""
        train_duration = time.time() - self.train_start_time
        print(f"Total training time: {train_duration:.0f} seconds\n", flush=True)

        if validation:
            best_epoch = max(self.eval_stats, key=lambda x: x.score)
        else:
            best_epoch = min(self.eval_stats, key=lambda x: x.loss)

        f1_score, precision, recall, accuracy = best_epoch.score
        loss = best_epoch.loss

        print(f"Best epoch: {best_epoch.epoch+1:d}\n------------------", flush=True)
        if validation:
            print(f"f1:         {f1_score:.4f}\n"
                  f"precision:  {precision:.4f}\n"
                  f"recall:     {recall:.4f}\n"
                  f"accuracy:   {accuracy:.4f}", flush=True)
        print(f"loss:       {loss:.4f}")

        if self.results_file:
            with open(self.results_file, "a") as resultsfile:
                resultsfile.write("results:\n")
                resultsfile.write(f"  best_epoch: {best_epoch.epoch+1}\n")
                if validation:
                    resultsfile.write(f"  micro_avg_f1: {f1_score:f}\n")
                    resultsfile.write(f"  micro_avg_precision: {precision:f}\n")
                    resultsfile.write(f"  micro_avg_recall: {recall:f}\n")
                resultsfile.write(f"  loss: {best_epoch.loss:f}\n")

    def append_eval_stats(self,
                          eval_stats: ModelStats,
                          validation: bool = False) -> None:
        """Add another set of evaluation metrics."""
        info = {"loss": self.epoch_loss}
        if validation:
            print(eval_stats, flush=True)
            score = eval_stats.score
            info["f1"] = score[0]
            info["precision"] = score[1]
            info["recall"] = score[2]
            info["accuracy"] = score[3]
        eval_stats.loss = self.epoch_loss
        self.eval_stats.append(eval_stats)
        if self.writer is not None:
            for tag, value in info.items():
                self.scalar_summary(tag, value, self.current_epoch)

    def update(self, iteration: int, loss: torch.Tensor, params: dict):
        """Update loss."""
        self.epoch_loss += loss.item()
        self.running_loss += loss.item()

        if self.verbose and (iteration + 1) % self.log_interval == 0:
            progress = 100 * (iteration + 1) / self.n_examples
            duration = time.time() - self.running_time
            print("[{:6.2f}%] loss: {:10.5f}, duration: {:.2f} seconds"
                  .format(progress, self.running_loss, duration), flush=True)
            self.running_loss = 0.
            self.running_time = time.time()

            if self.writer is not None:
                step = self.current_epoch * self.n_examples + iteration + 1
                for tag, value in params:
                    if not value.requires_grad:
                        continue
                    tag = tag.replace(".", "/")
                    self.histo_summary(tag, value.data.cpu().numpy(), step)
