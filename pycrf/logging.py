"""Train logging."""

import time
import datetime
from typing import List, Dict, Tuple

import numpy as np
import torch
import tensorflow as tf
from tqdm import tqdm, tqdm_notebook

from .eval import ModelStats
from .utils import in_ipynb


def _format_duration(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))


class TrainLogger:
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
                 results_file: str = None,
                 log_weights: bool = False) -> None:
        self.n_examples = n_examples
        self.log_interval = log_interval
        self.verbose = verbose
        self.results_file = results_file
        self.log_weights = log_weights
        self.pbar: tqdm = None

        self.eval_stats: List[ModelStats] = []

        self.train_start_time: float = time.time()
        self.time_to_epoch: float

        self.epoch_loss = 0.
        self.running_loss = 0.

        self._last_update_iteration = 0

        if log_dir is not None:
            self.writer = tf.summary.FileWriter(log_dir)
        else:
            self.writer = None

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

    def start_epoch(self, epoch: int) -> None:
        """Log start of epoch."""
        self.epoch_loss = 0.
        self.running_loss = 0.
        if in_ipynb():
            self.pbar = tqdm_notebook(total=self.n_examples,
                                      desc=f"{epoch+1:d}",
                                      leave=True)
        else:
            self.pbar = tqdm(total=self.n_examples,
                             desc=f"{epoch+1:d}",
                             leave=True)
        self._last_update_iteration = -1

    def end_epoch(self) -> None:
        """Log end of epoch."""
        now = time.time()
        self.time_to_epoch = now - self.train_start_time
        if self.pbar:
            self.pbar.update(self.n_examples - self.pbar.n)
            self.pbar.set_postfix({"loss": f"{self.epoch_loss:.2f}"})
            self.pbar.close()
            self.pbar = None

    def end_train(self, validation: bool = False) -> int:
        """End round of training."""
        # Close progress bar in case training was stopped early.
        if self.pbar:
            self.pbar.close()
            self.pbar = None

        if not self.eval_stats:
            return 0
        if validation:
            best_epoch = max(self.eval_stats, key=lambda x: x.score)
        else:
            best_epoch = min(self.eval_stats, key=lambda x: x.loss)

        f1_score, precision, recall, accuracy = best_epoch.score
        loss = best_epoch.loss

        print(f"\nBest epoch: {best_epoch.epoch+1:d}\n------------------", flush=True)
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
                    resultsfile.write(f"  total_time: {self.time_to_epoch:f}\n")
                resultsfile.write(f"  loss: {best_epoch.loss:f}\n")

        return best_epoch.epoch

    def record(self, metrics: Dict[str, float], iteration: int) -> None:
        """Record metrics to tensorboard."""
        for tag, value in metrics.items():
            self.scalar_summary(tag, value, iteration)

    def append_eval_stats(self,
                          eval_stats: ModelStats,
                          validation: bool = False) -> None:
        """Add another set of evaluation metrics."""
        info = {"train/loss": self.epoch_loss}
        if validation:
            print(eval_stats, flush=True)
            score = eval_stats.score
            info["validation/f1"] = score[0]
            info["validation/precision"] = score[1]
            info["validation/recall"] = score[2]
            info["validation/accuracy"] = score[3]
        eval_stats.loss = self.epoch_loss
        eval_stats.time_to_epoch = self.time_to_epoch
        self.eval_stats.append(eval_stats)
        self.record(info, eval_stats.epoch + 1)

    def update(self,
               epoch: int,
               iteration: int,
               loss: torch.Tensor,
               params: dict,
               lrs: Tuple[float, ...]):
        """Update loss."""
        self.epoch_loss += loss.item()
        self.running_loss += loss.item()

        if (iteration + 1) % self.log_interval == 0:
            self.pbar.update(iteration - self._last_update_iteration)
            postfix = {"loss": f"{self.epoch_loss:.4g} ({self.running_loss:.4g})",
                       "lr": f"{max(lrs):.4g}"}
            self.pbar.set_postfix(postfix)
            self._last_update_iteration = iteration
            self.running_loss = 0.

            if self.writer is not None and self.log_weights:
                step = epoch * self.n_examples + iteration + 1
                for tag, value in params:
                    if not value.requires_grad:
                        continue
                    tag = tag.replace(".", "/")
                    self.histo_summary(tag, value.data.cpu().numpy(), step)


class LRFinderLogger:
    # pylint: disable=too-few-public-methods
    """
    Logs progress during learning rate search.

    Parameters
    ----------
    n : int
        The size of the search space.

    """

    def __init__(self, n: int) -> None:
        self.n = n
        if in_ipynb():
            self.pbar = tqdm_notebook(total=self.n, leave=True)
        else:
            self.pbar = tqdm(total=self.n, leave=True)

    def __del__(self):
        self.pbar.close()

    def update(self, lr: float, loss: float) -> None:
        """Update progress."""
        self.pbar.update(1)
        self.pbar.set_postfix({
            "lr": lr,
            "loss": loss,
        })
