"""Train logging."""

import time
from typing import List

import torch

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

    @property
    def current_epoch(self):
        """Return current epoch number."""
        return len(self.eval_stats)

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

    def end_train(self) -> None:
        """End round of training."""
        train_duration = time.time() - self.train_start_time
        print(f"Total training time: {train_duration:.0f} seconds", flush=True)

        best_epoch = max(self.eval_stats, key=lambda x: (x.score[0], -1. * x.loss))

        f1_score, precision, recall, accuracy = best_epoch.score
        loss = best_epoch.loss

        print(f"Best epoch: {best_epoch.epoch+1:d}", flush=True)
        if f1_score:
            print(f"f1:        {f1_score:.4f}\n"
                  f"precision: {precision:.4f}\n"
                  f"recall:    {recall:.4f}\n"
                  f"accuracy:  {accuracy:.4f}", flush=True)
        print(f"  loss: {loss:.4f}")

        if self.results_file:
            with open(self.results_file, "a") as resultsfile:
                resultsfile.write("results:\n")
                resultsfile.write(f"  best_epoch: {best_epoch.epoch+1}\n")
                resultsfile.write(f"  micro_avg_f1: {f1_score:f}\n")
                resultsfile.write(f"  micro_avg_precision: {precision:f}\n")
                resultsfile.write(f"  micro_avg_recall: {recall:f}\n")
                resultsfile.write(f"  loss: {best_epoch.loss:f}\n")

    def append_eval_stats(self, eval_stats: ModelStats) -> None:
        """Add another set of evaluation metrics."""
        print(eval_stats, flush=True)
        eval_stats.loss = self.epoch_loss
        self.eval_stats.append(eval_stats)

    def update(self, iteration: int, loss: torch.Tensor):
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
