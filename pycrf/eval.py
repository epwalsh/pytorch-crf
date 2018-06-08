"""
Evaluation statistics.

The following statistics are broken down for each label:

- ``model``: instances given this label (positives).
- ``match``: instances given this label that are actually this label (true positives).
- ``count``: instances that are actually this label (trues).
- ``precision`` := match / model
- ``recall``    := match / count
- ``f``         := 2 * precision * recall / (precision + recall)

We also calculate the overal item accuracy, i.e. accuracy by each token, and
overall instance accuracy (accuracy by each sentence).
"""

from typing import List, Dict, Union


def num_format(num: Union[float, None]) -> str:
    """Format number prettily."""
    if num is None:
        return "NA"
    return f"{num:.4f}"


class LabelStats:
    # pylint: disable=invalid-name
    """Aggregates evaluation statistics for a single label."""

    def __init__(self, label_id: int, label_name: str) -> None:
        self.label: int = label_id
        self.label_name: str = label_name
        self.model: int = 0
        self.match: int = 0
        self.count: int = 0
        self.precision: float = 0.
        self.recall: float = 0.
        self.f1: float = 0.

    def update(self, true_label: int, prediction: int) -> None:
        """Update internal metrics given a new label, prediction pair."""
        if self.label == true_label:
            self.count += 1
            if true_label == prediction:
                self.match += 1
                self.model += 1
        elif prediction == self.label:
            self.model += 1

    def reset(self) -> None:
        """Reset metrics."""
        self.match = 0
        self.model = 0
        self.count = 0
        self.precision = 0.
        self.recall = 0.
        self.f1 = 0.

    def compile(self) -> None:
        """Compile statistics for a batch."""
        self.precision = self.match / self.model \
            if self.model else None
        self.recall = self.match / self.count
        self.f1 = 2 * self.precision * self.recall / (self.precision + self.recall) \
            if self.precision and self.recall else None


class ModelStats:
    # pylint: disable=invalid-name
    """Aggregates statistics for a model on an evaluation set."""

    def __init__(self,
                 labels_stoi: Dict[str, int],
                 verbose: bool = True) -> None:
        self.label_stats = {
            label_name: LabelStats(label_id, label_name)
            for label_name, label_id in labels_stoi.items()
        }

        self.item_match: int = 0
        self.instance_match: int = 0
        self.instance_count: int = 0
        self.item_count: int = 0
        self.verbose = verbose
        self.macro_avg_precision: float = 0.
        self.macro_avg_recall: float = 0.
        self.macro_avg_f1: float = 0.
        self.instance_accuracy: float = 0.
        self.item_accuracy: float = 0.

        self.best_f1: float = 0.
        self.best_epoch: int = 0

    def __getitem__(self, label: str) -> LabelStats:
        return self.label_stats[label]

    def __str__(self) -> str:
        out = ""
        if self.verbose:
            # Gather stats by label.
            max_label_width = max([len(x) for x in self.label_stats])
            out += f"{'':{max_label_width}s} (match, model, count), (  prec, recall,     f1)\n"
            for lab in self.label_stats:
                lab_stats = self.label_stats[lab]  # pylint: disable=unused-variable
                out += f"{lab:{max_label_width}s} "\
                    f"({lab_stats.match:>5d}, "\
                    f"{lab_stats.model:>5d}, "\
                    f"{lab_stats.count:>5d}), "\
                    f"({num_format(lab_stats.precision):>6s}, "\
                    f"{num_format(lab_stats.recall):>6s}, "\
                    f"{num_format(lab_stats.f1):>6s})\n"

        # Gather macro statistics.
        out += \
            f"Macro avg. precision: {num_format(self.macro_avg_precision)}\n"\
            f"Macro avg. recall:    {num_format(self.macro_avg_recall)}\n"\
            f"Macro avg. f1:        {num_format(self.macro_avg_f1)}\n"\
            f"Item accuracy:        {num_format(self.item_accuracy)}\n"\
            f"Instance accuracy:    {num_format(self.instance_accuracy)}"

        return out

    def update(self, true_labels: List[int], pred_labels: List[int]) -> None:
        """Update metrics for a new sequence of true and predicted labels."""
        results: List[bool] = []
        for true_label, pred_label in zip(true_labels, pred_labels):
            results.append(self._update_item(true_label, pred_label))
        if all(results):
            self.instance_match += 1
        self.item_count += len(true_labels)
        self.instance_count += 1

    def _update_item(self, true_label: int, prediction: int) -> bool:
        for label in self.label_stats:
            self.label_stats[label].update(true_label, prediction)
        if true_label == prediction:
            self.item_match += 1
            return True
        return False

    def reset(self) -> None:
        """Reset metrics."""
        self.item_match = 0
        self.instance_match = 0
        self.item_count = 0
        self.instance_count = 0
        self.macro_avg_precision = 0.
        self.macro_avg_recall = 0.
        self.macro_avg_f1 = 0.
        self.instance_accuracy = 0.
        self.item_accuracy = 0.
        for label in self.label_stats:
            self.label_stats[label].reset()

    def compile(self, epoch: int) -> None:
        """Compile statistics for a batch."""
        precision_sum = 0.
        recall_sum = 0.
        f1_sum = 0.

        # Gather statistics by label.
        for label in self.label_stats:
            label_stats = self.label_stats[label]
            label_stats.compile()
            precision_sum += label_stats.precision or 0.
            recall_sum += label_stats.recall or 0.
            f1_sum += label_stats.f1 or 0.

        # Gather macro statistics.
        self.macro_avg_precision = precision_sum / len(self.label_stats)
        self.macro_avg_recall = recall_sum / len(self.label_stats)
        self.macro_avg_f1 = f1_sum / len(self.label_stats)
        self.instance_accuracy = self.instance_match / self.instance_count \
            if self.instance_count else None
        self.item_accuracy = self.item_match / self.item_count \
            if self.item_count else None

        if self.macro_avg_f1 and self.macro_avg_f1 > self.best_f1:
            self.best_f1 = self.macro_avg_f1
            self.best_epoch = epoch
