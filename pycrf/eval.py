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

from functools import wraps
from typing import List, Dict, Union


def safe_divide(func):
    """Decorate functions that perform division and print the output."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        """Try to safely divide and then returns a formatted str."""
        try:
            return func(*args, **kwargs)
        except ZeroDivisionError:
            return None
    return wrapper


def num_format(num: Union[float, None]) -> str:
    """Format number prettily."""
    if num is None:
        return "NA"
    return f"{num:.4f}"


class LabelStats:
    # pylint: disable=invalid-name
    """Aggregates evaluation statistics for a single label."""

    def __init__(self, label_id: int, label_name: str) -> None:
        self.label = label_id
        self.label_name = label_name
        self.model = 0
        self.match = 0
        self.count = 0

    def __str__(self) -> str:
        precision, recall, f1, count = self.stats
        return self.str_format(self.label_name, precision, recall, f1, count)

    @staticmethod
    def str_format(label_name, precision, recall, f1, count) -> str:
        """Get pretty string format."""
        return \
            f"{label_name}:\n"\
            f"    Precision: {num_format(precision)}\n"\
            f"    Recall:    {num_format(recall)}\n"\
            f"    F1:        {num_format(f1)}\n"\
            f"    Count:     {count:d}"

    @property  # type: ignore
    def stats(self):
        """Get precision / recall / f1 and count."""
        precision = self.precision
        recall = self.recall
        f1 = self._get_f1(precision, recall)
        return precision, recall, f1, self.count

    @property  # type: ignore
    @safe_divide
    def precision(self):
        """Return label precision."""
        return self.match / self.model

    @property  # type: ignore
    @safe_divide
    def recall(self):
        """Return label recall."""
        return self.match / self.count

    @property  # type: ignore
    @safe_divide
    def f1(self):
        """Return label f1 score."""
        precision = self.precision
        recall = self.recall
        return self._get_f1(precision, recall)

    @staticmethod
    @safe_divide
    def _get_f1(precision, recall):
        if precision is None or recall is None:
            return None
        return 2 * precision * recall / (precision + recall)

    def update(self, true_label: int, prediction: int) -> None:
        """Update internal metrics given a new label, prediction pair."""
        if self.label == true_label:
            self.count += 1
            if true_label == prediction:
                self.match += 1
                self.model += 1
        elif prediction == self.label:
            self.model += 1

    def reset(self):
        """Reset metrics."""
        self.match = 0
        self.model = 0
        self.count = 0


class ModelStats:
    # pylint: disable=invalid-name
    """Aggregates statistics for a model on an evaluation set."""

    def __init__(self, labels_stoi: Dict[str, int]) -> None:
        self.label_stats = {
            label_name: LabelStats(label_id, label_name)
            for label_name, label_id in labels_stoi.items()
        }
        self.item_match = 0
        self.instance_match = 0
        self.instance_count = 0
        self.item_count = 0

    @property  # type: ignore
    @safe_divide
    def item_accuracy(self):
        """Accuracy per token."""
        return self.item_match / self.item_count

    @property  # type: ignore
    @safe_divide
    def instance_accuracy(self):
        """Accuracy per sentence."""
        return self.instance_match / self.instance_count

    def __getitem__(self, label: str) -> LabelStats:
        return self.label_stats[label]

    def __str__(self) -> str:
        precision_sum = 0.
        recall_sum = 0.
        f1_sum = 0.

        # Gather statistics by label.
        by_label = []
        for label in self.label_stats:
            label_stats = self.label_stats[label]
            precision, recall, f1, count = label_stats.stats
            by_label.append(
                label_stats.str_format(label, precision, recall, f1, count))
            precision_sum += precision or 0.
            recall_sum += recall or 0.
            f1_sum += f1 or 0.
        out = "\n".join(by_label) + "\n"

        # Gather macro statistics.
        macro_avg_precision = precision_sum / len(self.label_stats)
        macro_avg_recall = recall_sum / len(self.label_stats)
        macro_avg_f1 = f1_sum / len(self.label_stats)
        out += \
            f"Macro avg. precision: {num_format(macro_avg_precision)}\n"\
            f"Macro avg. recall:    {num_format(macro_avg_recall)}\n"\
            f"Macro avg. f1:        {num_format(macro_avg_f1)}\n"\
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

    def reset(self):
        """Reset metrics."""
        self.item_match = 0
        self.instance_match = 0
        self.item_count = 0
        self.instance_count = 0
        for label in self.label_stats:
            self.label_stats[label].reset()
