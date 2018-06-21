"""Evaluation statistics."""

from enum import Enum
from typing import Iterable, Dict, Set, List


class Scheme(Enum):
    """Enum class for labelling schemes."""

    IOB = "iob"
    IOBES = "iobes"


def iob_to_spans(sequence: Iterable,
                 lut: Dict[int, str],
                 strict_iob2: bool = False) -> Set[str]:
    # pylint: disable=invalid-name,too-many-branches,unsubscriptable-object
    """Convert to IOB to span."""
    iobtype = 2 if strict_iob2 else 1
    chunks: List[str] = []
    current: List[str] = None

    for i, y in enumerate(sequence):
        label = lut[y]

        if label.startswith('B-'):
            if current is not None:
                chunks.append('@'.join(current))
            current = [label.replace('B-', ''), '%d' % i]

        elif label.startswith('I-'):

            if current is not None:
                base = label.replace('I-', '')
                if base == current[0]:
                    current.append('%d' % i)
                else:
                    chunks.append('@'.join(current))
                    if iobtype == 2:
                        print(f"Warning, type=IOB2, unexpected format ([{label}]"
                              f"follows other tag type [{current[0]}] @ {i:d})")

                    current = [base, '%d' % i]

            else:
                current = [label.replace('I-', ''), '%d' % i]
                if iobtype == 2:
                    print('Warning, unexpected format (I before B @ {i:d}) {label}')
        else:
            if current is not None:
                chunks.append('@'.join(current))
            current = None

    if current is not None:
        chunks.append('@'.join(current))

    return set(chunks)


def iobes_to_spans(sequence: Iterable,
                   lut: Dict[int, str],
                   strict_iob2: bool = False) -> Set[str]:
    # pylint: disable=invalid-name,too-many-branches,unsubscriptable-object,too-many-statements
    """Convert to IOBES to span."""
    iobtype = 2 if strict_iob2 else 1
    chunks: List[str] = []
    current: List[str] = None

    for i, y in enumerate(sequence):
        label = lut[y]

        if label.startswith('B-'):

            if current is not None:
                chunks.append('@'.join(current))
            current = [label.replace('B-', ''), '%d' % i]

        elif label.startswith('S-'):

            if current is not None:
                chunks.append('@'.join(current))
                current = None
            base = label.replace('S-', '')
            chunks.append('@'.join([base, '%d' % i]))

        elif label.startswith('I-'):

            if current is not None:
                base = label.replace('I-', '')
                if base == current[0]:
                    current.append('%d' % i)
                else:
                    chunks.append('@'.join(current))
                    if iobtype == 2:
                        print('Warning')
                    current = [base, '%d' % i]

            else:
                current = [label.replace('I-', ''), '%d' % i]
                if iobtype == 2:
                    print('Warning')

        elif label.startswith('E-'):

            if current is not None:
                base = label.replace('E-', '')
                if base == current[0]:
                    current.append('%d' % i)
                    chunks.append('@'.join(current))
                    current = None
                else:
                    chunks.append('@'.join(current))
                    if iobtype == 2:
                        print('Warning')
                    current = [base, '%d' % i]
                    chunks.append('@'.join(current))
                    current = None

            else:
                current = [label.replace('E-', ''), '%d' % i]
                if iobtype == 2:
                    print('Warning')
                chunks.append('@'.join(current))
                current = None
        else:
            if current is not None:
                chunks.append('@'.join(current))
            current = None

    if current is not None:
        chunks.append('@'.join(current))

    return set(chunks)


def _detect_label_scheme(labels: Iterable):
    prefixes = [lab[:2] for lab in labels]
    if all([x in ["O", "B-", "I-"] for x in prefixes]):
        return Scheme.IOB
    if all([x in ["O", "B-", "I-", "S-", "E-"] for x in prefixes]):
        return Scheme.IOBES
    raise ValueError("Unknown labelling scheme")


class ModelStats:
    # pylint: disable=invalid-name
    """Aggregates statistics for a model on an evaluation set."""

    def __init__(self,
                 labels_itos: Dict[int, str],
                 epoch: int,
                 loss: float = 0.) -> None:
        self.labels_itos = labels_itos
        self.correct_labels = 0
        self.total_labels = 0
        self.gold_count = 0
        self.guess_count = 0
        self.overlap_count = 0
        self.loss = loss
        self.epoch = epoch
        self.time_to_epoch: float = None
        self._scheme = \
            _detect_label_scheme([lab for _, lab in labels_itos.items()])
        self.span_getter = iob_to_spans \
            if self._scheme == Scheme.IOB else iobes_to_spans

    @property
    def score(self):
        """
        Calculate aggregate scores.

        Returns the micro-average f1, precision, and recall for totally
        correct entities, as well accuracy by token.
        """
        if self.guess_count == 0:
            return 0., 0., 0., 0.
        accuracy = self.correct_labels / self.total_labels
        precision = self.overlap_count / self.guess_count
        recall = self.overlap_count / self.gold_count
        if precision == 0.0 or recall == 0.0:
            return 0.0, 0.0, 0.0, accuracy
        f = 2 * (precision * recall) / (precision + recall)
        return f, precision, recall, accuracy

    def __str__(self) -> str:
        f1, precision, recall, accuracy = self.score
        return \
            "Validation set scores:\n"\
            f"precision:      {precision:.4f}\n"\
            f"recall:         {recall:.4f}\n"\
            f"f1:             {f1:.4f}\n"\
            f"token accuracy: {accuracy:.4f}\n"

    def reset(self) -> None:
        """Reset metrics."""
        self.correct_labels = 0
        self.total_labels = 0
        self.gold_count = 0
        self.guess_count = 0
        self.overlap_count = 0

    def update(self, gold_labels: List[int], predicted: List[int]) -> None:
        """Update counts based on a new instance."""
        self.total_labels += len(gold_labels)
        self.correct_labels += \
            len([(l, p) for l, p in zip(gold_labels, predicted) if l == p])

        gold_chunks = self.span_getter(gold_labels, self.labels_itos)
        self.gold_count += len(gold_chunks)

        guess_chunks = self.span_getter(predicted, self.labels_itos)
        self.guess_count += len(guess_chunks)

        overlap_chunks = gold_chunks & guess_chunks
        self.overlap_count += len(overlap_chunks)
