"""
Makes sure evaluation metrics are calculated correctly.
"""

import pytest


SENT_ITEMS = [
    ([0, 0, 0, 0], [0, 0, 0, 0]),
    ([0, 0, 1, 0], [0, 0, 1, 2]),
    ([1, 2], [1, 0]),
]


@pytest.fixture(scope="module")
def eval_stats(get_model_stats):
    return get_model_stats(SENT_ITEMS)


cases = [
    ("O", "match", 6),
    ("O", "model", 7),
    ("O", "count", 7),
    ("O", "precision", 6 / 7),
    ("O", "recall", 6 / 7),
    ("O", "f1", 6 / 7),
]


@pytest.mark.parametrize("label, metric, check", cases)
def test_metrics(eval_stats, label, metric, check):
    assert getattr(eval_stats[label], metric) == check
