"""Tests for Dataset class."""

import torch

from pycrf.nn.utils import assert_equal


def test_size(dataset):
    """Make sure the right number of sentences were loaded."""
    assert len(dataset) == 4


def test_source(dataset):
    """Make sure the `source` attr contains what we expect."""
    assert len(dataset.source) == 4

    # Check the first item, which comes from the sentence:
    # [("hi", "O"), ("there", "O")]
    words, word_lens, idxs, word_idxs, context = dataset.source[0]
    assert isinstance(words, torch.Tensor)
    assert isinstance(word_lens, torch.Tensor)
    assert isinstance(idxs, torch.Tensor)
    assert isinstance(word_idxs, torch.Tensor)
    assert context is None

    assert list(words.size()) == [2, 5]
    assert_equal(word_lens, torch.tensor([5, 2]))
    assert_equal(idxs, torch.tensor([1, 0]))
    assert list(word_idxs.size()) == [2]


def test_target(dataset):
    """Make sure the `target` attr contains what we expect."""
    assert len(dataset.target) == 4
    assert_equal(dataset.target[0], torch.tensor([0, 0]))
    assert_equal(dataset.target[1], torch.tensor([0, 0, 0, 1]))
    assert_equal(dataset.target[2], torch.tensor([0, 0, 1, 2]))
    assert_equal(dataset.target[3], torch.tensor([0]))
