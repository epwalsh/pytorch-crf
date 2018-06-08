"""Tests for Dataset class."""

import torch

from pycrf.nn.utils import assert_equal


def test_size(vocab_dataset):
    """Make sure the right number of sentences were loaded."""
    assert len(vocab_dataset[1]) == 4


def test_source(vocab_dataset):
    """Make sure the `source` attr contains what we expect."""
    vocab, dataset = vocab_dataset
    assert len(dataset.source) == 4

    # Check the first item, which comes from the sentence:
    # [("hi", "O"), ("there", "O")]
    words, word_lens, word_idxs, word_embs = dataset.source[0]
    assert isinstance(words, torch.Tensor)
    assert isinstance(word_lens, torch.Tensor)
    assert isinstance(word_idxs, torch.Tensor)
    assert isinstance(word_embs, torch.Tensor)

    assert list(words.size()) == [2, 5]
    assert_equal(word_lens, torch.tensor([5, 2]))
    assert_equal(word_idxs, torch.tensor([1, 0]))
    assert list(word_embs.size()) == [2, vocab.word_vec_dim]


def test_target(vocab_dataset):
    """Make sure the `target` attr contains what we expect."""
    vocab, dataset = vocab_dataset
    assert len(dataset.target) == 4
    assert_equal(dataset.target[0], torch.tensor([0, 0]))
    assert_equal(dataset.target[1], torch.tensor([0, 0, 0, 1]))
    assert_equal(dataset.target[2], torch.tensor([0, 0, 1, 2]))
    assert_equal(dataset.target[3], torch.tensor([0]))
