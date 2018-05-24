"""Tests for Dataset class."""

import torch


def test_size(dataset):
    """Make sure the right number of sentences were loaded."""
    assert len(dataset) == 4


def test_source(dataset, vocab):
    """Make sure the `source` attr contains what we expect."""
    assert len(dataset.source) == 4

    # Check the first item, which comes from the sentence:
    # [("hi", "O"), ("there", "O")]
    assert isinstance(dataset.source[0], tuple)
    assert isinstance(dataset.source[0][0], list)
    assert isinstance(dataset.source[0][1], torch.Tensor)
    assert len(dataset.source[0][0]) == 2
    assert dataset.source[0][0][0].size()[0] == 2
    assert dataset.source[0][0][0].size()[1] == vocab.n_chars


def test_target(dataset, vocab):
    """Make sure the `target` attr contains what we expect."""
    assert len(dataset.target) == 4
    assert (dataset.target[0] == torch.Tensor([0., 0.])).all()\
        .item() == 1
    assert (dataset.target[1] == torch.Tensor([0., 0., 0., 1.]))\
        .all().item() == 1
    assert (dataset.target[2] == torch.Tensor([0., 0., 1., 2.]))\
        .all().item() == 1
    assert (dataset.target[3] == torch.Tensor([0.]))\
        .all().item() == 1
