"""Tests for pycrf.modules.utils methods."""

import pytest

import torch

from pycrf.nn import utils


def test_check_equal():
    """
    Make sure our helper function for checking tensor equality actually works.
    """
    a = torch.tensor([[1, 2], [3, 4]])
    b = torch.tensor([[1, 2], [3, 4]])
    utils.assert_equal(a, b)


cases1 = [
    (torch.tensor([1]), torch.ByteTensor([[1]]), None),
    (torch.tensor([1]), torch.ByteTensor([[1]]), 1),
    (torch.tensor([2]), torch.ByteTensor([[1, 1]]), None),
    (torch.tensor([2]), torch.ByteTensor([[1, 1]]), 2),
    (torch.tensor([2, 3]), torch.ByteTensor([[1, 1, 0], [1, 1, 1]]), None),
    (torch.tensor([2, 3]), torch.ByteTensor([[1, 1, 0], [1, 1, 1]]), 3),
]


@pytest.mark.parametrize("inp, chk, max_len", cases1)
def test_sequence_mask(inp, chk, max_len):
    """Test `sequence_mask()` method."""
    res = utils.sequence_mask(inp, max_len=max_len)
    utils.assert_equal(res, chk)


cases2 = [
    (
        (torch.tensor([1, 1, 1]), 5),
        torch.tensor([1, 1, 1, 0, 0])
    ),
    (
        (torch.tensor([1, 1, 1, 0, 0]), 5),
        torch.tensor([1, 1, 1, 0, 0])
    ),
    (
        (torch.tensor([1, 1, 1, 1, 1]), 5),
        torch.tensor([1, 1, 1, 1, 1])
    ),
    (
        (
            torch.tensor(
                [[1, 1, 2],
                 [3, 4, 1]]
            ),
            3
        ),
        torch.tensor(
            [[1, 1, 2],
             [3, 4, 1],
             [0, 0, 0]]
        )
    ),
]


@pytest.mark.parametrize("inputs, check", cases2)
def test_pad(inputs, check):
    utils.assert_equal(utils.pad(*inputs), check)


cases3 = [
    (
        ([torch.tensor([1, 1]), torch.tensor([1, 1, 2])], torch.tensor([2, 3])),
        (torch.tensor([[1, 1, 2], [1, 1, 0]]), torch.tensor([3, 2]), torch.tensor([1, 0]))
    ),
    (
        ([torch.tensor([1, 3]), torch.tensor([1, 2])], torch.tensor([2, 2])),
        (torch.tensor([[1, 3], [1, 2]]), torch.tensor([2, 2]), torch.tensor([0, 1]))
    ),
]


@pytest.mark.parametrize("inputs, check", cases3)
def test_sort_and_pad(inputs, check):
    padded, lens, idx = utils.sort_and_pad(*inputs)
    utils.assert_equal(padded, check[0])
    utils.assert_equal(lens, check[1])
    utils.assert_equal(idx, check[2])


cases4 = [
    (
        (torch.tensor([[1, 2, 3], [3, 4, 1]]), torch.tensor([1, 0])),
        torch.tensor([[3, 4, 1], [1, 2, 3]])
    ),
    (
        (torch.tensor([[1, 2, 3], [3, 4, 1]]), torch.tensor([0, 1])),
        torch.tensor([[1, 2, 3], [3, 4, 1]])
    ),
]


@pytest.mark.parametrize("inputs, check", cases4)
def test_unsort(inputs, check):
    utils.assert_equal(utils.unsort(*inputs), check)
