"""Tests for yapycrf.modules.utils methods."""

import pytest

import torch

from yapycrf.modules.utils import sequence_mask


cases = [
    (torch.tensor([1]), torch.ByteTensor([[1]]), None),
    (torch.tensor([1]), torch.ByteTensor([[1]]), 1),
    (torch.tensor([2]), torch.ByteTensor([[1, 1]]), None),
    (torch.tensor([2]), torch.ByteTensor([[1, 1]]), 2),
    (torch.tensor([2, 3]), torch.ByteTensor([[1, 1, 0], [1, 1, 1]]), None),
    (torch.tensor([2, 3]), torch.ByteTensor([[1, 1, 0], [1, 1, 1]]), 3),
]


@pytest.mark.parametrize("inp, chk, max_len", cases)
def test_sequence_mask(inp, chk, max_len):
    """Test `sequence_mask()` method."""
    res = sequence_mask(inp, max_len=max_len)
    assert res.size() == chk.size()
    assert (res == chk).all().item() == 1
