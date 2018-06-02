"""Small helper functions."""

from typing import List, Tuple

import torch


def sequence_mask(lens: torch.Tensor, max_len: int = None) -> torch.ByteTensor:
    """
    Compute sequence mask.

    Parameters
    ----------
    lens : torch.Tensor
        Tensor of sequence lengths ``[batch_size]``.

    max_len : int, optional (default: None)
        The maximum length (optional).

    Returns
    -------
    torch.ByteTensor
        Returns a tensor of 1's and 0's of size ``[batch_size x max_len]``.

    """
    batch_size = lens.size(0)

    if max_len is None:
        max_len = lens.max().item()

    ranges = torch.arange(0, max_len).long()
    ranges = ranges.unsqueeze(0).expand(batch_size, max_len)
    ranges = torch.autograd.Variable(ranges)

    if lens.data.is_cuda:
        ranges = ranges.cuda()

    lens_exp = lens.unsqueeze(1).expand_as(ranges)
    mask = ranges < lens_exp

    return mask


def pad(tensor: torch.Tensor, length: int) -> torch.Tensor:
    """Pad a tensor with zeros."""
    return torch.cat([
        tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()
    ])


def sort_and_pad(tensors: List[torch.Tensor],
                 lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sort and pad list of tensors by their lengths and concatenate them.

    Parameters
    ----------
    tensors : List[torch.Tensor]
        The list of tensors to pad, each has dimension ``[L x *]``, where
        ``L`` is the variable length of each tensor. The remaining dimensions
        of the tensors must be the same.

    lengths : torch.Tensor
        A tensor that holds the length of each tensor in the list.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        The first tensor is the result of concatenating the list and has
        dimension ``[len(tensors) x max_length x *]``.
        The second tensor contains the sorted lengths from the original list.
        The third tensor contains the sorted indices from the original list.

    """
    sorted_lens, sorted_idx = lengths.sort(0, descending=True)
    max_len = sorted_lens[0].item()
    padded = []
    for i in sorted_idx:
        padded.append(pad(tensors[i], max_len).unsqueeze(0))
    return torch.cat(padded), sorted_lens, sorted_idx


def unsort(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Unsort a tensor along dimension 0."""
    unsorted = tensor.new_empty(tensor.size())
    unsorted.scatter_(0, indices.unsqueeze(-1).expand_as(tensor), tensor)
    return unsorted


def assert_equal(tensor_a: torch.Tensor, tensor_b: torch.Tensor) -> None:
    """Check if two tensors are equal."""
    assert tensor_a.size() == tensor_b.size()
    assert (tensor_a == tensor_b).all().item() == 1
