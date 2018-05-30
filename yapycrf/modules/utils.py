"""
Small helper functions.
"""

import torch


def sequence_mask(lens, max_len=None):
    """
    Compute sequence mask.

    Parameters
    ----------
    lens : :obj:`LongTensor`
        Tensor of sequence lengths `[batch_size]`.

    max_len : int
        The maximum length (optional).

    Returns
    -------
    :obj:`LongTensor`
        Returns a tensor of 1's and 0's of size `[batch_size x max_len]`.

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
