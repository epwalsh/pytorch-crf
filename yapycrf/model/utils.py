"""
Small helper functions.
"""

import torch


def log_sum_exp(vec, dim=0):
    """
    Computes log-sum-exp along the specified dimension in a numerically stable
    way.
    """
    max_score, _ = torch.max(vec, dim)
    max_exp = max_score.unsqueeze(-1).expand_as(vec)
    return max_score + torch.log(torch.sum(torch.exp(vec - max_exp), dim))


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
        max_len = lens.max().data[0]

    ranges = torch.arange(0, max_len).long()
    ranges = ranges.unsqueeze(0).expand(batch_size, max_len)
    ranges = torch.autograd.Variable(ranges)

    if lens.data.is_cuda:
        ranges = ranges.cuda()

    lens_exp = lens.unsqueeze(1).expand_as(ranges)
    mask = ranges < lens_exp

    return mask
