"""
Defines CRF model.

Adapted from kaniblu/pytorch-bilstmcrf.
"""

import torch
import torch.nn as nn

from .utils import log_sum_exp, sequence_mask


class CRF(nn.Module):
    """
    Linear Chain Conditional Random Field model.

    Paramters
    ---------
    n_labels : int
        The number of distinct labels/tags.

    Attributes
    ----------
    n_labels : int
        The number of distint labels/tags in addition to a start tag and stop
        tag.

    start_idx : int
        The index of the start label.

    stop_idx : int
        The index of the stop label.

    transitions : :obj:`torch.nn.Parameter`
        Transition score weights for labels.

    """

    def __init__(self, n_labels):
        super(CRF, self).__init__()

        self.n_labels = n_labels + 2
        self.start_idx = n_labels + 1
        self.stop_idx = n_labels
        self.transitions = \
            nn.Parameter(torch.randn(self.n_labels, self.n_labels))

    def reset_parameters(self):
        """
        Resets parameters with random weights.
        """
        torch.nn.init.normal(self.transitions.data, 0, 1)

    def forward_alg(self, feats, lens):
        """
        Calculates the normalizing constant using the a forward pass of the
        forward-backward algorithm.

        Parameters
        ----------
        feats : :obj:`FloatTensor`
            `[batch_size x seq_len x n_labels]`

        lens : :obj:`LongTensor`
            `[batch_size]`

        Returns
        -------
        :obj:`Tensor`
            The normalizing constants `[batch_size]`.

        """
        batch_size, _, _ = feats.size()
        alpha = feats.data.new(batch_size, self.n_labels).fill_(-10000)
        alpha[:, self.start_idx] = 0
        alpha = torch.autograd.Variable(alpha)
        c_lens = lens.clone()

        feats_t = feats.transpose(1, 0)
        for logit in feats_t:
            logit_exp = logit.unsqueeze(-1).expand(batch_size,
                                                   *self.transitions.size())
            alpha_exp = alpha.unsqueeze(1).expand(batch_size,
                                                  *self.transitions.size())
            trans_exp = self.transitions.unsqueeze(0).expand_as(alpha_exp)
            mat = trans_exp + alpha_exp + logit_exp
            alpha_nxt = log_sum_exp(mat, 2).squeeze(-1)

            mask = (c_lens > 0).float().unsqueeze(-1).expand_as(alpha)
            alpha = mask * alpha_nxt + (1 - mask) * alpha
            c_lens = c_lens - 1

        alpha = alpha + \
            self.transitions[self.stop_idx].unsqueeze(0).expand_as(alpha)
        norm = log_sum_exp(alpha, 1).squeeze(-1)

        return norm

    def forward(self, feats, lens):
        """
        Output the most likely tag sequence (just applies the Viterbi
        algorithm).
        """
        return self.viterbi_decode(feats, lens)

    def viterbi_decode(self, feats, lens):
        """
        Decode a sequence of features in the most likely tag sequence
        using the Viterbi algorithm.

        Parameters
        ----------
        feats : :obj:`FloatTensor`
            `[batch_size x seq_len x n_labels]`

        lens : :obj:`LongTensor`
            `[batch_size]`

        Returns
        -------
        tuple (:obj:`Tensor`, :obj:`LongTensor`)
            The viterbi scores `[batch_size]` and best paths
            `[batch_size x seq_len]`.

        """
        batch_size, _, n_labels = feats.size()
        vit = feats.data.new(batch_size, self.n_labels).fill_(-10000)
        vit[:, self.start_idx] = 0
        vit = torch.autograd.Variable(vit)
        c_lens = lens.clone()

        feats_t = feats.transpose(1, 0)
        pointers = []
        for logit in feats_t:
            vit_exp = vit.unsqueeze(1).expand(batch_size, n_labels, n_labels)
            trn_exp = self.transitions.unsqueeze(0).expand_as(vit_exp)
            vit_trn_sum = vit_exp + trn_exp
            vt_max, vt_argmax = vit_trn_sum.max(2)

            vt_max = vt_max.squeeze(-1)
            vit_nxt = vt_max + logit
            pointers.append(vt_argmax.squeeze(-1).unsqueeze(0))

            mask = (c_lens > 0).float().unsqueeze(-1).expand_as(vit_nxt)
            vit = mask * vit_nxt + (1 - mask) * vit

            mask = (c_lens == 1).float().unsqueeze(-1).expand_as(vit_nxt)
            vit += mask * self.transitions[self.stop_idx]\
                .unsqueeze(0)\
                .expand_as(vit_nxt)

            c_lens = c_lens - 1

        pointers = torch.cat(pointers)
        scores, idx = vit.max(1)
        idx = idx.squeeze(-1)
        paths = [idx.unsqueeze(1)]

        for argmax in reversed(pointers):
            idx_exp = idx.unsqueeze(-1)
            idx = torch.gather(argmax, 1, idx_exp)
            idx = idx.squeeze(-1)

            paths.insert(0, idx.unsqueeze(1))

        paths = torch.cat(paths[1:], 1)
        scores = scores.squeeze(-1)

        return scores, paths

    def transition_score(self, labels, lens):
        """
        Calculates transition scores from one label to the next.

        Parameters
        ----------
        labels : :obj:`LongTensor`
            Sequence of labels `[batch_size x seq_len]`.

        lens : :obj:`LongTensor`
            Gives the sequence length for each example in the batch
            `[batch_size]`.

        Returns
        -------
        :obj:`Tensor`
            The transition scores `[batch_size]`.

        """
        batch_size, seq_len = labels.size()

        # Pad labels with <start> and <stop> indices
        labels_ext = torch.autograd.Variable(
            labels.data.new(batch_size, seq_len + 2))
        labels_ext[:, 0] = self.start_idx
        labels_ext[:, 1:-1] = labels
        mask = sequence_mask(lens + 1, max_len=seq_len + 2).long()
        pad_stop = torch.autograd.Variable(
            labels.data.new(1).fill_(self.stop_idx))
        pad_stop = pad_stop.unsqueeze(-1).expand(batch_size, seq_len + 2)
        labels_ext = (1 - mask) * pad_stop + mask * labels_ext
        labels = labels_ext

        trn = self.transitions

        # Obtain transition vector for each label in batch and timestep
        # (except the last ones)
        trn_exp = trn.unsqueeze(0).expand(batch_size, *trn.size())
        lbl_r = labels[:, 1:]
        lbl_rexp = lbl_r.unsqueeze(-1).expand(*lbl_r.size(), trn.size(0))
        trn_row = torch.gather(trn_exp, 1, lbl_rexp)

        # Obtain transition score from the transition vector for each label
        # in batch and timestep (except the first ones)
        lbl_lexp = labels[:, :-1].unsqueeze(-1)
        trn_scr = torch.gather(trn_row, 2, lbl_lexp)
        trn_scr = trn_scr.squeeze(-1)

        mask = sequence_mask(lens + 1).float()
        trn_scr = trn_scr * mask
        score = trn_scr.sum(1).squeeze(-1)

        return score
