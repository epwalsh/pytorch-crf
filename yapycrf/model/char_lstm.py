"""
Implements a character-sequence LSTM.
"""

import torch.nn as nn


class CharLSTM(nn.Module):
    """
    Character LSTM.

    Parameters
    ----------

    """

    def __init__(self, n_chars):
        super(CharLSTM, self).__init__()
        self.n_chars = n_chars

    def forward(self, inputs):
        """
        Make a forward pass through the network.

        Parameters
        ----------
        inputs : :obj:`torch.Tensor`
            A tensor or :obj:`torch.nn.utils.rnn.PackedSequence` of shape
            `[batch_size x max_word_length x n_chars]`.

        """
        pass
