"""
Implements a character-sequence LSTM.
"""

import torch.nn as nn
from torch.nn import LSTM


class CharLSTM(nn.Module):
    """
    Character LSTM.

    Parameters
    ----------
    n_chars : int
        The number of characters in the vocabularly, i.e. the input size.

    hidden_size : int
        The number of features in the hidden state of the LSTM cells.

    bidirectional : bool
        If true, becomes a bidirectional LSTM.

    layers : int
        The number of cell layers.

    Attributes
    ----------
    n_chars : int
        The number of characters in the vocabularly, i.e. the input size.

    hidden_size : int
        The number of features in the hidden state of the LSTM cells.

    bidirectional : bool
        If true, becomes a bidirectional LSTM.

    layers : int
        The number of cell layers.

    rnn : :obj:`torch.nn`
        The LSTM layer.

    """

    def __init__(self, n_chars, hidden_size, bidirectional=True, layers=1):
        super(CharLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.n_chars = n_chars
        self.bidirectional = bidirectional
        self.layers = layers
        self.rnn = LSTM(self.n_chars, self.hidden_size, self.layers,
                        batch_first=True, bidirectional=self.bidirectional)

    def forward(self, inputs):
        """
        Make a forward pass through the network.

        Parameters
        ----------
        inputs : :obj:`torch.Tensor`
            A tensor or :obj:`torch.nn.utils.rnn.PackedSequence` of shape
            `[batch_size x max_word_length x n_chars]`.

        Returns
        -------
        :obj:`torch.Tensor`
            The last hidden state:
            `[batch_size x (layers x directions x hidden_size)]`

        """
        _, state = self.rnn(inputs)

        # `[(layers x directions) x batch_size x hidden_size]`
        hidden = state[0]

        # Put back to `batch_first` alignment.
        # Changes to `[batch_size x (layers x directions) x hidden_size]`
        hidden = hidden.permute(1, 0, 2)
        hidden = hidden.contiguous()

        # Concatenate forward/backward hidden states.
        # Changes to `[batch_size x (layers x directions x hidden_size)]`.
        hidden = hidden.view(2, -1)

        return hidden
