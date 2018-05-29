"""Implements a character-sequence LSTM to generate words features."""

import torch
import torch.nn as nn
from torch.nn import LSTM


class CharLSTM(nn.Module):
    """
    Character LSTM for generating word features from the final hidden state.

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
        inputs : list of :obj:`torch.Tensor`
            List of tensors of shape `[word_length x n_chars]`.

        Returns
        -------
        :obj:`torch.Tensor`
            The last hidden states:
            `[len(inputs) x (layers x directions x hidden_size)]`

        """
        hiddens = []
        for word in inputs:
            _, state = self.rnn(word.unsqueeze(0))

            # `[(layers x directions) x 1 x hidden_size]`
            hidden = state[0]

            # Get rid of batch_size dimension.
            # `[(layers x directions) x hidden_size]`
            hidden = hidden.squeeze()

            # Concatenate forward/backward hidden states.
            # Changes to `[1 x (layers x directions x hidden_size)]`.
            hidden = hidden.view(-1).unsqueeze(0)

            hiddens.append(hidden)

        # `[words x (layers x directions x hidden_size)]`
        hiddens = torch.cat(hiddens, dim=0)

        return hiddens
