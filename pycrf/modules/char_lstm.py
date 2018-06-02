"""Implements a character-sequence LSTM to generate words features."""

import torch
import torch.nn as nn
from torch.nn import LSTM
from torch.nn.utils.rnn import pack_padded_sequence

from pycrf.nn.utils import unsort


class CharLSTM(nn.Module):
    """
    Character LSTM for generating word features from the final hidden state.

    Parameters
    ----------
    n_chars : int
        The number of characters in the vocabularly, i.e. the input size.

    hidden_size : int
        The number of features in the hidden state of the LSTM cells.

    bidirectional : bool, optional (default: True)
        If true, becomes a bidirectional LSTM.

    layers : int, optional (default: 1)
        The number of cell layers.

    dropout : float, optional (default: 0.)
        The dropout probability for the recurrent layer.

    Attributes
    ----------
    n_chars : int
        The number of characters in the vocabularly, i.e. the input size.

    output_size : int
        The dimension of the output, which is
        ``layers * hidden_size * directions``.

    rnn : torch.nn
        The LSTM layer.

    """

    def __init__(self,
                 n_chars: int,
                 hidden_size: int,
                 bidirectional: bool = True,
                 layers: int = 1,
                 dropout: float = 0.) -> None:
        super(CharLSTM, self).__init__()

        self.n_chars = n_chars
        self.output_size = layers * hidden_size
        if bidirectional:
            self.output_size *= 2

        self.rnn = LSTM(input_size=self.n_chars,
                        hidden_size=hidden_size,
                        num_layers=layers,
                        batch_first=True,
                        dropout=dropout,
                        bidirectional=bidirectional)

    def forward(self,
                inputs: torch.Tensor,
                lengths: torch.Tensor,
                indices: torch.Tensor) -> torch.Tensor:
        """
        Make a forward pass through the network.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape ``[sent_length x word_length x n_chars]``.

        lengths : torch.Tensor
            The length of each word ``[sent_length]``.

        indices : torch.Tensor
            Sorted indices that we can recover the unsorted final hidden
            states.

        Returns
        -------
        torch.Tensor
            The last hidden states:
            ``[len(inputs) x (layers * directions * hidden_size)]``

        """
        # pylint: disable=arguments-differ
        sent_length = inputs.size()[0]

        packed = pack_padded_sequence(inputs, lengths, batch_first=True)

        _, state = self.rnn(packed)
        hidden = state[0]
        # hidden: ``[(layers * directions) x sent_length x hidden_size]``

        # Move sentence dimension to the first dimension and then
        # concatenate the forward/backward hidden states.
        hidden = hidden.permute(1, 0, 2).contiguous().view(sent_length, -1)
        # hidden: ``[sent_length x (layers * directions * hidden_size)]``

        # Unsort the hidden states.
        hidden = unsort(hidden, indices)

        return hidden
