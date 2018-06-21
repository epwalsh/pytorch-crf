"""Implements a character-sequence LSTM to generate words features."""

import argparse

import torch
import torch.nn as nn
from torch.nn import LSTM, Embedding, Dropout
from torch.nn.utils.rnn import pack_padded_sequence

from pycrf.io import Vocab
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
        The dropout probability for the recurrent layer and embedding layer.

    embedding_size : int, optional (default: 50)
        The size of the embedding layer.

    padding_idx : int, optional (default: 0)
        The id of the character using for padding.

    Attributes
    ----------
    n_chars : int
        The number of characters in the vocabularly, i.e. the input size.

    output_size : int
        The dimension of the output, which is
        ``layers * hidden_size * directions``.

    embedding : torch.nn.Embedding
        The character embedding layer.

    embedding_dropout : torch.nn.Dropout
        A dropout applied to the embedding features.

    rnn : torch.nn.LSTM
        The LSTM layer.

    """

    def __init__(self,
                 n_chars: int,
                 hidden_size: int,
                 bidirectional: bool = True,
                 embedding_size: int = 50,
                 layers: int = 1,
                 dropout: float = 0.,
                 padding_idx: int = 0) -> None:
        super(CharLSTM, self).__init__()

        self.n_chars = n_chars

        # Character embedding layer.
        self.embedding = \
            Embedding(self.n_chars, embedding_size, padding_idx=padding_idx)

        # Dropout applied to embeddings.
        self.embedding_dropout = \
            Dropout(p=dropout) if dropout else None

        # Recurrent layer.
        self.rnn = LSTM(input_size=embedding_size,
                        hidden_size=hidden_size,
                        num_layers=layers,
                        batch_first=True,
                        dropout=dropout if layers > 1 else 0,
                        bidirectional=bidirectional)

        # Calculate final output size.
        self.output_size = layers * hidden_size
        if bidirectional:
            self.output_size *= 2

    def forward(self,
                inputs: torch.Tensor,
                lengths: torch.Tensor,
                indices: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ
        """
        Make a forward pass through the network.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape ``[sent_length x max_word_length]``.

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
        sent_length = inputs.size()[0]

        # Pass inputs through embedding layer.
        inputs_emb = self.embedding(inputs)
        # inputs_emb: ``[sent_length x max_word_length x embedding_size]``

        # Apply dropout to embeddings.
        if self.embedding_dropout:
            inputs_emb = self.embedding_dropout(inputs_emb)

        # Turned the padded inputs into a packed sequence.
        packed = pack_padded_sequence(inputs_emb, lengths, batch_first=True)

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

    @staticmethod
    def cl_opts(parser: argparse.ArgumentParser, require=True) -> None:
        # pylint: disable=unused-argument
        """Define command-line options specific to this model."""
        group = parser.add_argument_group("Character LSTM options")
        group.add_argument(
            "--char-hidden-dim",
            type=int,
            default=50,
            help="""Dimension of the hidden layer for the character-level
            LSTM. Default is 50."""
        )
        group.add_argument(
            "--char-embedding-size",
            type=int,
            default=50,
            help="""The dimension of the character embedding layer."""
        )

    @classmethod
    def cl_init(cls, opts: argparse.Namespace, vocab: Vocab):
        """Initialize an instance of this model from command-line options."""
        return cls(vocab.n_chars,
                   opts.char_hidden_dim,
                   embedding_size=opts.char_embedding_size,
                   dropout=opts.dropout)
