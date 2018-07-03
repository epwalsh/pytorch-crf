"""Implements a character-level CNN for generating word features."""

import argparse

import torch
import torch.nn as nn
from torch.nn import Conv1d

from pycrf.io import Vocab
from pycrf.nn.utils import unsort
from .char_embedding import CharEmbedding


class CharCNN(nn.Module):
    """
    Character-level CNN for genrating word features from kernels.

    Parameters
    ----------
    n_chars : int
        The number of characters in the vocabularly, i.e. the input size.

    channels : int
        The number of convolution channels.

    kernel_size : int, optional (default: 3)
        The size of the kernels.

    padding : int, optional (default: 2)
        The padding applied before the convolutional layer.

    dropout : float, optional (default: 0.)
        The dropout probability for the embedding layer.

    embedding_size : int, optional (default: 50)
        The size of the embedding layer.

    padding_idx : int, optional (default: 0)
        The id of the character using for padding.

    Attributes
    ----------
    n_chars : int
        The number of characters in the vocabularly, i.e. the input size.

    char_embedding : torch.nn.Embedding
        The character embedding layer.

    cnn : torch.nn.Conv1d
        The convolution layer.

    output_size : int
        The dimension of output.

    """

    def __init__(self,
                 n_chars: int,
                 channels: int,
                 kernel_size: int = 3,
                 embedding_size: int = 50,
                 padding: int = 2,
                 padding_idx: int = 0,
                 dropout: float = 0.) -> None:
        super(CharCNN, self).__init__()

        self.n_chars = n_chars

        # Character embedding layer.
        self.char_embedding = CharEmbedding(n_chars, embedding_size,
                                            dropout=dropout,
                                            padding_idx=padding_idx)

        # Convolutional layer.
        self.cnn = \
            Conv1d(embedding_size, channels, kernel_size, padding=padding)

        self.output_size = channels

    def forward(self,
                inputs: torch.Tensor,
                lengths: torch.Tensor,
                indices: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ,unused-argument
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
            The word features:
            ``[sent_length x channels]``

        """
        # Pass inputs through embedding layer.
        inputs_emb = self.char_embedding(inputs).permute(0, 2, 1)
        # inputs_emb: ``[sent_length x embedding_size x max_word_length ]``

        # Run embeddings through convolution layer.
        output = self.cnn(inputs_emb)
        # output: ``[sent_length x channels x out_length]``
        # ``out_length`` is a function of the ``max_word_length``,
        # ``kernel_size``, and ``padding``.

        # Apply max pooling across each word.
        output, _ = torch.max(output, 2)
        # output: ``[sent_length x channels]``

        # Unsort the words.
        output = unsort(output, indices)

        return output

    @staticmethod
    def cl_opts(parser: argparse.ArgumentParser, require=True) -> None:
        # pylint: disable=unused-argument
        """Define command-line options specific to this model."""
        group = parser.add_argument_group("Character CNN options")
        CharEmbedding.cl_opts(group)
        group.add_argument(
            "--cnn-channels",
            type=int,
            default=30,
            help="""Number of convolutional channels. Default is 30."""
        )
        group.add_argument(
            "--cnn-padding",
            type=int,
            default=2,
            help="""Padding applied before CNN layer. Default is 2."""
        )
        group.add_argument(
            "--cnn-kernel-size",
            type=int,
            default=3,
            help="""Kernel size of the convolutions. Default is 3."""
        )

    @classmethod
    def cl_init(cls, opts: argparse.Namespace, vocab: Vocab):
        """Initialize an instance of this model from command-line options."""
        return cls(
            vocab.n_chars,
            opts.cnn_channels,
            kernel_size=opts.cnn_kernel_size,
            padding=opts.cnn_padding,
            dropout=opts.dropout,
            embedding_size=opts.char_embedding_size)
