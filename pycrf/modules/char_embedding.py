"""Character embedding module."""

import torch
import torch.nn as nn
from torch.nn import Dropout, Embedding


class CharEmbedding(nn.Module):
    """
    Dense character embedding.

    Parameters
    ----------
    n_chars : int
        The number of characters in the vocabularly, i.e. the input size.

    embedding_size : int
        The dimension of the embedding.

    dropout : float, optional (default: 0.)
        The dropout probability.

    padding_idx : int, optional (default: 0)
        The id of the character using for padding.

    Attributes
    ----------
    n_chars : int
        The number of characters in the vocabularly, i.e. the input size.

    embedding : torch.nn.Embedding
        The character embedding layer.

    embedding_dropout : torch.nn.Dropout
        A dropout applied to the embedding features.

    """

    def __init__(self,
                 n_chars: int,
                 embedding_size: int,
                 dropout: float = 0.,
                 padding_idx: int = 0) -> None:
        super(CharEmbedding, self).__init__()
        self.n_chars = n_chars

        # Character embedding layer.
        self.embedding = \
            Embedding(self.n_chars, embedding_size, padding_idx=padding_idx)

        # Dropout applied to embeddings.
        self.embedding_dropout = \
            Dropout(p=dropout) if dropout else None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ
        """
        Make foward pass.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape ``[sent_length x max_word_length]``.

        Returns
        -------
        torch.Tensor
            The last hidden states:
            ``[sent_length x max_word_length x embedding_size]``

        """
        # Pass inputs through embedding layer.
        inputs_emb = self.embedding(inputs)
        # inputs_emb: ``[sent_length x max_word_length x embedding_size]``

        # Apply dropout to embeddings.
        if self.embedding_dropout:
            inputs_emb = self.embedding_dropout(inputs_emb)

        return inputs_emb

    @staticmethod
    def cl_opts(group) -> None:
        """Define command-line options specific to this model."""
        group.add_argument(
            "--char-embedding-size",
            type=int,
            default=50,
            help="""The dimension of the character embedding layer.
            The default is 50."""
        )
