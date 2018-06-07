"""Defines a Bi-LSMT CRF model."""

import argparse
from typing import List, Tuple

import torch
import torch.nn as nn

from allennlp.modules.conditional_random_field import ConditionalRandomField
from pycrf.io import Vocab
from pycrf.nn.utils import sequence_mask
from .char_lstm import CharLSTM


class LSTMCRF(nn.Module):
    """
    (Bi-)LSTM CRF model.

    Parameters
    ----------
    vocab : pycrf.io.Vocab
        The vocab object which contains a dict of known characters and word
        embeddings.

    char_lstm : pycrf.model.CharLSTM
        The character-level LSTM layer.

    crf : allennlp.modules.conditional_random_field.ConditionalRandomField
        The CRF model.

    hidden_dim : int, optional (default: 100)
        The hidden dimension of the recurrent layer.

    layers : int, optional (default: 1)
        The number of layers of cells in the recurrent layer.

    dropout : float, optional (default: 0.)
        The dropout probability for the recurrent layer.

    bidirectional : bool, optional (default: True)
        If True, bidirectional recurrent layer is used, otherwise single
        direction.

    Attributes
    ----------
    vocab : pycrf.io.Vocab
        The vocab object which contains a dict of known characters and word
        embeddings.

    char_lstm : pycrf.model.CharLSTM
        The character-level LSTM layer.

    crf : allennlp.modules.conditional_random_field.ConditionalRandomField
        The CRF model.

    rnn_output_size : int
        The output dimension of the recurrent layer.

    rnn : nn.Module
        The recurrent layer of the network.

    rnn_to_crf : nn.Module
        The linear layer that maps the hidden states from the recurrent layer
        to the label space.

    """

    def __init__(self,
                 vocab: Vocab,
                 char_lstm: CharLSTM,
                 crf: ConditionalRandomField,
                 hidden_dim: int = 100,
                 layers: int = 1,
                 dropout: float = 0.,
                 bidirectional: bool = True) -> None:
        super(LSTMCRF, self).__init__()

        assert vocab.n_chars == char_lstm.n_chars
        assert vocab.n_labels == crf.num_tags

        self.vocab = vocab
        self.char_lstm = char_lstm
        self.crf = crf

        # Recurrent layer. Takes as input the concatenation of the char_lstm
        # final hidden state and pre-trained embedding for each word.
        # The dimension of the output is given by self.rnn_output_size (see
        # below).
        self.rnn = nn.LSTM(
            input_size=vocab.word_vec_dim + char_lstm.output_size,
            hidden_size=hidden_dim,
            num_layers=layers,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=True,
        )

        # This is the size of the recurrent layer's output (see self.rnn).
        self.rnn_output_size = hidden_dim
        if bidirectional:
            self.rnn_output_size *= 2

        # Linear layer that takes the output from the recurrent layer and each
        # time step and transforms into scores for each label.
        self.rnn_to_crf = nn.Linear(self.rnn_output_size, self.vocab.n_labels)

    def _feats(self,
               words: torch.Tensor,
               word_lengths: torch.Tensor,
               word_indices: torch.Tensor,
               word_embs: torch.Tensor) -> torch.Tensor:
        """
        Generate features for the CRF from input.

        First we generate a vector for each word by running each word
        char-by-char through the char_lstm and then concatenating those final
        hidden states for each word with the pre-trained embedding of the word.
        That word vector is than ran through the RNN and then the linear layer.

        Parameters
        ----------
        words : torch.Tensor
            Word tensors ``[sent_length x max_word_lenth x n_chars]``.

        word_lengths : torch.Tensor
            Contains the length of each word (in characters), with shape
            ``[sent_length]``.

        word_indices : torch.Tensor
            Contains sorted indices of words by length, with shape
            ``[sent_length]``.

        word_embs : torch.Tensor
            Pretrained word embeddings with shape
            ``[sent_length x word_emb_dim]``.

        Returns
        -------
        torch.Tensor
            ``[batch_size x sent_length x crf.n_labels]``

        """
        # Run each word character-by-character through the CharLSTM to generate
        # character-level word features.
        char_feats = self.char_lstm(words, word_lengths, word_indices)
        # char_feats: ``[sent_length x char_lstm.output_size]``

        # Concatenate the character-level word features and word embeddings.
        word_feats = torch.cat([char_feats, word_embs], dim=-1)
        # word_feats: ``[sent_length x
        #                (char_lstm.output_size + vocab.word_vec_dim)]``

        # Add a fake batch dimension.
        word_feats = word_feats.unsqueeze(0)
        # word_feats: ``[1 x sent_length x
        #                (char_lstm.output_size + vocab.word_vec_dim)]``

        # Run word features through the LSTM.
        lstm_feats, _ = self.rnn(word_feats)
        # lstm_feats: ``[1 x sent_length x rnn_output_size]``

        # Run recurrent output through linear layer to generate the by-label
        # features.
        feats = self.rnn_to_crf(lstm_feats)
        # feats: ``[1 x sent_length x crf.n_labels]``

        return feats

    def predict(self,
                words: torch.Tensor,
                word_lengths: torch.Tensor,
                word_indices: torch.Tensor,
                word_embs: torch.Tensor,
                lens: torch.Tensor = None) -> List[Tuple[List[int], float]]:
        """
        Compute the best tag sequence.

        Parameters
        ----------
        words : torch.Tensor
            Word tensors ``[sent_length x max_word_lenth x n_chars]``.

        word_lengths : torch.Tensor
            Contains the length of each word (in characters), with shape
            ``[sent_length]``.

        word_indices : torch.Tensor
            Contains sorted indices of words by length, with shape
            ``[sent_length]``.

        word_embs : torch.Tensor
            Pretrained word embeddings with shape
            ``[sent_length x word_emb_dim]``.

        lens : torch.Tensor, optional (default: None)
            Gives the length of each sentence in the batch ``[batch_size]``.

        Returns
        -------
        List[List[int]]
            The best path for each sentence in the batch.

        """
        # pylint: disable=not-callable
        if lens is None:
            lens = torch.tensor([words.size(0)])
        mask = sequence_mask(lens)

        # Gather word feats.
        feats = self._feats(words, word_lengths, word_indices, word_embs)
        # feats: ``[1 x sent_length x n_labels]``

        # Run features through Viterbi decode algorithm.
        preds = self.crf.viterbi_tags(feats, mask)

        return preds

    def forward(self,
                words: torch.Tensor,
                word_lengths: torch.Tensor,
                word_indices: torch.Tensor,
                word_embs: torch.Tensor,
                labs: torch.Tensor,
                lens: torch.Tensor = None) -> torch.Tensor:
        """
        Compute the negative of the log-likelihood.

        Parameters
        ----------
        words : torch.Tensor
            Word tensors ``[sent_length x max_word_lenth x n_chars]``.

        word_lengths : torch.Tensor
            Contains the length of each word (in characters), with shape
            ``[sent_length]``.

        word_indices : torch.Tensor
            Contains sorted indices of words by length, with shape
            ``[sent_length]``.

        word_embs : torch.Tensor
            Pretrained word embeddings with shape
            ``[sent_length x word_emb_dim]``.

        labs : torch.Tensor
            Corresponding target label sequence with shape ``[sent_length]``.

        lens : torch.Tensor, optional (default: None)
            Gives the length of each sentence in the batch ``[batch_size]``.

        Returns
        -------
        torch.Tensor
            The negative log-likelihood evaluated at the inputs.

        """
        # pylint: disable=arguments-differ,not-callable
        if lens is None:
            lens = torch.tensor([words.size(0)], device=words.device)
        mask = sequence_mask(lens)

        # Fake batch dimension for ``labs``.
        labs = labs.unsqueeze(0)
        # labs: ``[1 x sent_length]``

        # Gather word feats.
        feats = self._feats(words, word_lengths, word_indices, word_embs)
        # feats: ``[1 x sent_length x n_labels]``

        loglik = self.crf(feats, labs, mask=mask)

        return -1. * loglik

    @staticmethod
    def cl_opts(parser: argparse.ArgumentParser) -> None:
        """Define command-line options specific to this model."""
        group = parser.add_argument_group("Bi-LSTM CRF options")
        group.add_argument(
            "--char_hidden_dim",
            type=int,
            default=50,
            help="""Dimension of the hidden layer for the character-level
            features LSTM. Default is 50."""
        )
        group.add_argument(
            "--word_hidden_dim",
            type=int,
            default=50,
            help="""Dimension of the hidden layer for the word-level
            features LSTM. Default is 50."""
        )

    @classmethod
    def cl_init(cls, opts: argparse.Namespace, vocab: Vocab):
        """Initialize an instance of this model from command-line options."""
        crf = ConditionalRandomField(vocab.n_labels)
        char_lstm = CharLSTM(vocab.n_chars, opts.char_hidden_dim)
        return cls(vocab, char_lstm, crf, hidden_dim=opts.word_hidden_dim)
