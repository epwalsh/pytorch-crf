"""Defines a Bi-LSMT CRF model."""

import argparse
from typing import List, Tuple

import torch
import torch.nn as nn

from pycrf.io import Vocab
from pycrf.nn.utils import sequence_mask
from .crf import ConditionalRandomField


class LSTMCRF(nn.Module):
    """
    (Bi-)LSTM CRF model.

    Parameters
    ----------
    vocab : pycrf.io.Vocab
        The vocab object which contains a dict of known characters and word
        embeddings.

    char_feats_layer : pycrf.model.CharLSTM
        The character-level feature-generating layer.

    crf : allennlp.modules.conditional_random_field.ConditionalRandomField
        The CRF model.

    pretrained_word_vecs : torch.Tensor
        Pre-trained word vectors with shape ``[vocab_size x embedding_dimension]``.

    hidden_dim : int, optional (default: 100)
        The hidden dimension of the recurrent layer.

    layers : int, optional (default: 1)
        The number of layers of cells in the recurrent layer.

    dropout : float, optional (default: 0.)
        The dropout probability for the recurrent layer.

    bidirectional : bool, optional (default: True)
        If True, bidirectional recurrent layer is used, otherwise single
        direction.

    freeze_embeddings : bool, optional (default: True)
        If True, word embeddings will not be updated during training.

    sent_context_embedding : int, optional (default: 5)
        Size of sentence-level context embedding.

    Attributes
    ----------
    vocab : pycrf.io.Vocab
        The vocab object which contains a dict of known characters and word
        embeddings.

    dropout : torch.nn.Dropout
        Dropouts applied to various layers.

    word_embedding : torch.nn.Embedding
        The word vector embedding layer.

    char_feats_layer : torch.nn.Module
        The character-level feature generating layer.

    crf : allennlp.modules.conditional_random_field.ConditionalRandomField
        The CRF model.

    rnn_output_size : int
        The output dimension of the recurrent layer.

    rnn : nn.Module
        The recurrent layer of the network.

    rnn_to_crf : nn.Module
        The linear layer that maps the hidden states from the recurrent layer
        to the label space.

    sent_context_embedding : nn.Module
        Embedding for sentence-level context.

    """

    def __init__(self,
                 vocab: Vocab,
                 char_feats_layer: torch.nn.Module,
                 crf: ConditionalRandomField,
                 pretrained_word_vecs: torch.Tensor,
                 sent_context_embedding: int = 5,
                 hidden_dim: int = 100,
                 layers: int = 1,
                 dropout: float = 0.,
                 bidirectional: bool = True,
                 freeze_embeddings: bool = True) -> None:
        super(LSTMCRF, self).__init__()

        assert vocab.n_chars == char_feats_layer.n_chars
        assert vocab.n_labels == crf.num_tags
        assert vocab.n_words == pretrained_word_vecs.size()[0]

        self.vocab = vocab
        self.dropout = nn.Dropout(p=dropout) if dropout else None

        # Layer for generating character-level word features.
        self.char_feats_layer = char_feats_layer

        # Word-embedding layer.
        self.word_vec_dim = pretrained_word_vecs.size()[1]
        self.word_embedding = \
            nn.Embedding.from_pretrained(pretrained_word_vecs, freeze=freeze_embeddings)

        # Sentence-level context embedding.
        n_contexts = len(self.vocab.sent_context_stoi)
        if n_contexts > 1:
            self.sent_context_embedding = nn.Embedding(
                n_contexts, sent_context_embedding)
        else:
            self.sent_context_embedding = None

        # Recurrent layer. Takes as input the concatenation of the
        # char_feats_layer final hidden state and pre-trained embedding for
        # each word. The dimension of the output is given by
        # self.rnn_output_size (see below).
        rnn_input_size = self.word_vec_dim + char_feats_layer.output_size
        if self.sent_context_embedding:
            rnn_input_size += sent_context_embedding
        self.rnn = nn.LSTM(
            input_size=rnn_input_size,
            hidden_size=hidden_dim,
            num_layers=layers,
            bidirectional=bidirectional,
            dropout=dropout if layers > 1 else 0,
            batch_first=True,
        )

        # This is the size of the recurrent layer's output (see self.rnn).
        self.rnn_output_size = hidden_dim
        if bidirectional:
            self.rnn_output_size *= 2

        # Linear layer that takes the output from the recurrent layer and each
        # time step and transforms into scores for each label.
        self.rnn_to_crf = nn.Linear(self.rnn_output_size, self.vocab.n_labels)

        # Conditional Random Field that maps word features into labels.
        self.crf = crf

    def _feats(self,
               words: torch.Tensor,
               word_lengths: torch.Tensor,
               word_indices: torch.Tensor,
               word_idxs: torch.Tensor,
               sent_context: torch.Tensor) -> torch.Tensor:
        """
        Generate features for the CRF from input.

        First we generate a vector for each word by running each word
        char-by-char through the char_feats_layer and then concatenating those
        final hidden states for each word with the pre-trained embedding of the
        word. That word vector is than ran through the RNN and then the linear
        layer.

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

        word_idxs : torch.Tensor
            Word indices with shape ``[sent_length]``.

        sent_context : torch.Tensor
            Sentence context.

        Returns
        -------
        torch.Tensor
            ``[batch_size x sent_length x crf.n_labels]``

        """
        sent_length = words.size(0)

        # Run each word character-by-character through the CharLSTM to generate
        # character-level word features.
        char_feats = self.char_feats_layer(words, word_lengths, word_indices)
        # char_feats: ``[sent_length x char_feats_layer.output_size]``

        # Embed words into vector space.
        word_embs = self.word_embedding(word_idxs)
        # word_embs: ``[sent_length x self.word_vec_dim]``

        # Concatenate the character-level word features, word embeddings, and
        # sentence-level context embedding.
        if self.sent_context_embedding:
            # Embed sentence-level context.
            context_embs = self.sent_context_embedding(sent_context)
            # context_embs: ``[1 x sent_context_embedding_size]``

            context_embs = context_embs.expand(sent_length, -1)
            # context_embs: ``[sent_length x sent_context_embedding_size]``.

            word_feats = torch.cat([char_feats, word_embs, context_embs], dim=-1)
            # word_feats: ``[sent_length x (char_feats_layer.output_size +
            #                               self.word_vec_dim +
            #                               sent_context_embedding_size)]``
        else:
            word_feats = torch.cat([char_feats, word_embs], dim=-1)
            # word_feats: ``[sent_length x (char_feats_layer.output_size +
            #                               self.word_vec_dim)]``

        # Add a fake batch dimension.
        word_feats = word_feats.unsqueeze(0)
        # word_feats: ``[1 x sent_length x (char_feats_layer.output_size +
        #                                   self.word_vec_dim)]``

        # Apply dropout.
        if self.dropout:
            word_feats = self.dropout(word_feats)

        # Run word features through the LSTM.
        lstm_feats, _ = self.rnn(word_feats)
        # lstm_feats: ``[1 x sent_length x rnn_output_size]``

        # Apply dropout.
        if self.dropout:
            lstm_feats = self.dropout(lstm_feats)

        # Run recurrent output through linear layer to generate the by-label
        # features.
        feats = self.rnn_to_crf(lstm_feats)
        # feats: ``[1 x sent_length x crf.n_labels]``

        return feats

    def predict(self,
                words: torch.Tensor,
                word_lengths: torch.Tensor,
                word_indices: torch.Tensor,
                word_idxs: torch.Tensor,
                sent_context: torch.Tensor,
                lens: torch.Tensor = None) -> List[Tuple[List[int], float]]:
        # pylint: disable=not-callable
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

        word_idxs : torch.Tensor
            Word indices with shape ``[sent_length]``.

        sent_context : torch.Tensor
            Sentence context.

        lens : torch.Tensor, optional (default: None)
            Gives the length of each sentence in the batch ``[batch_size]``.

        Returns
        -------
        List[List[int]]
            The best path for each sentence in the batch.

        """
        if lens is None:
            lens = torch.tensor([words.size(0)], device=words.device)
        mask = sequence_mask(lens)

        # Gather word feats.
        feats = self._feats(
            words, word_lengths, word_indices, word_idxs, sent_context)
        # feats: ``[1 x sent_length x n_labels]``

        # Run features through Viterbi decode algorithm.
        preds = self.crf.viterbi_tags(feats, mask)

        return preds

    def forward(self,
                words: torch.Tensor,
                word_lengths: torch.Tensor,
                word_indices: torch.Tensor,
                word_idxs: torch.Tensor,
                sent_context: torch.Tensor,
                labs: torch.Tensor,
                lens: torch.Tensor = None) -> torch.Tensor:
        # pylint: disable=arguments-differ,not-callable
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

        word_idxs : torch.Tensor
            Word indices with shape ``[sent_length]``.

        sent_context : torch.Tensor
            Sentence context.

        labs : torch.Tensor
            Corresponding target label sequence with shape ``[sent_length]``.

        lens : torch.Tensor, optional (default: None)
            Gives the length of each sentence in the batch ``[batch_size]``.

        Returns
        -------
        torch.Tensor
            The negative log-likelihood evaluated at the inputs.

        """
        if lens is None:
            lens = torch.tensor([words.size(0)], device=words.device)
        mask = sequence_mask(lens)

        # Fake batch dimension for ``labs``.
        labs = labs.unsqueeze(0)
        # labs: ``[1 x sent_length]``

        # Gather word feats.
        feats = self._feats(
            words, word_lengths, word_indices, word_idxs, sent_context)
        # feats: ``[1 x sent_length x n_labels]``

        loglik = self.crf(feats, labs, mask=mask)

        return -1. * loglik

    @staticmethod
    def cl_opts(parser: argparse.ArgumentParser, require=True) -> None:
        # pylint: disable=unused-argument
        """Define command-line options specific to this model."""
        group = parser.add_argument_group("Bi-LSTM CRF options")
        group.add_argument(
            "--hidden-dim",
            type=int,
            default=50,
            help="""Dimension of the hidden layer for the word-level
            features LSTM. Default is 50."""
        )
        group.add_argument(
            "--update-word-embeddings",
            action="store_true",
            help="""Allow pretrained word embeddings to update throughout the
            training process."""
        )
        group.add_argument(
            "--sent-context-dim",
            type=int,
            default=5,
            help="""Embedding size of sentence-level context. Default is 5."""
        )

    @classmethod
    def cl_init(cls,
                opts: argparse.Namespace,
                vocab: Vocab,
                char_feats_layer: torch.nn.Module,
                word_vecs: torch.Tensor):
        """Initialize an instance of this model from command-line options."""
        crf = ConditionalRandomField(vocab.n_labels)
        return cls(vocab, char_feats_layer, crf, word_vecs,
                   hidden_dim=opts.hidden_dim,
                   dropout=opts.dropout,
                   freeze_embeddings=not opts.update_word_embeddings,
                   sent_context_embedding=opts.sent_context_dim)
