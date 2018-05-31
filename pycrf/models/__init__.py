"""Defines trainable models."""

from abc import ABC, abstractmethod
import argparse

import torch.nn as nn

from allennlp.modules.conditional_random_field import ConditionalRandomField
from pycrf.io import Vocab
from pycrf.modules import CharLSTM, Tagger


class Trainable(ABC):
    """
    Base class for all trainable models.
    """

    @staticmethod
    @abstractmethod
    def cl_opts(parser: argparse.ArgumentParser) -> None:
        """
        Add model-specific options to the argument parser.
        """
        pass

    @staticmethod
    @abstractmethod
    def cl_init(opts: argparse.Namespace, vocab: Vocab) -> nn.Module:
        """
        Initialize a model from the command line given the parsed arguments.
        """
        pass


class BiLSTMCRF(Trainable):
    """
    Bi-LSTM CRF model.
    """

    @staticmethod
    def cl_opts(parser: argparse.ArgumentParser) -> None:
        group = parser.add_argument_group("Bi-LSTM CRF options")
        group.add_argument(
            "--char_hidden_dim",
            type=float,
            default=50,
            help="""Dimension of the hidden layer for the character-level
            features LSTM."""
        )
        group.add_argument(
            "--word_hidden_dim",
            type=float,
            default=50,
            help="""Dimension of the hidden layer for the word-level
            features LSTM."""
        )

    @staticmethod
    def cl_init(opts: argparse.Namespace, vocab: Vocab) -> nn.Module:
        crf = ConditionalRandomField(vocab.n_labels)
        char_lstm = CharLSTM(vocab.n_chars, opts.char_hidden_dim)
        tagger = Tagger(vocab, char_lstm, crf, hidden_dim=opts.word_hidden_dim)
        return tagger


MODEL_ALIASES = {
    "bilstm_crf": BiLSTMCRF,
}
