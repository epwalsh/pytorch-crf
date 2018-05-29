"""Defines a Bi-LSMT CRF model."""

import torch.nn as nn


class Tagger(nn.Module):
    """
    Bi-LSTM CRF model.

    Parameters
    ----------


    Attributes
    ----------
    vocab : :obj:`yapycrf.io.Vocab`
        The vocab object which contains a dict of known characters and word
        embeddings.

    char_lstm : :obj:`yapycrf.model.CharLSTM`
        The character-level LSTM layer.

    crf : :obj:`yapycrf.model.crf`
        The CRF model.

    """

    def __init__(self, vocab, char_lstm, crf, hidden_dim=100, layers=1,
                 dropout=0,
                 bidirectional=True):
        super(Tagger, self).__init__()

        assert vocab.n_chars == char_lstm.n_chars

        self.vocab = vocab
        self.char_lstm = char_lstm
        self.crf = crf
        self.rnn_output_size = hidden_dim * layers
        if bidirectional:
            self.rnn_output_size *= 2

        self.rnn = nn.LSTM(
            input_size=vocab.word_vec_dim + char_lstm.output_size,
            hidden_size=hidden_dim,
            num_layers=layers,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=True,
        )

        self.rnn_to_crf = nn.Linear(self.rnn_output_size, self.crf.n_labels)

    def predict(self, inputs):
        """
        Outputs the best tag sequence.
        """
        pass

    def forward(self, inputs):
        """
        Computes the negative log-likelihood.
        """
        pass
