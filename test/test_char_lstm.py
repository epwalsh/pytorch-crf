"""Tests for CharLSTM class."""


def test_forward(char_lstm, dataset):
    """Test `CharLSTM.forward()` method."""
    for src, tgt in dataset:
        res = char_lstm(*src[:-1])
        n_words, dim = res.size()
        assert n_words == tgt.size()[0]
        assert dim == char_lstm.output_size
