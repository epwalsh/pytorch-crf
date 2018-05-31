"""Tests for CharLSTM class."""


def test_forward(char_lstm, dataset):
    """Test `CharLSTM.forward()` method."""
    for src, _ in dataset:
        chars = src[0]
        res = char_lstm(chars)
        n_words, dim = res.size()
        assert n_words == len(chars)
        assert dim == char_lstm.output_size
