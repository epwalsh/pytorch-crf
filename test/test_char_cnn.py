"""Tests for CharLSTM class."""


def test_forward(char_cnn, vocab_dataset):
    """Test `CharLSTM.forward()` method."""
    _, dataset = vocab_dataset
    for src, tgt in dataset:
        res = char_cnn(*src[:-2])
        n_words, dim = res.size()
        assert n_words == tgt.size()[0]
        assert dim == char_cnn.output_size
