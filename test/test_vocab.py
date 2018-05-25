"""Tests for Vocab object."""

import pytest
import torch


def test_labels(vocab):
    """Test that the target label attributes and methods work."""
    assert vocab.n_labels == 3
    assert vocab.labels_stoi["O"] == 0
    assert vocab.labels_itos[0] == "O"
    assert vocab.labels_stoi["B-NAME"] == 1
    assert vocab.labels_itos[1] == "B-NAME"
    assert vocab.labels_stoi["I-NAME"] == 2
    assert vocab.labels_itos[2] == "I-NAME"


def test_chars(vocab):
    """Check to make sure the vocab characters are initialized correctly."""
    assert vocab.chars_stoi[vocab.pad_char] == 0
    assert vocab.chars_stoi[vocab.unk_char] == 1
    assert vocab.chars_itos[0] == vocab.pad_char
    assert vocab.chars_itos[1] == vocab.unk_char
    assert vocab.n_chars == len(vocab.chars_stoi) == len(vocab.chars_itos)
    assert vocab.n_chars < 100


def test_words(vocab):
    """Check to make sure the GloVe attr was initialized correctly."""
    assert vocab.n_words > 10000


def test_vectors(vocab):
    """Check that the word embedding vectors look like they're supposed to."""
    assert vocab.glove.vectors[0].size()[0] == vocab.word_vec_dim


cases = [
    ["hi", "there"],
    ["hi"],
]


@pytest.mark.parametrize("sent", cases)
def test_sent2tensor(vocab, sent):
    """Check that Vocab.sent2tensor has the correct output format."""
    char_tensors, word_tensors = vocab.sent2tensor(sent)
    assert isinstance(char_tensors, list)
    assert len(char_tensors) == len(sent)
    assert isinstance(word_tensors, torch.Tensor)
    assert word_tensors.size()[0] == len(sent)
    assert word_tensors.size()[1] == vocab.word_vec_dim
    for i, item in enumerate(char_tensors):
        assert isinstance(item, torch.Tensor)
        assert item.size()[0] == len(sent[i])
        assert item.size()[1] == vocab.n_chars
        assert item.sum().item() == len(sent[i])
