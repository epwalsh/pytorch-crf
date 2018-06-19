"""Tests for Vocab object."""

import pytest
import torch

from pycrf.nn.utils import assert_equal


def test_labels(vocab_dataset):
    """Test that the target label attributes and methods work."""
    vocab = vocab_dataset[0]
    assert vocab.n_labels == 3
    assert vocab.labels_stoi["O"] == 0
    assert vocab.labels_itos[0] == "O"
    assert vocab.labels_stoi["B-NAME"] == 1
    assert vocab.labels_itos[1] == "B-NAME"
    assert vocab.labels_stoi["I-NAME"] == 2
    assert vocab.labels_itos[2] == "I-NAME"


def test_chars(vocab_dataset):
    """Check to make sure the vocab characters are initialized correctly."""
    vocab = vocab_dataset[0]
    assert vocab.chars_stoi[vocab.pad_char] == 0
    assert vocab.chars_stoi[vocab.unk_char] == 1
    assert vocab.chars_itos[0] == vocab.pad_char
    assert vocab.chars_itos[1] == vocab.unk_char
    assert vocab.n_chars == len(vocab.chars_stoi) == len(vocab.chars_itos)
    assert vocab.n_chars < 100


def test_words(vocab_dataset):
    """Check to make sure word dicts were initialized correctly."""
    vocab = vocab_dataset[0]
    assert vocab.n_words > 10000
    assert vocab.words_itos[vocab.words_stoi["hi"]] == "hi"


cases = [
    ["hi", "there"],
    ["hi", "there", "what", "is", "your", "name", "?"],
    ["hi"],
]


@pytest.mark.parametrize("sent", cases)
def test_sent2tensor(vocab_dataset, sent):
    """Check that Vocab.sent2tensor has the correct output format."""
    vocab = vocab_dataset[0]
    char_tensors, word_lengths, word_idxs, word_tensors = \
        vocab.sent2tensor(sent)

    check_lens = [len(s) for s in sent]
    check_sorted_lens = sorted(check_lens, reverse=True)
    check_idxs = sorted(range(len(sent)), reverse=True, key=lambda i: check_lens[i])

    # Verify sizes.
    assert isinstance(char_tensors, torch.Tensor)
    assert list(char_tensors.size()) == [len(sent), max(check_lens)]
    assert isinstance(word_tensors, torch.Tensor)
    assert list(word_tensors.size()) == [len(sent)]

    # Verify order of word lengths and idxs.
    assert_equal(word_lengths, torch.tensor(check_sorted_lens))
    assert_equal(word_idxs, torch.tensor(check_idxs))

    for i, word_tensor in enumerate(char_tensors):
        check_word = sent[check_idxs[i]]
        check_word_tensor = torch.tensor(
            [vocab.chars_stoi[c] for c in check_word] + [0] * (max(check_lens) - len(check_word))
        )
        assert_equal(word_tensor, check_word_tensor)
