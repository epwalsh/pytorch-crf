"""Defines test fixtures."""

import pytest

from allennlp.modules.conditional_random_field import ConditionalRandomField
from pycrf.io.dataset import Dataset
from pycrf.io.vocab import Vocab
from pycrf.modules import CharLSTM, LSTMCRF


LABELS = ["O", "B-NAME", "I-NAME"]


@pytest.fixture(scope="session")
def vocab():
    return Vocab(LABELS, cache="test/.vector_cache")


@pytest.fixture(scope="session")
def dataset(vocab):
    dataset = Dataset()
    dataset.load_file("test/data/sample_dataset.txt", vocab)
    return dataset


@pytest.fixture(scope="session")
def crf():
    return ConditionalRandomField(len(LABELS))


@pytest.fixture(scope="session")
def char_lstm(vocab):
    return CharLSTM(vocab.n_chars, 50)


@pytest.fixture(scope="session")
def lstm_crf(vocab, char_lstm, crf):
    return LSTMCRF(vocab, char_lstm, crf, hidden_dim=50)
