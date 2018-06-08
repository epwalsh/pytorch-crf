"""Defines test fixtures."""

import pytest

from allennlp.modules.conditional_random_field import ConditionalRandomField
from pycrf.eval import ModelStats
from pycrf.io.dataset import Dataset
from pycrf.io.vocab import Vocab
from pycrf.modules import CharLSTM, LSTMCRF


@pytest.fixture(scope="session")
def vocab_dataset():
    vocab = Vocab(cache="test/.vector_cache")
    dataset = Dataset()
    dataset.load_file("test/data/sample_dataset.txt", vocab)
    return vocab, dataset


@pytest.fixture(scope="session")
def crf(vocab_dataset):
    return ConditionalRandomField(vocab_dataset[0].n_labels)


@pytest.fixture(scope="session")
def char_lstm(vocab_dataset):
    return CharLSTM(vocab_dataset[0].n_chars, 50)


@pytest.fixture(scope="session")
def lstm_crf(vocab_dataset, char_lstm, crf):
    return LSTMCRF(vocab_dataset[0], char_lstm, crf, hidden_dim=50)


@pytest.fixture(scope="session")
def get_model_stats(vocab_dataset):

    def _get_model_stats(items):
        model_stats = ModelStats(vocab_dataset[0].labels_stoi)
        for labels, preds in items:
            model_stats.update(labels, preds)
        model_stats.compile(0)
        return model_stats

    return _get_model_stats
