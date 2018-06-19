"""Defines test fixtures."""

import pytest
import torchtext

from allennlp.modules.conditional_random_field import ConditionalRandomField
from pycrf.eval import ModelStats
from pycrf.io.dataset import Dataset
from pycrf.io.vocab import Vocab
from pycrf.modules import CharLSTM, LSTMCRF, CharCNN


@pytest.fixture(scope="session")
def glove():
    return torchtext.vocab.GloVe(name="6B", dim=100, cache="test/.vector_cache")


@pytest.fixture(scope="session")
def vocab_dataset(glove):
    vocab = Vocab(glove.stoi, glove.itos)
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
def char_cnn(vocab_dataset):
    return CharCNN(vocab_dataset[0].n_chars, 30)


@pytest.fixture(scope="session")
def lstm_crf(vocab_dataset, char_lstm, crf, glove):
    return LSTMCRF(vocab_dataset[0], char_lstm, crf, glove.vectors, hidden_dim=50)


@pytest.fixture(scope="session")
def get_model_stats(vocab_dataset):

    def _get_model_stats(items):
        model_stats = ModelStats(vocab_dataset[0].labels_itos, 0)
        for labels, preds in items:
            model_stats.update(labels, preds)
        return model_stats

    return _get_model_stats
