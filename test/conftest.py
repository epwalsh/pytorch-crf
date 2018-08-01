"""Defines test fixtures."""

import os
from urllib.request import urlopen
import sys
import zipfile

import pytest

from pycrf.eval import ModelStats
from pycrf.io.dataset import Dataset
from pycrf.io.vocab import Vocab
from pycrf.io.vectors import load_pretrained
from pycrf.modules import CharLSTM, LSTMCRF, CharCNN, ConditionalRandomField


VECTOR_CACHE = "test/.vector_cache/glove.6B.100d.txt"
VECTOR_CACHE_ZIP = "test/.vector_cache/glove.6B.zip"
VECTOR_CACHE_URL = "http://nlp.stanford.edu/data/glove.6B.zip"


def extract_vecs(fname, dest):
    zip_ref = zipfile.ZipFile(fname, 'r')
    zip_ref.extractall(dest)
    zip_ref.close()


def download_vecs():
    """Download pretrained GloVe word embeddings from Stanford."""
    u = urlopen(VECTOR_CACHE_URL)
    f = open(VECTOR_CACHE_ZIP, 'wb')
    file_size = int(next((data for name, data in u.getheaders() if name == "Content-Length"), 0))
    file_size_dl = 0
    block_sz = 8192
    while True:
        buffer = u.read(block_sz)
        if not buffer:
            break
    
        file_size_dl += len(buffer)
        f.write(buffer)
        status = f"\r{file_size_dl:10d} {file_size_dl * 100 / file_size:3.2f}%"
        sys.stdout.write(status)
        sys.stdout.flush()
    
    f.close()

    print("Extracting word vectors", flush=True)
    extract_vecs(VECTOR_CACHE_ZIP, "test/.vector_cache")


@pytest.fixture(scope="session")
def glove():
    if not os.path.isfile(VECTOR_CACHE):
        os.mkdir("test/.vector_cache")
        download_vecs()
    vectors, itos, stoi = load_pretrained(VECTOR_CACHE)
    return {
        "vectors": vectors,
        "terms_itos": itos,
        "terms_stoi": stoi,
    }


@pytest.fixture(scope="session")
def vocab_dataset(glove):
    vocab = Vocab(glove["terms_stoi"], glove["terms_itos"])
    dataset = Dataset()
    dataset.load_file("test/data/sample_dataset.txt", vocab)
    return vocab, dataset


@pytest.fixture(scope="session")
def vocab(vocab_dataset):
    return vocab_dataset[0]


@pytest.fixture(scope="session")
def dataset(vocab_dataset):
    return vocab_dataset[1]


@pytest.fixture(scope="session")
def dataset_dev(vocab):
    dataset = Dataset(is_test=True)
    dataset.load_file("test/data/sample_dataset_dev.txt", vocab)
    return dataset


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
    return LSTMCRF(vocab_dataset[0], char_lstm, crf, glove["vectors"], hidden_dim=50)


@pytest.fixture(scope="session")
def get_model_stats(vocab_dataset):

    def _get_model_stats(items):
        model_stats = ModelStats(vocab_dataset[0].labels_itos, 0)
        for labels, preds in items:
            model_stats.update(labels, preds)
        return model_stats

    return _get_model_stats
