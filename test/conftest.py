"""Defines test fixtures."""

import pytest

from yapycrf.io.dataset import Dataset
from yapycrf.io.vocab import Vocab


@pytest.fixture(scope="session")
def vocab():
    """Create a vocab object as a fixture, so use across all test sessions."""
    return Vocab(["O", "B-NAME", "I-NAME"], cache="test/.vector_cache")


@pytest.fixture(scope="session")
def dataset(vocab):
    """Initialize a Dataset object as a fixture for use across all tests."""
    dataset = Dataset()
    dataset.load_file("test/data/sample_dataset.txt", vocab)
    return dataset
