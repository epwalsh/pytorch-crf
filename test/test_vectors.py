"""Tests for Vocab object."""

from pycrf.io.vectors import load_pretrained


def test_load_pretrained(glove):
    """Test that pretrained GloVe vectors were loaded properly."""
    assert glove["terms_stoi"]["the"] == 0
    assert glove["terms_itos"][0] == "the"
    assert list(glove["vectors"].size()) == [400000, 100]
