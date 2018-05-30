"""Tests for yapycrf.modules.Tagger methods."""


def test_feats(tagger, dataset):
    """Test `Tagger._feats()` method."""
    for src, _ in dataset:
        res = tagger._feats(*src)
        batch_size, sent_length, n_labels = res.size()
        assert batch_size == 1
        assert sent_length == len(src[0])
        assert n_labels == tagger.vocab.n_labels


def test_predict(tagger, dataset):
    """Test `Tagger.predict()` method."""
    for src, _ in dataset:
        res = tagger.predict(*src)
        # Just get first batch since there is only one.
        res = res[0]

        # Output sequence should be same length as input sequence.
        assert len(res) == len(src[0])

        # Each lab index should represent an actual label, i.e. it should be
        # in the set `{0, ..., n_labels - 1}`.
        for lab_idx in res:
            assert lab_idx in tagger.vocab.labels_itos


def test_forward(tagger, dataset):
    """Test `Tagger.forward()` method."""
    for src, tgt in dataset:
        tagger.forward(*src, tgt)
