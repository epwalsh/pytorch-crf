"""Tests for pycrf.modules.LSTMCRF methods."""


def test_feats(lstm_crf, dataset):
    """Test `Tagger._feats()` method."""
    for src, _ in dataset:
        res = lstm_crf._feats(*src)
        batch_size, sent_length, n_labels = res.size()
        assert batch_size == 1
        assert sent_length == len(src[0])
        assert n_labels == lstm_crf.vocab.n_labels


def test_predict(lstm_crf, dataset):
    """Test `Tagger.predict()` method."""
    for src, _ in dataset:
        res = lstm_crf.predict(*src)
        # Just get first batch since there is only one.
        res = res[0]

        # Output sequence should be same length as input sequence.
        assert len(res) == len(src[0])

        # Each lab index should represent an actual label, i.e. it should be
        # in the set `{0, ..., n_labels - 1}`.
        for lab_idx in res:
            assert lab_idx in lstm_crf.vocab.labels_itos


def test_forward(lstm_crf, dataset):
    """Test `Tagger.forward()` method."""
    for src, tgt in dataset:
        lstm_crf.forward(*src, tgt)
