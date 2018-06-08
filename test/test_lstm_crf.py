"""Tests for pycrf.modules.LSTMCRF methods."""


def test_feats(lstm_crf, vocab_dataset):
    """Test `Tagger._feats()` method."""
    dataset = vocab_dataset[1]
    for src, _ in dataset:
        res = lstm_crf._feats(*src)
        batch_size, sent_length, n_labels = res.size()
        assert batch_size == 1
        assert sent_length == len(src[0])
        assert n_labels == lstm_crf.vocab.n_labels


def test_predict(lstm_crf, vocab_dataset):
    """Test `Tagger.predict()` method."""
    dataset = vocab_dataset[1]
    for src, _ in dataset:
        preds = lstm_crf.predict(*src)
        # Just get first batch since there is only one.
        preds = preds[0]
        tags, _ = preds

        # Output sequence should be same length as input sequence.
        assert len(tags) == len(src[0])

        # Each lab index should represent an actual label, i.e. it should be
        # in the set `{0, ..., n_labels - 1}`.
        for lab_idx in tags:
            assert lab_idx in lstm_crf.vocab.labels_itos


def test_forward(lstm_crf, vocab_dataset):
    """Test `Tagger.forward()` method."""
    dataset = vocab_dataset[1]
    for src, tgt in dataset:
        lstm_crf.forward(*src, tgt)
