"""
Makes sure evaluation metrics are calculated correctly.
"""

import pytest

from pycrf.eval import \
    iob_to_spans, \
    iobes_to_spans, \
    _detect_label_scheme, \
    Scheme


cases1 = [
    (
        [0, 1, 2],
        {0: "O", 1: "B-A", 2: "I-A"},
        {"A@1@2"}
    ),
    (
        [0, 1, 2, 0],
        {0: "O", 1: "B-A", 2: "I-A"},
        {"A@1@2"}
    ),
    (
        [0, 1, 2, 0, 2],  # still use I- by itself.
        {0: "O", 1: "B-A", 2: "I-A"},
        {"A@1@2", "A@4"}
    ),
    (
        [0, 1, 2, 0, 2, 2],
        {0: "O", 1: "B-A", 2: "I-A"},
        {"A@1@2", "A@4@5"}
    ),
    (
        [0, 1, 2, 0, 2, 2, 1],
        {0: "O", 1: "B-A", 2: "I-A"},
        {"A@1@2", "A@4@5", "A@6"}
    ),
    (
        [0],
        {0: "O", 1: "B-A", 2: "I-A"},
        set()
    ),
]


@pytest.mark.parametrize("tags, lab_itos, chunks", cases1)
def test_iob_to_spans(vocab_dataset, tags, lab_itos, chunks):
    vocab, _ = vocab_dataset
    result = iob_to_spans(tags, lab_itos)
    assert result == chunks


cases2 = [
    (
        [0, 1, 2],
        {0: "O", 1: "B-A", 2: "I-A"},
        {"A@1@2"}
    ),
    (
        [0, 1, 2, 0],
        {0: "O", 1: "B-A", 2: "I-A"},
        {"A@1@2"}
    ),
    (
        [0, 1, 2, 0, 2],  # still use I- by itself.
        {0: "O", 1: "B-A", 2: "I-A"},
        {"A@1@2", "A@4"}
    ),
    (
        [0, 1, 2, 0, 2, 2],
        {0: "O", 1: "B-A", 2: "I-A"},
        {"A@1@2", "A@4@5"}
    ),
    (
        [0, 1, 2, 0, 2, 2, 1],
        {0: "O", 1: "B-A", 2: "I-A"},
        {"A@1@2", "A@4@5", "A@6"}
    ),
    (
        [0],
        {0: "O", 1: "B-A", 2: "I-A"},
        set()
    ),
    (
        [0, 1, 3, 0, 1, 2, 3],
        {0: "O", 1: "B-A", 2: "I-A", 3: "E-A", 4: "S-A"},
        {"A@1@2", "A@4@5@6"}
    ),
    (
        [0, 1, 3, 0, 1, 2, 3, 0, 4],
        {0: "O", 1: "B-A", 2: "I-A", 3: "E-A", 4: "S-A"},
        {"A@1@2", "A@4@5@6", "A@8"}
    ),
    (
        [0, 1, 3, 4],
        {0: "O", 1: "B-A", 2: "I-A", 3: "E-A", 4: "S-A"},
        {"A@1@2", "A@3"}
    ),
    (
        [0, 1, 2, 4],
        {0: "O", 1: "B-A", 2: "I-A", 3: "E-A", 4: "S-A"},
        {"A@1@2", "A@3"}
    ),
]


@pytest.mark.parametrize("tags, lab_itos, chunks", cases2)
def test_iobes_to_spans(vocab_dataset, tags, lab_itos, chunks):
    vocab, _ = vocab_dataset
    result = iobes_to_spans(tags, lab_itos)
    assert result == chunks


cases3 = [
    (
        ["B-A", "I-A", "O"],
        Scheme.IOB
    ),
    (
        ["B-A", "E-A", "O"],
        Scheme.IOBES
    ),
]


@pytest.mark.parametrize("labels, check", cases3)
def test_detect_label_scheme(labels, check):
    assert _detect_label_scheme(labels) == check


cases4 = [
    (["B"]),
]


@pytest.mark.parametrize("labels", cases4)
def test_detect_label_scheme_errors(labels):
    with pytest.raises(ValueError):
        _detect_label_scheme(labels)


cases5 = [
    (
        [
            ([0, 0, 0, 0], [0, 0, 0, 0]),
            ([0, 0, 1, 0], [0, 0, 1, 2]),
            ([1, 2], [1, 0]),
        ],
        0., 0., 0., (4 + 3 + 1) / 10
    ),
    (
        [
            ([0, 0, 0, 0], [0, 0, 0, 0]),
            ([0, 0, 1, 2], [0, 0, 1, 2]),
            ([1, 2], [1, 0]),
        ],
        (1 / 2), (1 / 2), (1 / 2), (4 + 4 + 1) / 10
    ),
    (
        [
            ([0, 0, 0, 0], [1, 2, 0, 0]),
            ([0, 0, 1, 2], [0, 0, 1, 2]),
            ([1, 2], [1, 0]),
        ],
        2 * (1 / 2) * (1 / 3) / (1/2 + 1/3), (1 / 3), (1 / 2), (2 + 4 + 1) / 10
    ),
]


@pytest.mark.parametrize("sentences, f1, precision, recall, accuracy", cases5)
def test_eval_stats(get_model_stats, sentences, f1, precision, recall, accuracy):
    eval_stats = get_model_stats(sentences)
    f1_, precision_, recall_, accuracy_ = eval_stats.score
    assert f1_ == f1
    assert precision_ == precision
    assert recall_ == recall
    assert accuracy_ == accuracy
