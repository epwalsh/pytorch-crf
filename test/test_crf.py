"""Tests for CRF module."""

import pytest
import torch


cases = [
    (torch.tensor([[0, 1, 2]]), torch.tensor([3])),
]


@pytest.mark.parametrize("y, lens", cases)
def test_transition_score(crf, y, lens):
    res = crf.transition_score(y, lens)
    trn = crf.transitions
    chk = 0.
    for sent in y:
        prv_lab = crf.start_idx
        for lab in sent:
            lab = lab.item()
            chk += trn[lab][prv_lab]
            prv_lab = lab
        chk += trn[crf.stop_idx][prv_lab]
    assert chk == res
