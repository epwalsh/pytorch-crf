"""Test SGD optimizer."""

import pytest

from pycrf.optim import SGD


cyclic_lr_cases = [
    (30, 100, 10, 30, 1.0, [(0, 1.0),
                            (15, 0.5000000000000001),
                            (29, 0.0027390523158632996)]),
    (10, 100, 10, 1, 2.0, [(0, 1.0),
                           (1, 1.0),
                           (2, 0.5),
                           (3, 1.0)]),
    (30, 100, 10, None, 1.0, [(0, 1.0),
                              (15, 1.0),
                              (29, 1.0)]),
    (60, 100, 10, 30, 1.0, [(0, 1.0),
                            (15, 0.5000000000000001),
                            (29, 0.0027390523158632996),
                            (30, 1.0),
                            (45, 0.5000000000000001),
                            (59, 0.0027390523158632996)]),
    (100, 1, 1, 30, 1.5, [(0, 1.0),
                          (29, 0.0027390523158632996),
                          (30, 1.0),
                          (74, 0.0012179748700879012)]),
    (210, 1, 1, 30, 2, [(0, 1.0),
                        (29, 0.0027390523158632996),
                        (30, 1.0),
                        (89, 0.0006852326227130834),
                        (90, 1.0),
                        (209, 0.00017133751222137006)]),
    (150, 1, 1, 30, 1, [(0, 1.0),
                        (29, 0.0027390523158632996),
                        (30, 1.0),
                        (59, 0.0027390523158632996),
                        (60, 1.0),
                        (89, 0.0027390523158632996),
                        (90, 1.0)]),
]


@pytest.mark.parametrize("epochs, training_size, batch_size, cycle_len, cycle_mult, checks",
                         cyclic_lr_cases)
def test_cyclic_lr(lstm_crf, training_size, batch_size, epochs, cycle_len, cycle_mult, checks):
    """Test cyclic cosine annealing when updated once per iteration (mini-batch)."""
    sgd = SGD(lstm_crf.get_trainable_params((1.0,)),
              cycle_len=cycle_len,
              cycle_mult=cycle_mult)

    sgd.epoch_prepare(training_size, batch_size)

    lrs = []
    for e in range(epochs):
        print(f"{e} -> {sgd.lr[0]}")
        lrs.append(sgd.lr[0])
        i = 0
        while i < training_size:
            for _ in range(min([batch_size, training_size - i])):
                i += 1
            sgd.iteration_update(i)
        sgd.epoch_update(10)

    for it, lr in checks:
        assert lrs[it] == lr
