"""Test SGD optimizer."""

from pycrf.optim import SGD


def test_vanilla_decay(lstm_crf):
    """Ensure that vanilla decay schedule works as expected."""
    sgd = SGD(lstm_crf.get_trainable_params((1.0,)),
              decay_rate=0.5,
              decay_start=0,
              conditional_decay=False)

    sgd.epoch_update(10)
    assert sgd.lr[0] == 1.0 / (1 + 0.5 * 1)

    sgd.epoch_update(10)
    assert sgd.lr[0] == 1.0 / (1 + 0.5 * 2)


def test_cyclic_decay(lstm_crf):
    """Ensure cyclic LR annealing schedule works as expected."""
    sgd = SGD(lstm_crf.get_trainable_params((1.0,)),
              cyclic=True,
              cycle_len=30)

    lrs = [sgd.lr[0]]
    for i in range(150):
        sgd.epoch_update(10)
        lrs.append(sgd.lr[0])

    min_lr = min(lrs)
    max_lr = max(lrs)
    assert min_lr == 0.0027390523158632996
    assert max_lr == 1.0

    for i, lr in enumerate(lrs):
        if i % 30 == 0:
            assert lr == max_lr
        elif i % 30 == 29:
            assert lr == min_lr


def test_variable_length_cyclic_decay(lstm_crf):
    """Cyclic LR rate with cycle_mult set."""
    sgd = SGD(lstm_crf.get_trainable_params((1.0,)),
              cyclic=True,
              cycle_len=30,
              cycle_mult=2)

    lrs = [sgd.lr[0]]
    for i in range(210):
        sgd.epoch_update(10)
        lrs.append(sgd.lr[0])

    assert max(lrs) == 1.0

    # First cycle, 30 iterations.
    assert lrs[0] == 1.0
    assert lrs[29] == 0.0027390523158632996

    # Second cycle, 60 iterations.
    assert lrs[30] == 1.0
    assert lrs[89] ==  0.0006852326227130834

    # Third cycle, 120 iterations.
    assert lrs[90] == 1.0
    assert lrs[209] == 0.00017133751222137006


def test_variable_length_cyclic_decay_non_int_multiplier(lstm_crf):
    """Now try with a non-int multiplier."""
    sgd = SGD(lstm_crf.get_trainable_params((1.0,)),
              cyclic=True,
              cycle_len=30,
              cycle_mult=1.5)

    lrs = [sgd.lr[0]]
    for i in range(100):
        sgd.epoch_update(10)
        lrs.append(sgd.lr[0])

    assert max(lrs) == 1.0

    # First cycle, 30 iterations.
    assert lrs[0] == 1.0
    assert lrs[29] == 0.0027390523158632996

    # Second cycle, 45 iterations.
    assert lrs[30] == 1.0
    assert lrs[74] == 0.0012179748700879012
