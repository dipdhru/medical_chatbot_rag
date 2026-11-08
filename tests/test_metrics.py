import numpy as np
import pytest

from medrag import metrics as m


def test_first_hit_rank():
    assert m.first_hit_rank([False, True, True]) == 2
    assert m.first_hit_rank([False, False]) is None


def test_hit_rate_and_mrr():
    ranks = [1, 3, None, 2]
    assert m.hit_rate(ranks, 1) == 0.25
    assert m.hit_rate(ranks, 3) == 0.75
    assert m.mean_reciprocal_rank(ranks) == pytest.approx((1 + 1 / 3 + 0 + 0.5) / 4)


def test_wilson_interval_stays_in_bounds():
    low, high = m.wilson_interval(10, 10)
    assert 0.65 < low < 1 and high == 1
    assert m.wilson_interval(0, 0) == (0.0, 1.0)


def test_calibration_separates_clean_scores():
    inside, outside = np.array([0.8, 0.9]), np.array([0.3, 0.4])
    threshold = m.calibrate_threshold(inside, outside)
    assert 0.4 < threshold < 0.8
    assert m.rejection_rates(inside, outside, threshold) == (1.0, 1.0)
