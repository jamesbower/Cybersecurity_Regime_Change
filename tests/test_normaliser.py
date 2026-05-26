import pickle
import numpy as np
import pytest
from pard_ssm.data.normaliser import WelfordNormaliser


def test_welford_converges_to_true_mean_var(rng):
    norm = WelfordNormaliser(n_features=3)
    samples = rng.normal(loc=[5.0, -2.0, 0.0], scale=[2.0, 1.0, 3.0], size=(20000, 3))
    for row in samples:
        norm.update(row)
    np.testing.assert_allclose(norm.mean, [5.0, -2.0, 0.0], atol=0.05)
    np.testing.assert_allclose(norm.variance, [4.0, 1.0, 9.0], atol=0.1)


def test_transform_zero_mean_unit_variance(rng):
    norm = WelfordNormaliser(n_features=2)
    samples = rng.normal(loc=[10.0, -3.0], scale=[5.0, 2.0], size=(5000, 2))
    norm.fit(samples)
    out = norm.transform(samples)
    np.testing.assert_allclose(out.mean(axis=0), [0.0, 0.0], atol=0.05)
    np.testing.assert_allclose(out.std(axis=0), [1.0, 1.0], atol=0.05)


def test_transform_does_not_divide_by_zero():
    norm = WelfordNormaliser(n_features=1)
    norm.fit(np.zeros((10, 1)))  # all-zero, variance = 0
    out = norm.transform(np.zeros((3, 1)))
    assert np.all(np.isfinite(out))


def test_pickle_round_trip(rng):
    norm = WelfordNormaliser(n_features=2)
    norm.fit(rng.normal(size=(100, 2)))
    blob = pickle.dumps(norm)
    norm2 = pickle.loads(blob)
    np.testing.assert_array_equal(norm.mean, norm2.mean)
    np.testing.assert_array_equal(norm.M2, norm2.M2)
    assert norm.n == norm2.n
