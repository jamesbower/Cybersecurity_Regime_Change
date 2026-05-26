"""Shared pytest fixtures. Fixtures populated as tasks add them."""
import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(seed=42)
