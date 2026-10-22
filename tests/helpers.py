"""Shared fixtures for the test-suite."""
import numpy as np
import pandas as pd

from portfolio_engine.data import generate_synthetic_prices, to_returns

_CACHE = {}


def small_returns(start="2012-01-02", end="2016-12-30", seed=1) -> pd.DataFrame:
    key = (start, end, seed)
    if key not in _CACHE:
        _CACHE[key] = to_returns(generate_synthetic_prices(start=start, end=end, seed=seed))
    return _CACHE[key].copy()


def noise_returns(n=600, p=4, seed=0, vols=(0.01, 0.02, 0.005, 0.015)) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(rng.normal(0.0003, vols[:p], (n, p)), idx, columns=list("ABCD")[:p])
