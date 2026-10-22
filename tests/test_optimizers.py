import unittest

import numpy as np
import pandas as pd

from portfolio_engine.optimizers import (STRATEGIES, apply_cap, efficient_frontier, estimate_cov, hrp,
                                         inverse_vol, ledoit_wolf_cov, max_sharpe, min_variance, risk_parity)
from portfolio_engine.risk import risk_contributions
from tests.helpers import noise_returns, small_returns


class TestOptimizers(unittest.TestCase):
    def setUp(self):
        self.win = small_returns().iloc[-252:]

    def test_all_strategies_return_valid_long_only_weights(self):
        for name, fn in STRATEGIES.items():
            w = fn(self.win, max_weight=0.4, rf=0.02)
            self.assertAlmostEqual(w.sum(), 1.0, places=8, msg=name)
            self.assertGreaterEqual(w.min(), -1e-9, msg=name)
            self.assertLessEqual(w.max(), 0.4 + 1e-6, msg=name)
            self.assertEqual(list(w.index), list(self.win.columns))

    def test_min_variance_beats_equal_weight_in_sample(self):
        cov = estimate_cov(self.win).values
        w_mv = min_variance(self.win, max_weight=1.0).values
        w_ew = np.full(len(cov), 1 / len(cov))
        self.assertLess(w_mv @ cov @ w_mv, w_ew @ cov @ w_ew)

    def test_risk_parity_equalises_risk_contributions(self):
        w = risk_parity(self.win, max_weight=1.0)
        rc = risk_contributions(w, estimate_cov(self.win))
        self.assertLess(rc.max() - rc.min(), 1e-3)

    def test_inverse_vol_weights_ordered_by_volatility(self):
        r = noise_returns()
        w = inverse_vol(r)
        order_vol = r.std().sort_values().index.tolist()
        self.assertEqual(w.sort_values(ascending=False).index.tolist(), order_vol)

    def test_hrp_two_assets_is_inverse_variance(self):
        r = noise_returns(p=2)
        v = np.diag(estimate_cov(r)).copy()
        w = hrp(r)
        self.assertAlmostEqual(w.iloc[0], v[1] / (v[0] + v[1]), places=9)

    def test_hrp_is_positive_and_sums_to_one(self):
        w = hrp(self.win)
        self.assertTrue((w > 0).all())
        self.assertAlmostEqual(w.sum(), 1.0, places=9)

    def test_max_sharpe_prefers_higher_return_for_equal_risk(self):
        rng = np.random.default_rng(3)
        idx = pd.bdate_range("2020-01-01", periods=1000)
        r = pd.DataFrame({"hi": rng.normal(0.0012, 0.01, 1000), "lo": rng.normal(0.0000, 0.01, 1000)}, idx)
        w = max_sharpe(r, max_weight=1.0, mean_shrink=0.0)
        self.assertGreater(w["hi"], w["lo"])

    def test_apply_cap(self):
        w = apply_cap(np.array([0.7, 0.2, 0.05, 0.05]), 0.4)
        self.assertAlmostEqual(w.sum(), 1.0)
        self.assertLessEqual(w.max(), 0.4 + 1e-9)
        with self.assertRaises(ValueError):
            apply_cap(np.array([0.5, 0.5]), 0.3)

    def test_ledoit_wolf_matches_sklearn(self):
        try:
            from sklearn.covariance import LedoitWolf
        except ImportError:
            self.skipTest("scikit-learn not installed")
        X = self.win.values
        ours, shrink = ledoit_wolf_cov(X)
        ref = LedoitWolf().fit(X)
        np.testing.assert_allclose(ours, ref.covariance_, rtol=1e-8, atol=1e-12)
        self.assertAlmostEqual(shrink, ref.shrinkage_, places=8)

    def test_shrinkage_is_a_valid_covariance(self):
        c, s = ledoit_wolf_cov(self.win.values)
        self.assertTrue(0 <= s <= 1)
        self.assertGreater(np.linalg.eigvalsh(c).min(), 0)

    def test_efficient_frontier_is_monotone(self):
        mu = (self.win.mean() * 252).values
        cov = estimate_cov(self.win).values
        f = efficient_frontier(mu, cov, n_points=15, max_weight=1.0)
        self.assertGreater(len(f), 10)
        self.assertTrue((f["return"].diff().dropna() > 0).all())
        self.assertTrue((f["vol"].diff().dropna() > -1e-9).all())


if __name__ == "__main__":
    unittest.main()
