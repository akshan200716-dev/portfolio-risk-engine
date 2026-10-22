import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from portfolio_engine import metrics
from portfolio_engine.config import UNIVERSE
from portfolio_engine.data import _correlation, generate_synthetic_prices, load_prices, to_returns


def series(vals):
    return pd.Series(vals, index=pd.bdate_range("2020-01-01", periods=len(vals)))


class TestMetrics(unittest.TestCase):
    def test_cagr_constant_return(self):
        r = series([0.001] * 252)
        self.assertAlmostEqual(metrics.cagr(r), 1.001 ** 252 - 1)

    def test_sharpe_zero_mean_and_sign(self):
        r = series([0.01, -0.01] * 100)
        self.assertAlmostEqual(metrics.sharpe(r), 0.0, places=9)
        self.assertGreater(metrics.sharpe(r + 0.002), 0)

    def test_sharpe_matches_definition(self):
        r = series(np.random.default_rng(0).normal(0.0005, 0.01, 1000).tolist())
        expected = (r.mean() - 0.02 / 252) / r.std() * np.sqrt(252)
        self.assertAlmostEqual(metrics.sharpe(r, 0.02), expected)

    def test_sortino_ignores_upside_volatility(self):
        a = series([0.01, -0.01, 0.01, -0.01] * 50)
        b = series([0.03, -0.01, 0.03, -0.01] * 50)
        self.assertGreater(metrics.sortino(b), metrics.sortino(a))

    def test_max_drawdown(self):
        self.assertAlmostEqual(metrics.max_drawdown(series([0.1, -0.5, 0.2])), -0.5)

    def test_benchmark_relative_stats_against_itself(self):
        r = series(np.random.default_rng(1).normal(0.0004, 0.01, 800).tolist())
        s = metrics.performance_summary(r, r)
        self.assertAlmostEqual(s["Beta vs benchmark"], 1.0)
        self.assertAlmostEqual(s["Alpha (ann.)"], 0.0, places=9)
        self.assertAlmostEqual(s["Tracking error"], 0.0, places=9)

    def test_bootstrap_interval_brackets_point_estimate(self):
        r = series(np.random.default_rng(2).normal(0.0005, 0.01, 2000).tolist())
        lo, hi = metrics.sharpe_bootstrap_ci(r, n_boot=500)
        self.assertLess(lo, metrics.sharpe(r))
        self.assertGreater(hi, metrics.sharpe(r))

    def test_bootstrap_is_reproducible(self):
        r = series(np.random.default_rng(3).normal(0, 0.01, 500).tolist())
        self.assertEqual(metrics.sharpe_bootstrap_ci(r, n_boot=200, seed=4), metrics.sharpe_bootstrap_ci(r, n_boot=200, seed=4))


class TestData(unittest.TestCase):
    def test_synthetic_is_deterministic_per_seed(self):
        a = generate_synthetic_prices(start="2015-01-01", end="2016-12-31", seed=3)
        b = generate_synthetic_prices(start="2015-01-01", end="2016-12-31", seed=3)
        c = generate_synthetic_prices(start="2015-01-01", end="2016-12-31", seed=4)
        pd.testing.assert_frame_equal(a, b)
        self.assertFalse(a.equals(c))

    def test_regime_correlation_matrices_are_valid(self):
        tickers = list(UNIVERSE)
        for regime in (0, 1, 2):
            C = _correlation(tickers, regime)
            np.testing.assert_allclose(np.diag(C), 1.0)
            np.testing.assert_allclose(C, C.T)
            self.assertGreater(np.linalg.eigvalsh(C).min(), 0)

    def test_crisis_raises_equity_correlation_and_2022_flips_stock_bond_sign(self):
        t = list(UNIVERSE)
        calm, crisis, shock = (_correlation(t, r) for r in (0, 1, 2))
        i, j, k = t.index("SPY"), t.index("EEM"), t.index("TLT")
        self.assertGreater(crisis[i, j], calm[i, j])
        self.assertLess(crisis[i, k], 0.05)
        self.assertGreater(shock[i, k], 0.2)

    def test_prices_positive_and_calibrated_to_targets(self):
        px = generate_synthetic_prices()
        self.assertTrue((px > 0).all().all())
        n = len(to_returns(px))
        cagr = (px.iloc[-1] / px.iloc[0]) ** (252 / n) - 1
        self.assertAlmostEqual(cagr["SPY"], 0.095, delta=0.005)

    def test_csv_round_trip(self):
        px = generate_synthetic_prices(start="2015-01-01", end="2015-12-31")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "px.csv"
            px.to_csv(p)
            out, label = load_prices(source="csv", csv_path=p)
        self.assertEqual(label, "csv:px.csv")
        pd.testing.assert_frame_equal(out, px, check_freq=False, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
