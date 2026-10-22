import unittest

import numpy as np
import pandas as pd

from portfolio_engine import risk
from tests.helpers import noise_returns


class TestVarEs(unittest.TestCase):
    def test_historical_var_on_uniform_grid(self):
        r = np.linspace(-0.10, 0.10, 2001)
        self.assertAlmostEqual(risk.historical_var(r, 0.95), 0.09, places=3)

    def test_es_exceeds_var(self):
        r = np.random.default_rng(0).standard_t(4, 20000) * 0.01
        for a in (0.95, 0.99):
            self.assertGreater(risk.historical_es(r, a), risk.historical_var(r, a))
            self.assertGreater(risk.parametric_es(r, a), risk.parametric_var(r, a))

    def test_parametric_var_matches_normal_quantile(self):
        r = np.random.default_rng(1).normal(0, 0.01, 200000)
        self.assertAlmostEqual(risk.parametric_var(r, 0.95), 1.6449 * 0.01, delta=2e-4)

    def test_cornish_fisher_equals_normal_for_gaussian_data(self):
        r = np.random.default_rng(2).normal(0, 0.01, 200000)
        self.assertAlmostEqual(risk.cornish_fisher_var(r, 0.99), risk.parametric_var(r, 0.99), delta=3e-4)

    def test_monte_carlo_close_to_parametric_for_large_df(self):
        r = noise_returns(n=2000)
        w = pd.Series(0.25, index=r.columns)
        mc, _ = risk.monte_carlo_var_es(w, r, 0.95, n_sims=300000, df=200)
        port = r @ w
        self.assertAlmostEqual(mc, risk.parametric_var(port.values, 0.95), delta=6e-4)


class TestDrawdownAndDecomposition(unittest.TestCase):
    def test_max_drawdown_known_path(self):
        r = pd.Series([0.10, -0.50, 0.20], index=pd.bdate_range("2020-01-01", periods=3))
        self.assertAlmostEqual(risk.drawdown_series(r).min(), -0.5)
        t = risk.drawdown_table(r)
        self.assertAlmostEqual(t.loc[0, "Depth"], -0.5)
        self.assertEqual(t.loc[0, "Recovery"], "not recovered")

    def test_drawdown_table_finds_recovery(self):
        r = pd.Series([0.1, -0.2, 0.1, 0.2, 0.05], index=pd.bdate_range("2020-01-01", periods=5))
        t = risk.drawdown_table(r)
        self.assertNotEqual(t.loc[0, "Recovery"], "not recovered")

    def test_risk_contributions_sum_to_one(self):
        cov = pd.DataFrame(np.cov(noise_returns().values, rowvar=False) * 252, columns=list("ABCD"), index=list("ABCD"))
        rc = risk.risk_contributions(pd.Series([0.4, 0.3, 0.2, 0.1], index=cov.index), cov)
        self.assertAlmostEqual(rc.sum(), 1.0)

    def test_equal_vol_uncorrelated_assets_split_risk_evenly(self):
        cov = pd.DataFrame(np.diag([0.04, 0.04]), index=["X", "Y"], columns=["X", "Y"])
        rc = risk.risk_contributions(pd.Series([0.5, 0.5], index=["X", "Y"]), cov)
        np.testing.assert_allclose(rc.values, [0.5, 0.5])


class TestStress(unittest.TestCase):
    def test_hypothetical(self):
        w = pd.Series({"A": 0.6, "B": 0.4})
        out = risk.stress_hypothetical(w, {"crash": {"A": -0.30, "B": 0.10}})
        self.assertAlmostEqual(out["crash"], -0.14)

    def test_historical_buy_and_hold(self):
        idx = pd.bdate_range("2020-01-01", periods=20)
        r = pd.DataFrame({"A": 0.01, "B": 0.0}, index=idx)
        out = risk.stress_historical(r, pd.Series({"A": 0.5, "B": 0.5}),
                                     {"win": (str(idx[0].date()), str(idx[9].date()))})
        self.assertAlmostEqual(out["win"], 0.5 * (1.01 ** 10 - 1))

    def test_missing_window_is_skipped(self):
        r = noise_returns()
        out = risk.stress_historical(r, pd.Series(0.25, index=r.columns), {"old": ("1990-01-01", "1990-03-01")})
        self.assertEqual(len(out), 0)


class TestVarValidation(unittest.TestCase):
    def test_kupiec_accepts_correct_rate_and_rejects_wrong_rate(self):
        lr, p = risk.kupiec_pof(10, 1000, 0.99)
        self.assertAlmostEqual(lr, 0.0, places=9)
        self.assertAlmostEqual(p, 1.0, places=6)
        self.assertLess(risk.kupiec_pof(30, 1000, 0.99)[1], 0.001)
        self.assertLess(risk.kupiec_pof(0, 1000, 0.99)[1], 0.01)

    def test_basel_zones(self):
        self.assertEqual([risk.basel_zone(k) for k in (0, 4, 5, 9, 10)],
                         ["Green", "Green", "Yellow", "Yellow", "Red"])

    def test_rolling_var_uses_only_past_data(self):
        r = pd.Series(np.random.default_rng(0).normal(0, 0.01, 400), index=pd.bdate_range("2020-01-01", periods=400))
        r2 = r.copy()
        r2.iloc[300] = -0.5                                    # shock on day 300
        for m in ("Historical", "Parametric (normal)", "Cornish-Fisher"):
            a, b = risk.rolling_var(r, m), risk.rolling_var(r2, m)
            self.assertEqual(a.iloc[300], b.iloc[300], msg=f"{m} forecast for day 300 must not see day 300")
            self.assertNotEqual(a.iloc[301], b.iloc[301], msg=f"{m} forecast for day 301 should see day 300")

    def test_backtest_flags_normal_var_on_fat_tails(self):
        rng = np.random.default_rng(5)
        r = pd.Series(rng.standard_t(3, 6000) * 0.006, index=pd.bdate_range("2000-01-01", periods=6000))
        summary, _ = risk.var_backtest(r, alpha=0.99)
        row = summary.set_index("Method").loc["Parametric (normal)"]
        self.assertGreater(row["Exceptions"], row["Expected"])


if __name__ == "__main__":
    unittest.main()
