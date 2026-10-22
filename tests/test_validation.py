import unittest

import numpy as np
import pandas as pd

from portfolio_engine.validation import validate_prices


def clean_prices(n=500, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(100 * np.cumprod(1 + rng.normal(0.0003, 0.01, (n, 3)), axis=0), idx, columns=["A", "B", "C"])


def status(report, name_fragment):
    row = report.checks[report.checks["check"].str.contains(name_fragment, regex=False)]
    return row["status"].iloc[0]


class TestValidation(unittest.TestCase):
    def test_clean_data_passes_everything_material(self):
        rep = validate_prices(clean_prices())
        self.assertTrue(rep.passed)
        self.assertEqual(status(rep, "Missing"), "PASS")
        self.assertEqual(status(rep, "Stale"), "PASS")
        self.assertEqual(len(rep.cleaned), 500)

    def test_detects_extreme_move_and_logs_it(self):
        px = clean_prices()
        px.iloc[200:, 0] *= 1.6                       # +60% in a day
        rep = validate_prices(px)
        self.assertEqual(status(rep, "Extreme"), "WARN")
        self.assertTrue(((rep.audit_log["action"] == "flag") & (rep.audit_log["ticker"] == "A")).any())
        self.assertEqual(len(rep.cleaned), len(px))   # flagged, never silently removed

    def test_detects_stale_prices(self):
        px = clean_prices()
        px.iloc[100:108, 1] = px.iloc[100, 1]
        self.assertEqual(status(validate_prices(px), "Stale"), "WARN")

    def test_fails_on_non_positive_price(self):
        px = clean_prices()
        px.iloc[50, 2] = -1.0
        self.assertEqual(status(validate_prices(px), "positive"), "FAIL")

    def test_fails_on_duplicate_dates_and_repairs_them(self):
        px = clean_prices()
        px = pd.concat([px, px.iloc[[10]]])
        rep = validate_prices(px)
        self.assertEqual(status(rep, "Index"), "FAIL")
        self.assertTrue(rep.cleaned.index.is_unique)

    def test_short_gaps_are_forward_filled_and_logged(self):
        px = clean_prices()
        px.iloc[60:62, 0] = np.nan
        rep = validate_prices(px)
        self.assertFalse(rep.cleaned.isna().any().any())
        self.assertTrue((rep.audit_log["action"] == "ffill").any())

    def test_ticker_with_too_much_missing_data_is_dropped(self):
        px = clean_prices()
        px.iloc[:100, 2] = np.nan                     # 20% missing
        rep = validate_prices(px)
        self.assertNotIn("C", rep.cleaned.columns)
        self.assertEqual(status(rep, "Missing"), "FAIL")
        self.assertTrue(((rep.audit_log["action"] == "drop_ticker") & (rep.audit_log["ticker"] == "C")).any())

    def test_never_back_fills(self):
        px = clean_prices()
        px.iloc[0, 0] = np.nan
        rep = validate_prices(px)
        self.assertNotIn(px.index[0], rep.cleaned.index)


if __name__ == "__main__":
    unittest.main()
