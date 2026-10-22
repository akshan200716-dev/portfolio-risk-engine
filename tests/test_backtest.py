import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from portfolio_engine.backtest import rebalance_positions, run_all, run_backtest
from portfolio_engine.config import BacktestConfig
from portfolio_engine.controls import check_no_lookahead, run_controls
from portfolio_engine.optimizers import STRATEGIES, fixed_weights, risk_parity
from tests.helpers import small_returns

CFG = replace(BacktestConfig(), lookback_days=120)


class TestEngineMechanics(unittest.TestCase):
    def setUp(self):
        idx = pd.bdate_range("2020-01-01", periods=400)
        self.r = pd.DataFrame({"A": 0.0, "B": 0.0004}, index=idx)
        self.only_b = fixed_weights({"B": 1.0})

    def test_frictionless_return_equals_asset_return(self):
        res = run_backtest(self.r, self.only_b, "B", lookback=60, tc_bps=0)
        np.testing.assert_allclose(res.returns.values[1:], 0.0004, atol=1e-15)

    def test_initial_purchase_cost_only_when_no_turnover(self):
        res = run_backtest(self.r, self.only_b, "B", lookback=60, tc_bps=10)
        self.assertAlmostEqual(res.returns.iloc[0], -0.001)
        np.testing.assert_allclose(res.returns.values[1:], 0.0004, atol=1e-15)
        self.assertAlmostEqual(res.rebalances["cost"].iloc[1:].sum(), 0.0, places=12)

    def test_costs_reduce_returns_and_match_turnover(self):
        r = small_returns()
        net = run_backtest(r, risk_parity, "RP", lookback=120, tc_bps=25)
        free = run_backtest(r, risk_parity, "RP", lookback=120, tc_bps=0)
        self.assertLess(net.nav.iloc[-1], free.nav.iloc[-1])
        np.testing.assert_allclose(net.rebalances["cost"], net.rebalances["turnover"] * 25 / 1e4, atol=1e-15)

    def test_trade_happens_after_signal_and_lag_is_respected(self):
        for lag in (1, 2, 5):
            res = run_backtest(small_returns(), risk_parity, "RP", lookback=120, execution_lag=lag)
            gaps = [(r.Index, r.signal_date) for r in res.rebalances.itertuples()]
            idx = small_returns().index
            for trade, sig in gaps:
                self.assertEqual(idx.get_loc(trade) - idx.get_loc(sig), lag)

    def test_no_trading_before_lookback_is_available(self):
        r = small_returns()
        res = run_backtest(r, risk_parity, "RP", lookback=200)
        self.assertGreaterEqual(r.index.get_loc(res.rebalances["signal_date"].iloc[0]), 199)

    def test_rebalance_positions_are_last_day_of_each_month(self):
        idx = pd.bdate_range("2021-01-01", "2021-12-31")
        pos = rebalance_positions(idx, "monthly")
        self.assertEqual(len(pos), 12)
        for p in pos[:-1]:
            self.assertNotEqual(idx[p].month, idx[p + 1].month)
        self.assertEqual(len(rebalance_positions(idx, "quarterly")), 4)

    def test_invalid_weights_are_rejected(self):
        bad = lambda w, **_: pd.Series([0.7, 0.7], index=w.columns)
        with self.assertRaises(ValueError):
            run_backtest(self.r, bad, "bad", lookback=60)

    def test_holdings_drift_between_rebalances(self):
        r = pd.DataFrame({"A": 0.01, "B": 0.0}, index=pd.bdate_range("2020-01-01", periods=120))
        res = run_backtest(r, fixed_weights({"A": 0.5, "B": 0.5}), "5050", lookback=30, tc_bps=0)
        mid = res.holdings.iloc[5]
        self.assertGreater(mid["A"], 0.5)


class TestControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = small_returns()
        cls.params = {"max_weight": CFG.max_weight, "rf": 0.02}
        cls.res = run_all(cls.r, {"Risk Parity": STRATEGIES["Risk Parity"]}, CFG)["Risk Parity"]

    def _run(self, res):
        return run_controls(res, self.r, STRATEGIES["Risk Parity"], CFG, self.params, lookahead=False)

    def test_clean_run_passes_every_control(self):
        ctrls = run_controls(self.res, self.r, STRATEGIES["Risk Parity"], CFG, self.params)
        self.assertTrue(all(c.status == "PASS" for c in ctrls), [c for c in ctrls if c.status == "FAIL"])

    def test_lookahead_control_passes_for_every_strategy(self):
        for name, fn in STRATEGIES.items():
            c = check_no_lookahead(self.r, fn, name, CFG, self.params)
            self.assertEqual(c.status, "PASS", f"{name}: {c.detail}")

    def test_reconciliation_catches_tampered_returns(self):
        bad = replace(self.res, returns=self.res.returns.copy())
        bad.returns.iloc[50] += 1e-6
        failed = {c.control for c in self._run(bad) if c.status == "FAIL"}
        self.assertIn("Return reconciliation (holdings x returns)", failed)

    def test_weight_sum_control_catches_bad_weights(self):
        bad = replace(self.res, rebalances=self.res.rebalances.copy())
        bad.rebalances.iloc[3, bad.rebalances.columns.get_loc("SPY")] += 0.05
        failed = {c.control for c in self._run(bad) if c.status == "FAIL"}
        self.assertIn("Target weights sum to 100%", failed)

    def test_cost_control_catches_wrong_costs(self):
        bad = replace(self.res, rebalances=self.res.rebalances.copy())
        bad.rebalances["cost"] = bad.rebalances["cost"] * 2
        failed = {c.control for c in self._run(bad) if c.status == "FAIL"}
        self.assertIn("Cost = turnover x cost rate", failed)

    def test_concentration_control_catches_cap_breach(self):
        tight = replace(CFG, max_weight=0.05)
        failed = {c.control for c in run_controls(self.res, self.r, STRATEGIES["Risk Parity"], tight, self.params,
                                                  lookahead=False) if c.status == "FAIL"}
        self.assertTrue(any("Concentration cap" in f for f in failed))


if __name__ == "__main__":
    unittest.main()
