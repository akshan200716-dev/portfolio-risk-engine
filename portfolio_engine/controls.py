"""Independent integrity controls for backtest results.

These re-derive results by a different route than the engine used (matrix recomputation, perturbation
experiments) rather than trusting the engine's own bookkeeping, the way a model-validation or internal
audit team would test a model they did not build.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest import BacktestResult, run_backtest


@dataclass
class Control:
    strategy: str
    control: str
    status: str      # PASS | FAIL
    detail: str


def _c(strategy, name, ok, detail) -> Control:
    return Control(strategy, name, "PASS" if ok else "FAIL", detail)


def check_no_lookahead(returns: pd.DataFrame, strategy, name: str, cfg, params: dict | None = None,
                       cutoff_frac: float = 0.6, seed: int = 123) -> Control:
    """Replace every return after a cutoff date with random noise and re-run the backtest.
    If the engine or strategy peeks at the future, the pre-cutoff weights/returns will change."""
    n = len(returns)
    cut = int(n * cutoff_frac)
    rng = np.random.default_rng(seed)
    corrupted = returns.copy()
    corrupted.iloc[cut + 1:] = rng.normal(0.0, 0.05, corrupted.iloc[cut + 1:].shape)
    kw = dict(lookback=cfg.lookback_days, rebalance=cfg.rebalance, tc_bps=cfg.tc_bps,
              execution_lag=cfg.execution_lag, params=params)
    a = run_backtest(returns, strategy, name, **kw)
    b = run_backtest(corrupted, strategy, name, **kw)
    cutoff_date = returns.index[cut]
    # holdings are known through `cutoff_date` and must be identical in both worlds
    h_a, h_b = a.holdings.loc[:cutoff_date], b.holdings.loc[:cutoff_date]
    r_a, r_b = a.returns.loc[:cutoff_date], b.returns.loc[:cutoff_date]
    diff = max(float((h_a - h_b).abs().to_numpy().max()), float((r_a - r_b).abs().max()))
    return _c(name, "No look-ahead (future-data perturbation)", diff < 1e-12,
              f"max difference before {cutoff_date.date()} after scrambling all later data = {diff:.2e}")


def run_controls(result: BacktestResult, returns: pd.DataFrame, strategy, cfg,
                 params: dict | None = None, capped: bool = True, lookahead: bool = True) -> list[Control]:
    nm = result.name
    out: list[Control] = []
    tw = result.target_weights

    dev = float((tw.sum(axis=1) - 1).abs().max())
    out.append(_c(nm, "Target weights sum to 100%", dev < 1e-9, f"max deviation {dev:.2e}"))

    mn = float(tw.min().min())
    out.append(_c(nm, "Long-only (no negative weights)", mn >= -1e-9, f"min weight {mn:.2e}"))

    if capped:
        mx = float(tw.max().max())
        out.append(_c(nm, f"Concentration cap ({cfg.max_weight:.0%})", mx <= cfg.max_weight + 1e-6,
                      f"largest weight {mx:.2%}"))

    # independent recomputation: net_t = (1 + h_{t-1} . r_t) * (1 - cost_t) - 1
    R = returns.reindex(result.holdings.index)
    prev_h = result.holdings.shift(1).iloc[1:]
    g = (prev_h.values * R.iloc[1:].values).sum(axis=1)
    cost = result.rebalances["cost"].reindex(result.holdings.index).fillna(0.0).iloc[1:].values
    recomputed = (1 + g) * (1 - cost) - 1
    err = float(np.abs(recomputed - result.returns.iloc[1:].values).max())
    out.append(_c(nm, "Return reconciliation (holdings x returns)", err < 1e-12,
                  f"max abs difference vs independent recomputation {err:.2e}"))

    expected_cost = result.rebalances["turnover"] * result.tc_bps / 1e4
    cerr = float((expected_cost - result.rebalances["cost"]).abs().max())
    out.append(_c(nm, "Cost = turnover x cost rate", cerr < 1e-12, f"max difference {cerr:.2e}"))

    lag = (result.rebalances.index.to_series().values - result.rebalances["signal_date"].values)
    ok_lag = bool((lag > np.timedelta64(0, "D")).all())
    out.append(_c(nm, "Trade executes after signal date", ok_lag,
                  f"minimum gap {pd.Timedelta(lag.min()).days} calendar day(s), lag setting {result.execution_lag}"))

    out.append(_c(nm, "No missing / infinite returns", bool(np.isfinite(result.returns.values).all()),
                  f"{int((~np.isfinite(result.returns.values)).sum())} bad values"))

    if lookahead:
        out.append(check_no_lookahead(returns, strategy, nm, cfg, params))
    return out


def controls_matrix(controls: list[Control]) -> pd.DataFrame:
    df = pd.DataFrame([c.__dict__ for c in controls])
    return df.pivot(index="control", columns="strategy", values="status")[list(dict.fromkeys(c.strategy for c in controls))]
