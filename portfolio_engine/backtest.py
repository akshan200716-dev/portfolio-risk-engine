"""Walk-forward backtest engine.

Timeline for one rebalance (execution_lag = 1):
    close of day t      : strategy sees returns up to and including t, computes target weights
    close of day t + 1  : trade to target at that close (costs = tc * sum|target - drifted weights|)
    from day t + 2      : new weights earn returns
Weights drift with market moves between rebalances. Nothing after the signal date is ever visible to
the strategy; `controls.check_no_lookahead` proves this empirically by corrupting the future.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    name: str
    returns: pd.Series            # net-of-cost daily returns
    gross_returns: pd.Series      # before trading costs
    holdings: pd.DataFrame        # end-of-day weights (after any rebalance trade)
    rebalances: pd.DataFrame      # one row per trade date: signal_date, turnover, cost, + target weights
    tc_bps: float
    execution_lag: int

    @property
    def nav(self) -> pd.Series:
        return (1 + self.returns).cumprod()

    @property
    def target_weights(self) -> pd.DataFrame:
        cols = [c for c in self.rebalances.columns if c not in ("signal_date", "turnover", "cost")]
        return self.rebalances[cols]

    @property
    def annual_turnover(self) -> float:
        years = len(self.returns) / 252
        return float(self.rebalances["turnover"].iloc[1:].sum() / years) if years > 0 else float("nan")

    @property
    def total_cost_drag(self) -> float:
        """Annualised return lost to trading costs."""
        c = lambda r: (1 + r).prod() ** (252 / len(r)) - 1
        return float(c(self.gross_returns) - c(self.returns))


def rebalance_positions(index: pd.DatetimeIndex, freq: str) -> np.ndarray:
    """Integer positions of the last trading day of each month/quarter (alias-free, any pandas version)."""
    pos = pd.Series(np.arange(len(index)), index=index)
    if freq == "monthly":
        keys = [index.year, index.month]
    elif freq == "quarterly":
        keys = [index.year, index.quarter]
    else:
        raise ValueError("freq must be 'monthly' or 'quarterly'")
    return pos.groupby(keys).last().values


def _sanitize(w: pd.Series, columns, name: str) -> np.ndarray:
    v = w.reindex(columns).fillna(0.0).values.astype(float)
    if not np.isfinite(v).all() or v.min() < -1e-9 or abs(v.sum() - 1) > 1e-6:
        raise ValueError(f"{name}: invalid weights (sum={v.sum():.6f}, min={v.min():.6f})")
    v = np.clip(v, 0, None)
    return v / v.sum()


def run_backtest(returns: pd.DataFrame, strategy: Callable, name: str, lookback: int = 252,
                 rebalance: str = "monthly", tc_bps: float = 10.0, execution_lag: int = 1,
                 params: dict | None = None) -> BacktestResult:
    params = params or {}
    cols = returns.columns
    R = returns.values
    n, p = R.shape
    tc = tc_bps / 1e4

    # signal days need a full lookback window; trade `execution_lag` days later
    signals = [i for i in rebalance_positions(returns.index, rebalance) if i + 1 >= lookback]
    plan = {i + execution_lag: i for i in signals if i + execution_lag < n}
    if not plan:
        raise ValueError("not enough history for the chosen lookback")
    first = min(plan)

    def target_at(sig_pos: int) -> np.ndarray:
        window = returns.iloc[sig_pos - lookback + 1: sig_pos + 1]       # <- strictly causal slice
        return _sanitize(strategy(window, **params), cols, name)

    w = target_at(plan[first])
    net, gross = np.zeros(n - first), np.zeros(n - first)
    hold = np.zeros((n - first, p))
    trades = []

    # initial purchase from cash at the first execution close
    net[0] = -tc * w.sum()
    hold[0] = w
    trades.append({"date": returns.index[first], "signal_date": returns.index[plan[first]],
                   "turnover": float(w.sum()), "cost": tc * float(w.sum()), "w": w.copy()})

    for k, i in enumerate(range(first + 1, n), start=1):
        g = float(w @ R[i])
        gross[k] = g
        w = w * (1 + R[i]) / (1 + g)                                     # drift
        r_net = g
        if i in plan:
            tgt = target_at(plan[i])
            turnover = float(np.abs(tgt - w).sum())
            cost = tc * turnover
            r_net = (1 + g) * (1 - cost) - 1
            trades.append({"date": returns.index[i], "signal_date": returns.index[plan[i]],
                           "turnover": turnover, "cost": cost, "w": tgt.copy()})
            w = tgt
        net[k] = r_net
        hold[k] = w

    idx = returns.index[first:]
    reb = pd.DataFrame([{"signal_date": t["signal_date"], "turnover": t["turnover"], "cost": t["cost"],
                         **dict(zip(cols, t["w"]))} for t in trades],
                       index=pd.DatetimeIndex([t["date"] for t in trades], name="trade_date"))
    return BacktestResult(name=name, returns=pd.Series(net, idx, name=name),
                          gross_returns=pd.Series(gross, idx, name=name),
                          holdings=pd.DataFrame(hold, idx, cols), rebalances=reb,
                          tc_bps=tc_bps, execution_lag=execution_lag)


def run_all(returns: pd.DataFrame, strategies: dict, cfg, benchmarks: dict | None = None) -> dict:
    """Run every strategy (with config params) plus fixed-weight benchmarks on identical dates."""
    common = dict(lookback=cfg.lookback_days, rebalance=cfg.rebalance, tc_bps=cfg.tc_bps,
                  execution_lag=cfg.execution_lag)
    out = {}
    for nm, fn in strategies.items():
        out[nm] = run_backtest(returns, fn, nm, params={"max_weight": cfg.max_weight,
                                                        "rf": cfg.risk_free_annual}, **common)
    for nm, fn in (benchmarks or {}).items():
        out[nm] = run_backtest(returns, fn, nm, **common)
    return out
