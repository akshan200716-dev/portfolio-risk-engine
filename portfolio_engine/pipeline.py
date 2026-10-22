"""Runs the whole study once and returns a single `Analysis` object that the report, memo, console
summary and CSV exports all read from (one source of truth, no number is computed twice)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import risk
from .backtest import BacktestResult, run_all
from .config import HISTORICAL_SCENARIOS, HYPOTHETICAL_SCENARIOS, UNIVERSE, BacktestConfig
from .controls import Control, controls_matrix, run_controls
from .data import load_prices, to_returns
from .metrics import performance_summary, sharpe_bootstrap_ci
from .optimizers import STRATEGIES, efficient_frontier, estimate_cov, fixed_weights, shrunk_mean
from .validation import DataQualityReport, validate_prices

BENCHMARKS = {
    "60/40": fixed_weights({"SPY": 0.6, "AGG": 0.4}),
    "SPY Buy & Hold": fixed_weights({"SPY": 1.0}),
}
PRIMARY_BENCHMARK = "60/40"


@dataclass
class Analysis:
    cfg: BacktestConfig
    source: str
    prices: pd.DataFrame
    dq: DataQualityReport
    returns: pd.DataFrame
    results: dict[str, BacktestResult]
    metrics: pd.DataFrame
    controls: list[Control]
    risk_contrib: pd.DataFrame               # strategies x assets, latest holdings
    latest_weights: pd.DataFrame             # strategies x assets
    var_featured: pd.DataFrame
    var_all: pd.DataFrame                    # strategies x {VaR95, VaR99, ES99} from realised returns
    var_bt: pd.DataFrame
    var_forecasts: pd.DataFrame
    dd_table: pd.DataFrame
    stress_hist: pd.DataFrame
    stress_hyp: pd.DataFrame
    frontier: pd.DataFrame
    asset_stats: pd.DataFrame
    strat_points: pd.DataFrame
    stock_bond_corr: pd.Series = field(default_factory=pd.Series)

    @property
    def strategy_names(self): return list(STRATEGIES)
    @property
    def benchmark_names(self): return list(BENCHMARKS)
    @property
    def controls_matrix(self): return controls_matrix(self.controls)
    @property
    def controls_failed(self): return [c for c in self.controls if c.status == "FAIL"]


def run_pipeline(cfg: BacktestConfig | None = None, source: str = "auto", csv_path=None, seed: int = 7,
                 cache_dir: str = "data", n_boot: int = 1000, log=print) -> Analysis:
    cfg = cfg or BacktestConfig()

    log("1/7 loading prices ...")
    prices, label = load_prices(list(UNIVERSE), cfg.start, cfg.end, source, csv_path, cache_dir, seed)

    log("2/7 validating data ...")
    dq = validate_prices(prices)
    returns = to_returns(dq.cleaned)

    log("3/7 running walk-forward backtests ...")
    results = run_all(returns, STRATEGIES, cfg, BENCHMARKS)
    names = list(results)

    log("4/7 computing performance metrics ...")
    bench_r = results[PRIMARY_BENCHMARK].returns
    rows = {}
    for nm, res in results.items():
        m = performance_summary(res.returns, None if nm == PRIMARY_BENCHMARK else bench_r, cfg.risk_free_annual)
        lo, hi = sharpe_bootstrap_ci(res.returns, cfg.risk_free_annual, n_boot=n_boot)
        m.update({"Sharpe CI low": lo, "Sharpe CI high": hi, "Annual turnover": res.annual_turnover,
                  "Cost drag (ann.)": res.total_cost_drag})
        rows[nm] = m
    metrics = pd.DataFrame(rows).T

    log("5/7 running independent controls ...")
    all_fn = {**STRATEGIES, **BENCHMARKS}
    controls: list[Control] = []
    for nm, res in results.items():
        is_strat = nm in STRATEGIES
        params = {"max_weight": cfg.max_weight, "rf": cfg.risk_free_annual} if is_strat else None
        controls += run_controls(res, returns, all_fn[nm], cfg, params, capped=is_strat)

    log("6/7 risk analytics and stress tests ...")
    latest = pd.DataFrame({nm: res.holdings.iloc[-1] for nm, res in results.items()}).T
    cov_latest = estimate_cov(returns.iloc[-cfg.lookback_days:])
    contrib = pd.DataFrame({nm: risk.risk_contributions(latest.loc[nm], cov_latest) for nm in names}).T

    feat = cfg.featured_strategy
    var_featured = risk.var_table(results[feat].returns, latest.loc[feat], returns.iloc[-cfg.lookback_days * 4:])
    var_all = pd.DataFrame({nm: {"VaR 95%": risk.historical_var(res.returns, 0.95),
                                 "VaR 99%": risk.historical_var(res.returns, 0.99),
                                 "ES 99%": risk.historical_es(res.returns, 0.99)}
                            for nm, res in results.items()}).T
    var_bt, forecasts = risk.var_backtest(results[feat].returns, alpha=0.99)
    dd_table = risk.drawdown_table(results[feat].returns)

    stress_hist = pd.DataFrame({nm: risk.stress_historical(returns, latest.loc[nm], HISTORICAL_SCENARIOS)
                                for nm in names})
    stress_hyp = pd.DataFrame({nm: risk.stress_hypothetical(latest.loc[nm], HYPOTHETICAL_SCENARIOS)
                               for nm in names})

    log("7/7 efficient frontier ...")
    mu = shrunk_mean(returns, 0.0)
    cov_full = estimate_cov(returns)
    frontier = efficient_frontier(mu.values, cov_full.values, 40, 1.0)
    asset_stats = pd.DataFrame({"return": mu, "vol": np.sqrt(np.diag(cov_full.values))}, index=returns.columns)
    pts = {nm: {"return": res.returns.mean() * 252, "vol": res.returns.std() * np.sqrt(252)}
           for nm, res in results.items()}
    sb = returns["SPY"].rolling(126).corr(returns["AGG"]).dropna() if {"SPY", "AGG"} <= set(returns) else pd.Series(dtype=float)

    return Analysis(cfg, label, prices, dq, returns, results, metrics, controls, contrib, latest,
                    var_featured, var_all, var_bt, forecasts, dd_table, stress_hist, stress_hyp, frontier,
                    asset_stats, pd.DataFrame(pts).T, sb)
