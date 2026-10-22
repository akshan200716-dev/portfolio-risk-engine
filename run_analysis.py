#!/usr/bin/env python3
"""Run the full portfolio construction and risk study.

    python run_analysis.py --source yfinance        # real market data (needs internet)
    python run_analysis.py --source synthetic       # offline demo data, NOT real results
    python run_analysis.py --source csv --csv my_prices.csv
"""
import argparse
import warnings
from dataclasses import replace
from pathlib import Path

import pandas as pd

from portfolio_engine.config import BacktestConfig
from portfolio_engine.memo import build_memo
from portfolio_engine.pipeline import run_pipeline
from portfolio_engine.report import build_report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="auto", choices=["auto", "yfinance", "synthetic", "csv"])
    ap.add_argument("--csv", help="price CSV (date index, one column per ticker) when --source csv")
    ap.add_argument("--start", default=BacktestConfig.start)
    ap.add_argument("--end", default=BacktestConfig.end)
    ap.add_argument("--lookback", type=int, default=BacktestConfig.lookback_days)
    ap.add_argument("--rebalance", default=BacktestConfig.rebalance, choices=["monthly", "quarterly"])
    ap.add_argument("--tc-bps", type=float, default=BacktestConfig.tc_bps)
    ap.add_argument("--max-weight", type=float, default=BacktestConfig.max_weight)
    ap.add_argument("--seed", type=int, default=7, help="seed for synthetic data")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    cfg = replace(BacktestConfig(), start=args.start, end=args.end, lookback_days=args.lookback,
                  rebalance=args.rebalance, tc_bps=args.tc_bps, max_weight=args.max_weight)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings():
        warnings.simplefilter("always")
        A = run_pipeline(cfg, args.source, args.csv, args.seed)

    # ---- exports (every number in the report is reproducible from these files)
    A.metrics.to_csv(out / "metrics.csv")
    pd.concat({n: r.returns for n, r in A.results.items()}, axis=1).to_csv(out / "daily_returns.csv")
    pd.concat({n: r.rebalances for n, r in A.results.items()}).to_csv(out / "rebalance_ledger.csv")
    pd.DataFrame([c.__dict__ for c in A.controls]).to_csv(out / "controls.csv", index=False)
    A.dq.checks.to_csv(out / "data_quality_checks.csv", index=False)
    A.dq.audit_log.to_csv(out / "data_audit_log.csv", index=False)
    A.stress_hist.to_csv(out / "stress_historical.csv")
    A.stress_hyp.to_csv(out / "stress_hypothetical.csv")
    A.risk_contrib.to_csv(out / "risk_contributions.csv")
    A.var_bt.to_csv(out / "var_backtest.csv", index=False)
    report = build_report(A, str(out / "dashboard.html"))
    memo = build_memo(A, str(out / "investment_memo.md"))

    # ---- console summary
    pd.set_option("display.width", 160)
    show = A.metrics[["CAGR", "Volatility", "Sharpe", "Max drawdown", "Calmar", "Annual turnover"]].copy()
    for c in ("CAGR", "Volatility", "Max drawdown"):
        show[c] = show[c].map(lambda x: f"{x:.1%}")
    show["Annual turnover"] = show["Annual turnover"].map(lambda x: f"{x:.0%}")
    show[["Sharpe", "Calmar"]] = show[["Sharpe", "Calmar"]].map(lambda x: f"{x:.2f}")
    print(f"\nData source: {A.source}   |   {A.returns.index[0]:%Y-%m-%d} to {A.returns.index[-1]:%Y-%m-%d}\n")
    print(show.to_string())
    print(f"\nData checks: {(A.dq.checks.status == 'PASS').sum()} pass / {(A.dq.checks.status == 'WARN').sum()} warn / "
          f"{(A.dq.checks.status == 'FAIL').sum()} fail   |   Controls: {len(A.controls) - len(A.controls_failed)}/{len(A.controls)} pass")
    print(f"\nOpen {report} in a browser. Memo: {memo}")


if __name__ == "__main__":
    main()
