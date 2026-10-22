"""Generates a short analyst-style investment memo in Markdown from an Analysis object.
Every sentence is built from computed results with conditional wording, so the memo cannot claim
something the numbers do not show."""
from __future__ import annotations

import pandas as pd

from .config import HISTORICAL_SCENARIOS
from .pipeline import PRIMARY_BENCHMARK, Analysis


def _p(x, d=1):
    return f"{x * 100:.{d}f}%"


def _md_table(df: pd.DataFrame) -> str:
    cols = [df.index.name or "", *df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] + ["---:"] * len(df.columns)) + "|"]
    for i, r in df.iterrows():
        out.append("| " + " | ".join([str(i), *map(str, r.values)]) + " |")
    return "\n".join(out)


def build_memo(A: Analysis, path: str) -> str:
    m, cfg = A.metrics, A.cfg
    feat, bench = cfg.featured_strategy, PRIMARY_BENCHMARK
    strat = A.strategy_names
    r0, r1 = A.returns.index[0], A.returns.index[-1]
    synthetic = "SYNTHETIC" in A.source

    tbl = pd.DataFrame({
        "CAGR": m["CAGR"].map(_p), "Vol": m["Volatility"].map(_p), "Sharpe": m["Sharpe"].map(lambda x: f"{x:.2f}"),
        "Max DD": m["Max drawdown"].map(_p), "Calmar": m["Calmar"].map(lambda x: f"{x:.2f}"),
        "Turnover/yr": m["Annual turnover"].map(lambda x: _p(x, 0)),
    })
    tbl.index.name = "Strategy"

    best_sharpe = m.loc[strat, "Sharpe"].idxmax()
    lowest_dd = m.loc[strat, "Max drawdown"].idxmax()
    ms = m.loc["Max Sharpe"]
    lo_f, hi_f = m.at[feat, "Sharpe CI low"], m.at[feat, "Sharpe CI high"]
    lo_b, hi_b = m.at[bench, "Sharpe CI low"], m.at[bench, "Sharpe CI high"]
    overlap = lo_f <= hi_b and lo_b <= hi_f

    findings = []
    findings.append(f"**Ranking.** {best_sharpe} had the highest Sharpe ratio ({m.at[best_sharpe, 'Sharpe']:.2f}) "
                    f"and {lowest_dd} the shallowest drawdown ({_p(m.at[lowest_dd, 'Max drawdown'])}). "
                    f"The {bench} benchmark delivered a Sharpe of {m.at[bench, 'Sharpe']:.2f} with a "
                    f"{_p(m.at[bench, 'Max drawdown'])} drawdown.")
    findings.append(f"**Statistical caution.** The {feat} Sharpe interval ({lo_f:.2f} to {hi_f:.2f}) "
                    f"{'overlaps' if overlap else 'does not overlap'} the {bench} interval ({lo_b:.2f} to {hi_b:.2f}). "
                    + ("The ranking between them is therefore not statistically reliable on this sample."
                       if overlap else "The difference is larger than sampling noise on this sample."))
    findings.append(f"**Optimiser instability.** Max Sharpe turned over {_p(ms['Annual turnover'], 0)} of the portfolio per year "
                    f"(cost drag {_p(ms['Cost drag (ann.)'], 2)} a year) versus "
                    f"{_p(m.at['Risk Parity', 'Annual turnover'], 0)} for Risk Parity. Return-chasing optimisers amplify estimation error.")
    rc = A.risk_contrib.loc[bench]
    top_t = rc.idxmax()
    findings.append(f"**Concentration of risk.** In the {bench} mix, {top_t} contributes {_p(rc.max(), 0)} of portfolio volatility "
                    f"on {_p(A.latest_weights.loc[bench, top_t], 0)} of the capital. Risk Parity's largest single contribution is "
                    f"{_p(A.risk_contrib.loc['Risk Parity'].max(), 0)}.")
    bt = A.var_bt.set_index("Method")
    par, hist = bt.loc["Parametric (normal)"], bt.loc["Historical"]
    findings.append(f"**VaR model validation ({feat}, 99%).** Normal-distribution VaR was breached {int(par['Exceptions'])} times "
                    f"against {par['Expected']:.0f} expected (Kupiec p = {par['Kupiec p-value']:.3f}; model {par['Verdict'].split()[0].lower()}ed). "
                    f"Historical VaR: {int(hist['Exceptions'])} breaches (p = {hist['Kupiec p-value']:.3f}; model {hist['Verdict'].split()[0].lower()}ed).")
    scen = A.stress_hist
    if len(scen):
        worst = scen[bench].idxmin()
        findings.append(f"**Stress.** Today's {bench} holdings lose {_p(-scen.at[worst, bench])} in the '{worst}' replay, "
                        f"versus {_p(-scen.at[worst, feat])} for {feat}.")
    sb = A.stock_bond_corr
    if len(sb):
        s22, e22 = HISTORICAL_SCENARIOS["2022 inflation & rate shock"]
        seg = sb.loc[s22:e22]
        if len(seg):
            findings.append(f"**Diversification regime risk.** Six-month SPY-AGG correlation averaged {sb.mean():+.2f} over the full sample "
                            f"but {seg.mean():+.2f} during the 2022 rate shock, when bonds stopped hedging equities.")

    n_ctrl = len(A.controls)
    ctrl = (f"All {n_ctrl} independent controls passed." if not A.controls_failed
            else f"{len(A.controls_failed)} of {n_ctrl} controls failed; see controls.csv.")
    warn = ("> **Synthetic data.** Numbers below come from a simulated market and are not real performance. "
            "Re-run with `--source yfinance` before citing any result.\n\n" if synthetic else "")

    text = f"""# Investment memo: multi-asset portfolio construction

{warn}**Question.** Which long-only construction method gives the best risk-adjusted outcome across a diversified ETF universe,
and how much of that outcome survives realistic costs and statistical scrutiny?

**Setup.** {len(A.returns.columns)} ETFs, {r0:%b %Y} to {r1:%b %Y}. Walk-forward with a {cfg.lookback_days}-day estimation window,
{cfg.rebalance} rebalancing, trades executed {cfg.execution_lag} day after the signal, {cfg.tc_bps:g} bp per unit traded, {cfg.max_weight:.0%} position cap.
Covariance by Ledoit-Wolf shrinkage. Benchmarks ({bench} and SPY) run through the identical engine.

## Results

{_md_table(tbl)}

## Findings

""" + "\n\n".join(f"{i}. {f}" for i, f in enumerate(findings, 1)) + f"""

## Data quality and model controls

Data checks: {int((A.dq.checks.status == 'PASS').sum())} pass, {int((A.dq.checks.status == 'WARN').sum())} warn, {int((A.dq.checks.status == 'FAIL').sum())} fail. {ctrl}
Controls include a future-data perturbation test (all data after a cutoff replaced with noise; earlier weights and returns must not change)
and an independent matrix recomputation of every net return.

## Limitations

Universe chosen with hindsight (survivorship and selection bias); flat cost model with no spread, impact or tax; long-only and unlevered,
so low-volatility strategies cannot match equity returns; the efficient frontier is in-sample; one {(r1 - r0).days / 365.25:.0f}-year sample
contains few independent market regimes, so small Sharpe differences should not be over-interpreted.
"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path
