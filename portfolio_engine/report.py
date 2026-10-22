"""Self-contained HTML research dashboard (charts embedded as images, no external assets or internet)."""
from __future__ import annotations

import base64
import html
import io
from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

from .config import HISTORICAL_SCENARIOS, UNIVERSE  # noqa: E402
from .pipeline import PRIMARY_BENCHMARK, Analysis  # noqa: E402

INK, MUTED, RULE = "#17212B", "#5B6773", "#DCE1E6"
COLORS = {"Equal Weight": "#7A8CA3", "Inverse Volatility": "#3A8FB7", "Min Variance": "#2F9E8A",
          "Max Sharpe": "#D6A02B", "Risk Parity": "#1B3A5C", "HRP": "#8C5BA6",
          "60/40": "#4A4A4A", "SPY Buy & Hold": "#B24A5A"}
ASSET_COLORS = {"SPY": "#1B3A5C", "QQQ": "#2E5E8C", "IWM": "#5683B0", "EFA": "#8AA9CF", "EEM": "#BCCCE3",
                "VNQ": "#7A6BA8", "AGG": "#1F7A5C", "TLT": "#3FA37F", "LQD": "#8CCBA9", "HYG": "#D98A3D",
                "GLD": "#D6B02B", "DBC": "#8B6B3A"}
SHORT = {"Global financial crisis": "GFC", "US downgrade / euro debt": "2011", "Q4 2018 sell-off": "Q4 '18",
         "COVID crash": "COVID", "2022 inflation & rate shock": "2022"}


# ----------------------------------------------------------------------------- formatting
def _clean(txt: str) -> str:
    """Avoid '-0.0%' / '-0.00' and use a true minus sign."""
    core = txt.rstrip("%")
    if txt.startswith("-") and float(core) == 0:
        txt = txt[1:]
    return txt.replace("-", "−")


def pct(x, d=1):
    return "–" if pd.isna(x) else _clean(f"{x * 100:.{d}f}%")


def num(x, d=2):
    return "–" if pd.isna(x) else _clean(f"{x:.{d}f}")


def pill(status):
    return f'<span class="pill {status.lower()}">{html.escape(status)}</span>'


def table(df: pd.DataFrame, fmt: dict | None = None, heat: bool = False, index_label: str = "",
          raw_cols: tuple = (), left_cols: tuple = ()) -> str:
    fmt = fmt or {}
    head = "".join(f'<th class="l">{html.escape(str(c))}</th>' if c in left_cols else f"<th>{html.escape(str(c))}</th>"
                   for c in [index_label, *df.columns])
    rows = []
    for idx, row in df.iterrows():
        cells = [f'<th scope="row">{html.escape(str(idx))}</th>']
        for c in df.columns:
            v = row[c]
            f = fmt.get(c, lambda x: html.escape(str(x)))
            txt = v if c in raw_cols else f(v)
            style = ""
            if heat and isinstance(v, (int, float, np.floating)) and not pd.isna(v):
                a = min(1.0, abs(v) / 0.30)
                style = (f' style="background:rgba(163,48,63,{0.08 + 0.5 * a:.2f})"' if v < 0
                         else f' style="background:rgba(31,122,92,{0.06 + 0.35 * a:.2f})"')
            cls = ' class="l"' if c in left_cols else ""
            cells.append(f"<td{cls}{style}>{txt}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="tw"><table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def img(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=125, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return f'<img alt="chart" src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


def _style_axes(ax):
    ax.grid(True, color="#E9EDF1", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(RULE)
    ax.tick_params(colors=MUTED, labelsize=8.5)


def _line_style(A: Analysis, nm: str) -> dict:
    feat = nm == A.cfg.featured_strategy
    bench = nm in A.benchmark_names
    return dict(color=COLORS.get(nm, "#888"), lw=2.6 if feat else 1.3, ls="--" if bench else "-",
                alpha=1 if feat else 0.9, label=nm, zorder=5 if feat else 3)


def _bands(A: Analysis, ax, labels=True):
    lo, hi = A.returns.index[0], A.returns.index[-1]
    for nm, (s, e) in HISTORICAL_SCENARIOS.items():
        s, e = pd.Timestamp(s), pd.Timestamp(e)
        if e < lo or s > hi:
            continue
        ax.axvspan(s, e, color="#B24A5A", alpha=0.10, lw=0)
        if labels:
            ax.text(s + (e - s) / 2, 1.0, SHORT[nm], transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=7.5, color=MUTED)


# ----------------------------------------------------------------------------- charts
def fig_growth(A: Analysis):
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for nm, res in A.results.items():
        ax.plot(res.nav.index, res.nav.values, **_line_style(A, nm))
    _bands(A, ax)
    ax.set_yscale("log")
    lo, hi = ax.get_ylim()
    ticks = [t for t in (0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 7, 10) if lo <= t <= hi]
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel("Growth of 1 (log scale, net of costs)", color=MUTED, fontsize=9)
    ax.legend(ncol=4, frameon=False, fontsize=8.5, loc="upper left")
    _style_axes(ax)
    return img(fig)


def fig_drawdowns(A: Analysis):
    fig, ax = plt.subplots(figsize=(11, 3.8))
    from .risk import drawdown_series
    for nm, res in A.results.items():
        ax.plot(res.returns.index, drawdown_series(res.returns).values * 100, **_line_style(A, nm))
    _bands(A, ax, labels=False)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_ylabel("Drawdown from peak", color=MUTED, fontsize=9)
    _style_axes(ax)
    return img(fig)


def fig_frontier(A: Analysis):
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    f = A.frontier
    ax.plot(f["vol"] * 100, f["return"] * 100, color=INK, lw=2, label="Long-only efficient frontier (in-sample)", zorder=2)
    ax.scatter(A.asset_stats["vol"] * 100, A.asset_stats["return"] * 100, s=22, color="#B7C0CA", zorder=3)
    for t, r in A.asset_stats.iterrows():
        ax.annotate(t, (r["vol"] * 100, r["return"] * 100), xytext=(4, 3), textcoords="offset points",
                    fontsize=7.5, color=MUTED)
    for nm, r in A.strat_points.iterrows():
        ax.scatter(r["vol"] * 100, r["return"] * 100, s=70, color=COLORS[nm], edgecolor="white", zorder=5,
                   marker="D" if nm in A.benchmark_names else "o", label=nm)
    ax.set_xlabel("Annualised volatility (%)", color=MUTED, fontsize=9)
    ax.set_ylabel("Annualised mean return (%)", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    _style_axes(ax)
    return img(fig)


def _asset_legend(fig, cols, y=-0.02):
    handles = [plt.Rectangle((0, 0), 1, 1, color=ASSET_COLORS[c]) for c in cols]
    fig.legend(handles, cols, ncol=len(cols), loc="lower center", frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, y), handlelength=1, columnspacing=1)


def fig_weights(A: Analysis):
    names = A.strategy_names
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.4), sharex=True, sharey=True)
    for ax, nm in zip(axes.ravel(), names):
        h = A.results[nm].holdings
        ax.stackplot(h.index, h.T.values * 100, colors=[ASSET_COLORS[c] for c in h.columns], linewidth=0)
        ax.set_title(nm, fontsize=9.5, loc="left", color=INK)
        ax.set_ylim(0, 100)
        _style_axes(ax)
        ax.grid(False)
    for ax in axes[:, 0]:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    fig.tight_layout()
    _asset_legend(fig, list(A.returns.columns), y=-0.045)
    return img(fig)


def fig_risk_alloc(A: Analysis):
    names = list(A.results)[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (data, title) in zip(axes, ((A.latest_weights, "Share of capital"), (A.risk_contrib, "Share of risk"))):
        left = np.zeros(len(names))
        for c in A.returns.columns:
            v = data.loc[names, c].values * 100
            ax.barh(names, v, left=left, color=ASSET_COLORS[c], height=0.66, linewidth=0)
            left += v
        ax.set_xlim(0, 100)
        ax.set_title(title + " (end of sample)", fontsize=9.5, loc="left", color=INK)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
        _style_axes(ax)
        ax.grid(False)
    fig.tight_layout()
    _asset_legend(fig, list(A.returns.columns), y=-0.06)
    return img(fig)


def fig_rolling(A: Analysis):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True, gridspec_kw={"height_ratios": [1.2, 1]})
    for nm, res in A.results.items():
        a1.plot(res.returns.index, res.returns.rolling(252).std() * np.sqrt(252) * 100, **_line_style(A, nm))
    _bands(A, a1)
    a1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    a1.set_ylabel("Rolling 1-year volatility", color=MUTED, fontsize=9)
    a1.legend(ncol=4, frameon=False, fontsize=8, loc="upper left")
    c = A.stock_bond_corr
    if len(c):
        a2.fill_between(c.index, c.values, 0, where=c.values >= 0, color="#B24A5A", alpha=0.55, lw=0)
        a2.fill_between(c.index, c.values, 0, where=c.values < 0, color="#2F9E8A", alpha=0.55, lw=0)
        a2.axhline(0, color=INK, lw=0.8)
    _bands(A, a2, labels=False)
    a2.set_ylabel("SPY–AGG correlation (6-month)", color=MUTED, fontsize=9)
    for a in (a1, a2):
        _style_axes(a)
    return img(fig)


def fig_var_backtest(A: Analysis):
    f = A.var_forecasts
    r = A.results[A.cfg.featured_strategy].returns.reindex(f.index)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.bar(r.index, r.values * 100, width=1.5, color="#C9D1D9", linewidth=0)
    cols = {"Historical": "#1B3A5C", "Parametric (normal)": "#D6A02B", "Cornish-Fisher": "#2F9E8A"}
    for m, c in cols.items():
        ax.plot(f.index, -f[m] * 100, color=c, lw=1.3, label=f"{m} 99% VaR")
    exc = r < -f["Parametric (normal)"]
    ax.scatter(r.index[exc], r[exc] * 100, s=14, color="#B24A5A", zorder=5, label="Exceptions (normal VaR)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}%"))
    ax.set_ylabel(f"{A.cfg.featured_strategy}: daily return", color=MUTED, fontsize=9)
    ax.legend(ncol=4, frameon=False, fontsize=8, loc="lower left")
    _style_axes(ax)
    return img(fig)


# ----------------------------------------------------------------------------- text helpers
def headline(A: Analysis) -> str:
    f, b = A.cfg.featured_strategy, PRIMARY_BENCHMARK
    mf, mb = A.metrics.loc[f], A.metrics.loc[b]
    r = A.results[f].returns
    return (f"From {r.index[0]:%B %Y} to {r.index[-1]:%B %Y}, the {f} portfolio returned {pct(mf['CAGR'])} a year "
            f"at {pct(mf['Volatility'])} volatility (Sharpe {num(mf['Sharpe'])}), against {pct(mb['CAGR'])} at "
            f"{pct(mb['Volatility'])} for a {b} mix (Sharpe {num(mb['Sharpe'])}). Its worst drawdown was "
            f"{pct(mf['Max drawdown'])} versus {pct(mb['Max drawdown'])}.")


METHODS = pd.DataFrame([
    ("Equal Weight", "Hold 1/N of every asset; no estimation needed.", "Ignores risk: equities dominate portfolio volatility."),
    ("Inverse Volatility", "Weight each asset by 1/volatility.", "Ignores correlations between assets."),
    ("Min Variance", "Minimise portfolio variance (Ledoit-Wolf covariance, 40% cap).", "Piles into a few low-volatility assets, mostly bonds."),
    ("Max Sharpe", "Maximise return per unit of risk using shrunk mean estimates.", "Amplifies estimation error; unstable weights and high turnover."),
    ("Risk Parity", "Every asset contributes equally to portfolio volatility (Spinu convex formulation).", "Bond-heavy; reaching equity-like returns needs leverage."),
    ("HRP", "Cluster assets by correlation, then split risk down the tree (Lopez de Prado).", "Heuristic with no return view; sensitive to the clustering linkage."),
    ("60/40, SPY", "Static benchmarks run through the same engine, costs and dates.", "Benchmarks, not strategies: 60/40 is ~90% equity risk."),
], columns=["Method", "Idea", "Known weakness"]).set_index("Method")

CSS = """
:root{--ink:#17212B;--muted:#5B6773;--rule:#DCE1E6;--navy:#1B3A5C;--bg:#fff}
*{box-sizing:border-box}html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
nav{position:fixed;inset:0 auto 0 0;width:210px;padding:28px 22px;border-right:1px solid var(--rule);background:#F7F9FB}
nav b{display:block;font:600 15px Georgia,serif;margin-bottom:18px;line-height:1.3}
nav a{display:block;color:var(--muted);text-decoration:none;padding:6px 0;font-size:14px}
nav a:hover,nav a:focus-visible{color:var(--navy);outline:none;text-decoration:underline}
main{margin-left:210px;padding:40px 48px 80px;max-width:1130px}
h1{font:600 30px/1.25 Georgia,"Iowan Old Style",serif;margin:0 0 6px;letter-spacing:-.01em}
h2{font:600 22px Georgia,serif;margin:56px 0 4px;padding-top:24px;border-top:1px solid var(--rule)}
h3{font:600 15px system-ui,sans-serif;margin:30px 0 8px}
p{max-width:75ch}.lede{font:400 19px/1.5 Georgia,serif;max-width:68ch;margin:18px 0 8px}
.sub{color:var(--muted);margin:0}.note{color:var(--muted);font-size:13.5px;max-width:80ch}
.banner{margin:18px 0;padding:10px 14px;border-left:4px solid #D6A02B;background:#FBF1D9;font-size:14px}
.banner.real{border-color:#1F7A5C;background:#E6F3EE}
img{max-width:100%;height:auto;display:block;margin:8px 0}
.tw{overflow-x:auto;margin:10px 0 6px}
table{border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums;min-width:100%}
th,td{padding:6px 9px;text-align:right;border-bottom:1px solid var(--rule);white-space:nowrap}
thead th{font-weight:600;color:var(--muted);border-bottom:1.5px solid var(--ink);vertical-align:bottom;white-space:normal;line-height:1.3}
tbody th{text-align:left;font-weight:500}thead th:first-child{text-align:left}
td.l,th.l,.methods td{text-align:left;white-space:normal}
.pill{display:inline-block;padding:1px 9px;border-radius:10px;font-size:12px;font-weight:600}
.pill.pass{background:#E6F3EE;color:#1F7A5C}.pill.warn{background:#FBF1D9;color:#8A5E00}.pill.fail{background:#F8E3E6;color:#A3303F}
ul{padding-left:20px;max-width:80ch}li{margin:5px 0}
@media(max-width:860px){nav{position:static;width:auto;border:0;border-bottom:1px solid var(--rule)}nav a{display:inline-block;margin-right:14px}main{margin:0;padding:24px 18px}}
"""


# ----------------------------------------------------------------------------- page
def build_report(A: Analysis, path: str) -> str:
    cfg, feat = A.cfg, A.cfg.featured_strategy
    synthetic = "SYNTHETIC" in A.source
    r0, r1 = A.returns.index[0], A.returns.index[-1]

    banner = ('<div class="banner"><b>Synthetic data.</b> This run used a simulated market (no internet or '
              'yfinance available). The machinery is real; the numbers are not market results. Re-run with '
              '<code>--source yfinance</code> for real figures.</div>' if synthetic else
              f'<div class="banner real"><b>Data source:</b> {html.escape(A.source)} (adjusted closes, dividends reinvested).</div>')

    m = A.metrics
    perf = m[["CAGR", "Volatility", "Sharpe", "Sortino", "Max drawdown", "Calmar", "Annual turnover", "Cost drag (ann.)"]].copy()
    perf = perf.rename(columns={"Annual turnover": "Turnover / yr", "Cost drag (ann.)": "Cost drag / yr"})
    perf.insert(3, "Sharpe 95% CI", [f"{num(m.at[i, 'Sharpe CI low'])} to {num(m.at[i, 'Sharpe CI high'])}" for i in m.index])
    perf_html = table(perf, {"CAGR": pct, "Volatility": pct, "Sharpe": num, "Sortino": num, "Max drawdown": pct,
                             "Calmar": num, "Turnover / yr": lambda x: pct(x, 0), "Cost drag / yr": lambda x: pct(x, 2)},
                      index_label="Strategy")

    dd = A.dd_table.copy()
    dd_html = table(dd.set_index("Peak"), {"Depth": pct, "Trough": str, "Recovery": str}, index_label="Peak date")

    vf = A.var_featured
    var_html = table(vf, {c: (lambda x: "–" if pd.isna(x) else pct(x, 2)) for c in vf.columns}, index_label="Measure (1 day)")
    va = A.var_all
    var_all_html = table(va, {c: (lambda x: pct(x, 2)) for c in va.columns}, index_label="Strategy")

    bt = A.var_bt.set_index("Method")
    bt_html = table(bt, {"Observations": lambda x: f"{x:,}", "Expected": lambda x: num(x, 1), "Exception rate": lambda x: pct(x, 2),
                         "Kupiec LR": num, "Kupiec p-value": lambda x: num(x, 3)}, index_label="99% VaR method")

    heat_fmt = lambda x: pct(x, 1)
    sh = table(A.stress_hist.rename(index=str), {c: heat_fmt for c in A.stress_hist.columns}, heat=True, index_label="Scenario")
    sy = table(A.stress_hyp, {c: heat_fmt for c in A.stress_hyp.columns}, heat=True, index_label="Shock")

    dqc = A.dq.checks.copy()
    dqc["status"] = dqc["status"].map(pill)
    dqc["value"] = dqc["value"].map(lambda v: f"{v:,.4g}")
    dq_html = table(dqc.set_index("check"), raw_cols=("status",), index_label="Check", left_cols=("detail",))

    cm = A.controls_matrix.copy().fillna("n/a")
    cm = cm.map(lambda s: pill(s) if s in ("PASS", "FAIL") else "<span class='sub'>n/a</span>")
    cm_html = table(cm, raw_cols=tuple(cm.columns), index_label="Control")
    n_ctrl = len(A.controls)
    ctrl_line = (f"All {n_ctrl} controls passed." if not A.controls_failed
                 else f"{len(A.controls_failed)} of {n_ctrl} controls FAILED. Treat results with caution.")

    n_flags = int((A.dq.audit_log["action"] == "flag").sum()) if len(A.dq.audit_log) else 0
    n_fix = len(A.dq.audit_log) - n_flags

    body = f"""
<nav><b>Portfolio construction &amp; risk analytics</b>
<a href="#performance">Performance</a><a href="#construction">Portfolio construction</a>
<a href="#risk">Risk</a><a href="#stress">Stress tests</a><a href="#controls">Data and controls</a>
<a href="#method">Method and limits</a></nav>
<main>
<h1>Multi-asset portfolio construction and risk report</h1>
<p class="sub">{len(A.returns.columns)} ETFs across equities, bonds, credit, gold and commodities &nbsp;|&nbsp;
{r0:%d %b %Y} to {r1:%d %b %Y} &nbsp;|&nbsp; {cfg.rebalance} rebalance, {cfg.lookback_days}-day lookback,
{cfg.tc_bps:g} bp per trade, {cfg.max_weight:.0%} position cap &nbsp;|&nbsp; generated {date.today():%d %b %Y}</p>
{banner}
<p class="lede">{html.escape(headline(A))}</p>

<h2 id="performance">Performance</h2>
<p class="note">Walk-forward and net of transaction costs. Every strategy only sees data up to the signal date and trades
{cfg.execution_lag} day later. Shaded bands mark stress windows. Sharpe intervals come from a block bootstrap; when
they overlap, the ranking between two strategies is not statistically reliable.</p>
{perf_html}
{fig_growth(A)}
{fig_drawdowns(A)}
<h3>Largest drawdowns: {html.escape(feat)}</h3>
{dd_html}

<h2 id="construction">Portfolio construction</h2>
{table(METHODS, raw_cols=tuple(METHODS.columns), index_label="Method").replace("<table>", '<table class="methods">')}
<h3>Where each strategy sits against the efficient frontier</h3>
<p class="note">The frontier is built from full-sample means and covariance, so it is an in-sample ceiling no real strategy
can reach. Strategy points use realised net returns.</p>
{fig_frontier(A)}
<h3>Allocation through time</h3>
{fig_weights(A)}
<h3>Capital allocation versus risk allocation</h3>
<p class="note">Equal capital is not equal risk. In a 60/40 mix the equity sleeve typically produces most of the volatility,
which is the central argument for risk-based construction.</p>
{fig_risk_alloc(A)}

<h2 id="risk">Risk</h2>
<h3>Value at Risk and Expected Shortfall: {html.escape(feat)}</h3>
<p class="note">Positive numbers are one-day losses as a share of portfolio value. The gap between normal and historical
or t-based figures at 99% is the cost of assuming returns are Gaussian.</p>
{var_html}
<h3>Tail risk across strategies (historical, realised returns)</h3>
{var_all_html}
<h3>Is the VaR model any good? Out-of-sample backtest at 99%</h3>
<p class="note">Each day's VaR uses only the previous {250} days. A well-calibrated 99% model is breached about 1% of the time;
the Kupiec test rejects models whose exception rate differs from that by more than chance. Basel zones apply the
regulatory traffic-light rule to the most recent 250 days.</p>
{bt_html}
{fig_var_backtest(A)}
<h3>Volatility regimes and the stock-bond correlation</h3>
{fig_rolling(A)}

<h2 id="stress">Stress tests</h2>
<p class="note">Today's (end-of-sample) holdings replayed through past crises and through hypothetical shocks. Historical rows
are buy-and-hold P&amp;L of fixed weights over the window. Hypothetical shocks are illustrative assumptions, not forecasts.</p>
<h3>Historical scenario replay</h3>{sh}
<h3>Hypothetical shocks</h3>{sy}

<h2 id="controls">Data and controls</h2>
<h3>Data quality checks</h3>
{dq_html}
<p class="note">Cleaning policy: forward-fill gaps of at most 3 days, never back-fill, drop assets with more than 5% missing,
flag but never silently delete extreme moves. {n_flags} flagged observations and {n_fix} corrective actions are logged in
<code>data_audit_log.csv</code>.</p>
<h3>Independent backtest controls</h3>
<p class="note">Each control re-derives a result by a different route than the engine used. The look-ahead test
replaces all data after a cutoff date with random noise and confirms that nothing before the cutoff changes.</p>
{cm_html}
<p>{ctrl_line}</p>

<h2 id="method">Method and limits</h2>
<ul>
<li><b>No look-ahead.</b> Signals use a trailing {cfg.lookback_days}-day window ending at the signal close; trades execute
{cfg.execution_lag} day later. Parameters are fixed defaults and were not tuned on the results shown.</li>
<li><b>Estimation error is the main enemy.</b> Covariances use Ledoit-Wolf shrinkage; max-Sharpe means are shrunk toward the
cross-sectional average; positions are capped at {cfg.max_weight:.0%}.</li>
<li><b>Costs.</b> A flat {cfg.tc_bps:g} bp per unit of traded notional. No bid-ask spread model, market impact, taxes or
financing; long-only, unlevered.</li>
<li><b>Universe.</b> A fixed list of liquid ETFs chosen with hindsight, and only assets with history over the full window.
This carries survivorship and selection bias.</li>
<li><b>Frontier and stress shocks.</b> The frontier is in-sample by construction. Hypothetical shocks are round-number
assumptions.</li>
<li><b>Statistical power.</b> {(r1 - r0).days / 365.25:.0f} years is a handful of independent market regimes. Differences in Sharpe between
neighbouring strategies are usually within sampling error (see the intervals above).</li>
</ul>
</main>"""
    doc = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
           f"<title>Portfolio construction and risk report</title><style>{CSS}</style></head><body>{body}</body></html>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(doc)
    return path
