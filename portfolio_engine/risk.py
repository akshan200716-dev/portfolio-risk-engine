"""Market-risk analytics: VaR / Expected Shortfall, drawdowns, risk decomposition, stress tests and
statistical validation of VaR models (Kupiec test, Basel traffic light).

Conventions: VaR and ES are reported as POSITIVE loss fractions (0.02 = 2% of portfolio value).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

TD = 252


# ----------------------------------------------------------------------------- VaR / ES
def historical_var(r, alpha=0.95) -> float:
    return float(-np.quantile(np.asarray(r), 1 - alpha))


def historical_es(r, alpha=0.95) -> float:
    r = np.asarray(r)
    cut = np.quantile(r, 1 - alpha)
    return float(-r[r <= cut].mean())


def parametric_var(r, alpha=0.95) -> float:
    r = np.asarray(r)
    return float(-(r.mean() + r.std(ddof=1) * stats.norm.ppf(1 - alpha)))


def parametric_es(r, alpha=0.95) -> float:
    r = np.asarray(r)
    z = stats.norm.ppf(1 - alpha)
    return float(-(r.mean() - r.std(ddof=1) * stats.norm.pdf(z) / (1 - alpha)))


def cornish_fisher_var(r, alpha=0.95) -> float:
    """Gaussian VaR adjusted for sample skewness and excess kurtosis (Cornish-Fisher expansion)."""
    r = np.asarray(r)
    z = stats.norm.ppf(1 - alpha)
    s, k = stats.skew(r), stats.kurtosis(r)
    zcf = z + (z ** 2 - 1) * s / 6 + (z ** 3 - 3 * z) * k / 24 - (2 * z ** 3 - 5 * z) * s ** 2 / 36
    return float(-(r.mean() + r.std(ddof=1) * zcf))


def monte_carlo_var_es(weights: pd.Series, returns: pd.DataFrame, alpha=0.95, n_sims=200_000,
                       df=5, seed=0) -> tuple[float, float]:
    """Simulate multivariate Student-t scenarios with the sample mean/covariance (fat joint tails)."""
    mu = returns.mean().values
    cov = np.cov(returns.values, rowvar=False)
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(cov + 1e-14 * np.eye(len(mu)))
    z = rng.standard_normal((n_sims, len(mu))) @ L.T
    mix = np.sqrt((df - 2) / rng.chisquare(df, n_sims))[:, None]   # unit-variance scaling
    sims = mu + z * mix
    pnl = sims @ weights.reindex(returns.columns).values
    q = np.quantile(pnl, 1 - alpha)
    return float(-q), float(-pnl[pnl <= q].mean())


def var_table(port_returns: pd.Series, weights: pd.Series, asset_returns: pd.DataFrame,
              alphas=(0.95, 0.99)) -> pd.DataFrame:
    rows = {}
    for a in alphas:
        mc_var, mc_es = monte_carlo_var_es(weights, asset_returns, a)
        rows[f"VaR {a:.0%}"] = {"Historical": historical_var(port_returns, a),
                                "Parametric (normal)": parametric_var(port_returns, a),
                                "Cornish-Fisher": cornish_fisher_var(port_returns, a),
                                "Monte Carlo (t)": mc_var}
        rows[f"ES {a:.0%}"] = {"Historical": historical_es(port_returns, a),
                               "Parametric (normal)": parametric_es(port_returns, a),
                               "Cornish-Fisher": np.nan,
                               "Monte Carlo (t)": mc_es}
    return pd.DataFrame(rows).T


# ----------------------------------------------------------------------------- drawdowns
def drawdown_series(r: pd.Series) -> pd.Series:
    nav = (1 + r).cumprod()
    return nav / nav.cummax() - 1


def drawdown_table(r: pd.Series, top: int = 5) -> pd.DataFrame:
    """Largest distinct drawdown episodes with depth, peak/trough/recovery dates and durations."""
    dd = drawdown_series(r)
    episodes, start = [], None
    for i, (d, v) in enumerate(dd.items()):
        if v < 0 and start is None:
            start = i - 1 if i > 0 else 0
        if v == 0 and start is not None:
            episodes.append((start, i))
            start = None
    if start is not None:
        episodes.append((start, None))
    rows = []
    for s, e in episodes:
        seg = dd.iloc[s:(e if e is not None else len(dd))]
        rows.append({"Peak": dd.index[s].date(), "Trough": seg.idxmin().date(), "Depth": float(seg.min()),
                     "Recovery": dd.index[e].date() if e is not None else "not recovered",
                     "Days to trough": int(len(dd.loc[dd.index[s]:seg.idxmin()]) - 1),
                     "Days underwater": int((e if e is not None else len(dd)) - s)})
    return pd.DataFrame(rows).sort_values("Depth").head(top).reset_index(drop=True)


# ----------------------------------------------------------------------------- risk decomposition
def risk_contributions(weights: pd.Series, cov: pd.DataFrame) -> pd.Series:
    """Percentage contribution of each asset to portfolio volatility (sums to 1)."""
    w = weights.reindex(cov.index).values
    marginal = cov.values @ w
    vol = np.sqrt(w @ marginal)
    return pd.Series(w * marginal / vol ** 2, index=cov.index)


# ----------------------------------------------------------------------------- stress tests
def stress_historical(asset_returns: pd.DataFrame, weights: pd.Series, scenarios: dict) -> pd.Series:
    """Buy-and-hold P&L of fixed weights over each historical window (skipped if no data in range)."""
    out = {}
    w = weights.reindex(asset_returns.columns).fillna(0.0)
    for name, (s, e) in scenarios.items():
        seg = asset_returns.loc[s:e]
        if len(seg) < 5:
            continue
        asset_pnl = (1 + seg).prod() - 1
        out[name] = float((w * asset_pnl).sum())
    return pd.Series(out)


def stress_hypothetical(weights: pd.Series, shocks: dict) -> pd.Series:
    return pd.Series({name: float(sum(weights.get(t, 0.0) * v for t, v in sh.items()))
                      for name, sh in shocks.items()})


# ----------------------------------------------------------------------------- VaR model validation
def rolling_var(r: pd.Series, method: str, alpha: float = 0.99, window: int = 250) -> pd.Series:
    """One-day-ahead VaR forecast using only data up to t-1 (shifted, so no look-ahead)."""
    if method == "Historical":
        v = -r.rolling(window).quantile(1 - alpha)
    elif method == "Parametric (normal)":
        v = -(r.rolling(window).mean() + r.rolling(window).std() * stats.norm.ppf(1 - alpha))
    elif method == "Cornish-Fisher":
        z = stats.norm.ppf(1 - alpha)
        s, k = r.rolling(window).skew(), r.rolling(window).kurt()
        zcf = z + (z ** 2 - 1) * s / 6 + (z ** 3 - 3 * z) * k / 24 - (2 * z ** 3 - 5 * z) * s ** 2 / 36
        v = -(r.rolling(window).mean() + r.rolling(window).std() * zcf)
    else:
        raise ValueError(method)
    return v.shift(1)


def kupiec_pof(n_exceptions: int, n_obs: int, alpha: float) -> tuple[float, float]:
    """Kupiec proportion-of-failures likelihood-ratio test. H0: exception rate == 1 - alpha."""
    p, x, n = 1 - alpha, n_exceptions, n_obs
    if x == 0:
        lr = -2 * n * np.log(1 - p)
    elif x == n:
        lr = -2 * n * np.log(p)
    else:
        lr = -2 * ((n - x) * np.log(1 - p) + x * np.log(p)
                   - (n - x) * np.log(1 - x / n) - x * np.log(x / n))
    return float(lr), float(1 - stats.chi2.cdf(lr, 1))


def basel_zone(n_exceptions_250d: int) -> str:
    return "Green" if n_exceptions_250d <= 4 else ("Yellow" if n_exceptions_250d <= 9 else "Red")


def var_backtest(r: pd.Series, alpha: float = 0.99, window: int = 250,
                 methods=("Historical", "Parametric (normal)", "Cornish-Fisher")) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (summary table, daily VaR forecasts)."""
    forecasts = pd.DataFrame({m: rolling_var(r, m, alpha, window) for m in methods}).dropna()
    realised = r.reindex(forecasts.index)
    rows = []
    for m in methods:
        exc = realised < -forecasts[m]
        n, x = len(exc), int(exc.sum())
        lr, pval = kupiec_pof(x, n, alpha)
        last250 = int(exc.iloc[-250:].sum())
        rows.append({"Method": m, "Observations": n, "Exceptions": x,
                     "Expected": round(n * (1 - alpha), 1), "Exception rate": x / n,
                     "Kupiec LR": lr, "Kupiec p-value": pval,
                     "Verdict": "Reject (misspecified)" if pval < 0.05 else "Accept",
                     "Basel zone (last 250d)": basel_zone(last250)})
    return pd.DataFrame(rows), forecasts
