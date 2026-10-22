"""Performance metrics. All inputs are daily simple returns."""
from __future__ import annotations

import numpy as np
import pandas as pd

TD = 252


def cagr(r: pd.Series) -> float:
    return float((1 + r).prod() ** (TD / len(r)) - 1)


def ann_vol(r: pd.Series) -> float:
    return float(r.std() * np.sqrt(TD))


def sharpe(r: pd.Series, rf: float = 0.0) -> float:
    ex = r - rf / TD
    sd = ex.std()
    return float(ex.mean() / sd * np.sqrt(TD)) if sd > 0 else float("nan")


def sortino(r: pd.Series, rf: float = 0.0) -> float:
    ex = r - rf / TD
    dd = np.sqrt((np.minimum(ex, 0) ** 2).mean())
    return float(ex.mean() / dd * np.sqrt(TD)) if dd > 0 else float("nan")


def drawdown(r: pd.Series) -> pd.Series:
    nav = (1 + r).cumprod()
    return nav / nav.cummax() - 1


def max_drawdown(r: pd.Series) -> float:
    return float(drawdown(r).min())


def performance_summary(r: pd.Series, benchmark: pd.Series | None = None, rf: float = 0.0) -> dict:
    r = r.dropna()
    out = {
        "CAGR": cagr(r),
        "Volatility": ann_vol(r),
        "Sharpe": sharpe(r, rf),
        "Sortino": sortino(r, rf),
        "Max drawdown": max_drawdown(r),
        "Calmar": cagr(r) / abs(max_drawdown(r)) if max_drawdown(r) < 0 else float("nan"),
        "VaR 95% (1d)": float(-np.quantile(r, 0.05)),
        "CVaR 95% (1d)": float(-r[r <= np.quantile(r, 0.05)].mean()),
        "Skew": float(r.skew()),
        "Excess kurtosis": float(r.kurt()),
    }
    monthly = (1 + r).groupby([r.index.year, r.index.month]).prod() - 1
    out["Best month"] = float(monthly.max())
    out["Worst month"] = float(monthly.min())
    out["% positive months"] = float((monthly > 0).mean())
    if benchmark is not None:
        b = benchmark.reindex(r.index).dropna()
        a = r.reindex(b.index)
        ex_a, ex_b = a - rf / TD, b - rf / TD
        beta = float(np.cov(ex_a, ex_b)[0, 1] / ex_b.var())
        out["Beta vs benchmark"] = beta
        out["Alpha (ann.)"] = float((ex_a.mean() - beta * ex_b.mean()) * TD)
        te = (a - b).std() * np.sqrt(TD)
        out["Tracking error"] = float(te)
        out["Information ratio"] = float((a - b).mean() * TD / te) if te > 0 else float("nan")
        up, dn = b > 0, b < 0
        out["Upside capture"] = float(a[up].mean() / b[up].mean())
        out["Downside capture"] = float(a[dn].mean() / b[dn].mean())
    return out


def sharpe_bootstrap_ci(r: pd.Series, rf: float = 0.0, n_boot: int = 2000, block: int = 21,
                        level: float = 0.95, seed: int = 0) -> tuple[float, float]:
    """Circular block-bootstrap confidence interval for the Sharpe ratio.
    Block resampling preserves volatility clustering and autocorrelation that an iid bootstrap destroys."""
    x = r.dropna().values
    n = len(x)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    out = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n] % n
        s = x[idx] - rf / TD
        out[b] = s.mean() / s.std() * np.sqrt(TD)
    lo, hi = np.quantile(out, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)
