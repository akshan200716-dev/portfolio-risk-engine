"""Portfolio construction methods.

Every strategy has the same signature, `f(window_returns: DataFrame, **params) -> Series` of long-only
weights summing to 1, so the backtester can treat them interchangeably.

Estimation-error controls used throughout:
  * Ledoit-Wolf shrinkage of the covariance matrix (stabilises the inverse-like behaviour of optimisers)
  * James-Stein-style shrinkage of expected returns toward the cross-sectional mean (max-Sharpe only)
  * a single-asset weight cap
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import linprog, minimize
from scipy.spatial.distance import squareform

TRADING_DAYS = 252


# ----------------------------------------------------------------------------- estimators
def ledoit_wolf_cov(returns: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf (2004) shrinkage toward a scaled identity. Returns (covariance, shrinkage intensity)."""
    X = np.asarray(returns, dtype=float)
    X = X - X.mean(axis=0)
    n, p = X.shape
    S = X.T @ X / n
    mu = np.trace(S) / p
    delta = ((S - mu * np.eye(p)) ** 2).sum() / p
    X2 = X ** 2
    beta = ((X2.T @ X2).sum() / n - (S ** 2).sum()) / (p * n)
    beta = min(beta, delta)
    shrink = 0.0 if beta <= 0 or delta <= 0 else beta / delta
    return (1 - shrink) * S + shrink * mu * np.eye(p), float(shrink)


def estimate_cov(returns: pd.DataFrame, method: str = "ledoit_wolf") -> pd.DataFrame:
    """Annualised covariance matrix."""
    if method == "sample":
        c = np.cov(returns.values, rowvar=False)
    elif method == "ledoit_wolf":
        c, _ = ledoit_wolf_cov(returns.values)
    else:
        raise ValueError(f"unknown covariance method {method}")
    return pd.DataFrame(c * TRADING_DAYS, index=returns.columns, columns=returns.columns)


def shrunk_mean(returns: pd.DataFrame, alpha: float = 0.5) -> pd.Series:
    """Annualised mean returns shrunk toward the grand mean (sample means are very noisy)."""
    m = returns.mean() * TRADING_DAYS
    return (1 - alpha) * m + alpha * m.mean()


# ----------------------------------------------------------------------------- helpers
def apply_cap(w: np.ndarray, cap: float) -> np.ndarray:
    """Iteratively clip weights at `cap` and redistribute the excess pro-rata to uncapped assets."""
    w = np.asarray(w, dtype=float).copy()
    if cap * len(w) < 1 - 1e-9:
        raise ValueError("cap too small: cap * n_assets must be >= 1")
    for _ in range(len(w)):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        free = ~over & (w < cap - 1e-12)
        if not free.any():
            break
        w[free] += excess * w[free] / w[free].sum()
    return w / w.sum()


def _solve(objective, grad, n, cap, w0=None):
    w0 = np.full(n, 1.0 / n) if w0 is None else w0
    res = minimize(objective, w0, jac=grad, method="SLSQP", bounds=[(0.0, cap)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones_like(w)}],
                   options={"maxiter": 500, "ftol": 1e-12})
    if not res.success:
        warnings.warn(f"optimizer did not converge ({res.message}); using equal weight", stacklevel=3)
        return np.full(n, 1.0 / n)
    return np.clip(res.x, 0, None) / np.clip(res.x, 0, None).sum()


def _series(w, cols) -> pd.Series:
    return pd.Series(np.asarray(w, dtype=float), index=cols)


# ----------------------------------------------------------------------------- strategies
def equal_weight(returns: pd.DataFrame, **_) -> pd.Series:
    n = returns.shape[1]
    return _series(np.full(n, 1.0 / n), returns.columns)


def inverse_vol(returns: pd.DataFrame, max_weight: float = 1.0, **_) -> pd.Series:
    iv = 1.0 / returns.std().values
    return _series(apply_cap(iv / iv.sum(), max_weight), returns.columns)


def min_variance(returns: pd.DataFrame, max_weight: float = 0.4, cov_method: str = "ledoit_wolf", **_) -> pd.Series:
    S = estimate_cov(returns, cov_method).values
    n = len(S)
    w = _solve(lambda w: w @ S @ w, lambda w: 2 * S @ w, n, max_weight)
    return _series(w, returns.columns)


def max_sharpe(returns: pd.DataFrame, max_weight: float = 0.4, rf: float = 0.0, mean_shrink: float = 0.5,
               cov_method: str = "ledoit_wolf", **_) -> pd.Series:
    S = estimate_cov(returns, cov_method).values
    mu = shrunk_mean(returns, mean_shrink).values
    n = len(mu)

    def neg_sharpe(w):
        return -(w @ mu - rf) / np.sqrt(w @ S @ w)

    def grad(w):
        sig = np.sqrt(w @ S @ w)
        r = w @ mu - rf
        return -(mu / sig - r * (S @ w) / sig ** 3)

    w = _solve(neg_sharpe, grad, n, max_weight)
    return _series(w, returns.columns)


def risk_parity(returns: pd.DataFrame, max_weight: float = 1.0, cov_method: str = "ledoit_wolf", **_) -> pd.Series:
    """Equal risk contribution (Spinu 2013): min 0.5 w'Sw - (1/n) sum(log w); the problem is convex."""
    S = estimate_cov(returns, cov_method).values
    n = len(S)
    res = minimize(lambda w: 0.5 * w @ S @ w - np.log(w).sum() / n, np.full(n, 1.0 / n),
                   jac=lambda w: S @ w - 1.0 / (n * w), method="L-BFGS-B", bounds=[(1e-9, None)] * n)
    w = res.x / res.x.sum()
    return _series(apply_cap(w, max_weight), returns.columns)


def hrp(returns: pd.DataFrame, max_weight: float = 1.0, cov_method: str = "ledoit_wolf", **_) -> pd.Series:
    """Hierarchical Risk Parity (Lopez de Prado 2016): cluster by correlation, then split risk top-down.
    No matrix inversion, so it is robust to noisy / near-singular covariance estimates."""
    cov = estimate_cov(returns, cov_method)
    sd = np.sqrt(np.diag(cov.values))
    corr = np.clip(cov.values / np.outer(sd, sd), -1, 1)
    dist = np.sqrt(0.5 * (1 - corr))
    np.fill_diagonal(dist, 0.0)
    order = list(leaves_list(linkage(squareform(dist, checks=False), method="single")))

    def cluster_var(items):
        sub = cov.values[np.ix_(items, items)]
        iv = 1.0 / np.diag(sub)
        w = iv / iv.sum()
        return float(w @ sub @ w)

    w = pd.Series(1.0, index=order)
    clusters = [order]
    while clusters:
        clusters = [c[j:k] for c in clusters for j, k in ((0, len(c) // 2), (len(c) // 2, len(c))) if len(c) > 1]
        for i in range(0, len(clusters), 2):
            a, b = clusters[i], clusters[i + 1]
            va, vb = cluster_var(a), cluster_var(b)
            alpha = 1 - va / (va + vb)
            w[a] *= alpha
            w[b] *= 1 - alpha
    out = w.sort_index().values
    return _series(apply_cap(out / out.sum(), max_weight), returns.columns)


def fixed_weights(weights: dict):
    """Factory for static benchmarks (e.g. 60/40). Rebalanced to target on every rebalance date."""
    def f(returns: pd.DataFrame, **_) -> pd.Series:
        w = pd.Series(0.0, index=returns.columns)
        for k, v in weights.items():
            w[k] = v
        return w / w.sum()
    return f


STRATEGIES = {
    "Equal Weight": equal_weight,
    "Inverse Volatility": inverse_vol,
    "Min Variance": min_variance,
    "Max Sharpe": max_sharpe,
    "Risk Parity": risk_parity,
    "HRP": hrp,
}


# ----------------------------------------------------------------------------- efficient frontier
def efficient_frontier(mu: np.ndarray, cov: np.ndarray, n_points: int = 40, max_weight: float = 1.0) -> pd.DataFrame:
    """Long-only efficient frontier: minimum variance for each target return (annualised inputs)."""
    n = len(mu)
    w_minvar = _solve(lambda w: w @ cov @ w, lambda w: 2 * cov @ w, n, max_weight)
    r_lo = float(w_minvar @ mu)
    lp = linprog(-mu, A_eq=np.ones((1, n)), b_eq=[1.0], bounds=[(0, max_weight)] * n)
    r_hi = float(-lp.fun)
    rows = []
    for target in np.linspace(r_lo, r_hi, n_points):
        res = minimize(lambda w: w @ cov @ w, w_minvar, jac=lambda w: 2 * cov @ w, method="SLSQP",
                       bounds=[(0, max_weight)] * n,
                       constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0},
                                    {"type": "eq", "fun": lambda w, t=target: w @ mu - t}],
                       options={"maxiter": 300, "ftol": 1e-12})
        if res.success:
            rows.append({"return": float(res.x @ mu), "vol": float(np.sqrt(res.x @ cov @ res.x))})
    return pd.DataFrame(rows)
