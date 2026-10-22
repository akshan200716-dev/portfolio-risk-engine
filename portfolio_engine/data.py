"""Price loading.

Sources
-------
yfinance   : real adjusted-close prices (dividends reinvested). Needs internet + `pip install yfinance`.
csv        : a CSV with a date column and one price column per ticker.
synthetic  : a regime-switching multi-asset simulator (calm / crisis / inflation-shock regimes,
             fat-tailed shocks, regime-dependent correlations). Used for offline development and tests.
             NEVER quote synthetic results as real performance.
"""
from __future__ import annotations

import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .config import UNIVERSE

# group, calm annual drift, calm annual vol
_ASSET_PARAMS = {
    "SPY": ("equity", 0.150, 0.125), "QQQ": ("equity", 0.185, 0.165), "IWM": ("equity", 0.130, 0.175),
    "EFA": ("equity", 0.105, 0.135), "EEM": ("equity", 0.115, 0.170), "VNQ": ("equity", 0.130, 0.170),
    "AGG": ("gov", 0.035, 0.032), "TLT": ("gov", 0.050, 0.115), "LQD": ("credit", 0.050, 0.060),
    "HYG": ("credit", 0.085, 0.065), "GLD": ("real", 0.095, 0.130), "DBC": ("real", 0.045, 0.135),
}
# Long-run annualised return targets (approximate real-world ETF outcomes) used to pin sample means.
_TARGET_RETURN = {"SPY": 0.095, "QQQ": 0.130, "IWM": 0.070, "EFA": 0.045, "EEM": 0.040, "VNQ": 0.060,
                  "AGG": 0.028, "TLT": 0.030, "LQD": 0.042, "HYG": 0.055, "GLD": 0.080, "DBC": 0.005}
# Factor loadings on [equity, rates (bond-price up), credit, commodity]
_LOADINGS = {
    "SPY": [0.93, 0.00, 0.00, 0.05], "QQQ": [0.88, 0.00, 0.00, 0.03], "IWM": [0.85, 0.00, 0.12, 0.05],
    "EFA": [0.85, 0.00, 0.00, 0.12], "EEM": [0.74, 0.00, 0.00, 0.28], "VNQ": [0.62, 0.30, 0.10, 0.05],
    "AGG": [0.02, 0.85, 0.15, 0.00], "TLT": [0.00, 0.92, 0.00, 0.00], "LQD": [0.15, 0.62, 0.55, 0.00],
    "HYG": [0.45, 0.05, 0.75, 0.05], "GLD": [0.00, 0.15, 0.00, 0.35], "DBC": [0.28, 0.00, 0.00, 0.72],
}
# How strongly each asset responds to a shock whose size is expressed in terms of SPY's move.
_CRISIS_BETA = {"SPY": 1.0, "QQQ": 1.1, "IWM": 1.2, "EFA": 1.1, "EEM": 1.2, "VNQ": 1.3,
                "AGG": -0.05, "TLT": -0.30, "LQD": 0.20, "HYG": 0.55, "GLD": -0.20, "DBC": 0.90}
_RATESHOCK_BETA = {"SPY": 1.0, "QQQ": 1.4, "IWM": 1.0, "EFA": 0.9, "EEM": 0.9, "VNQ": 1.1,
                   "AGG": 0.65, "TLT": 1.55, "LQD": 0.95, "HYG": 0.45, "GLD": 0.05, "DBC": -0.95}
_VOL_MULT = {  # regime -> group -> volatility multiplier
    1: {"equity": 2.3, "gov": 1.5, "credit": 2.0, "real": 1.6},
    2: {"equity": 1.4, "gov": 2.0, "credit": 1.5, "real": 1.3},
}
# (start, end, regime, SPY total move over window). Regime 0 windows are rallies.
_SCHEDULE = [
    ("2008-09-02", "2009-03-09", 1, -0.42),
    ("2009-03-10", "2010-04-23", 0, 0.75),
    ("2011-07-25", "2011-10-03", 1, -0.17),
    ("2018-09-20", "2018-12-24", 1, -0.19),
    ("2020-02-19", "2020-03-23", 1, -0.34),
    ("2020-03-24", "2020-12-31", 0, 0.60),
    ("2022-01-03", "2022-10-14", 2, -0.20),
    ("2022-10-17", "2023-12-29", 0, 0.35),
]


def _correlation(tickers, regime: int) -> np.ndarray:
    B = np.array([_LOADINGS[t] for t in tickers], dtype=float)
    if regime == 1:  # crisis: risk assets converge, Treasuries/gold become havens
        for i, t in enumerate(tickers):
            if _ASSET_PARAMS[t][0] == "equity":
                B[i, 0] = min(B[i, 0] * 1.06, 0.97)
        for t, v in {"TLT": -0.30, "AGG": -0.12, "GLD": 0.05}.items():
            if t in tickers:
                B[tickers.index(t), 0] = v
        for t in ("HYG", "LQD"):
            if t in tickers:
                B[tickers.index(t), 2] *= 1.1
    elif regime == 2:  # inflation shock: stocks and bonds fall together
        for t, v in {"TLT": 0.38, "AGG": 0.34, "LQD": 0.40, "HYG": 0.55}.items():
            if t in tickers:
                B[tickers.index(t), 0] = v
    psi = np.clip(1.0 - (B ** 2).sum(axis=1), 0.03, None)
    C = B @ B.T + np.diag(psi)
    d = np.sqrt(np.diag(C))
    return C / np.outer(d, d)


def generate_synthetic_prices(tickers=None, start="2007-06-01", end="2025-12-31",
                              seed: int = 7, df: float = 5.0, calibrate_means: bool = True) -> pd.DataFrame:
    """Regime-switching, fat-tailed multi-asset price simulator (deterministic given `seed`)."""
    tickers = list(tickers or UNIVERSE)
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end)
    n, p = len(idx), len(tickers)

    regime = np.zeros(n, dtype=int)
    in_scheduled = np.zeros(n, dtype=bool)
    mu_daily = np.tile(np.array([_ASSET_PARAMS[t][1] for t in tickers]) / 252.0, (n, 1))

    # random short stress episodes (calm <-> crisis Markov chain)
    state = 0
    for i in range(n):
        if state == 0 and rng.random() < 0.0012:
            state = 1
        elif state == 1 and rng.random() < 0.06:
            state = 0
        regime[i] = state
    crisis_beta = np.array([_CRISIS_BETA[t] for t in tickers])
    random_stress = regime == 1
    mu_daily[random_stress] = crisis_beta * (-0.04 / 17.0)

    # scheduled historical-style episodes override the random chain
    for s, e, reg, spy_move in _SCHEDULE:
        mask = (idx >= pd.Timestamp(s)) & (idx <= pd.Timestamp(e))
        k = int(mask.sum())
        if k == 0:
            continue
        beta = np.array([(_RATESHOCK_BETA if reg == 2 else _CRISIS_BETA)[t] for t in tickers])
        mu_daily[mask] = beta * np.log1p(spy_move) / k
        regime[mask] = reg
        in_scheduled |= mask

    vol_calm = np.array([_ASSET_PARAMS[t][2] for t in tickers]) / np.sqrt(252.0)
    groups = [_ASSET_PARAMS[t][0] for t in tickers]
    chol = {r: np.linalg.cholesky(_correlation(tickers, r)) for r in (0, 1, 2)}
    vmult = {r: np.array([1.0 if r == 0 else _VOL_MULT[r][g] for g in groups]) for r in (0, 1, 2)}

    rets = np.empty((n, p))
    scale_t = np.sqrt((df - 2.0) / df)
    for i in range(n):
        r = regime[i]
        z = chol[r] @ rng.standard_normal(p)
        mix = min(np.sqrt(df / rng.chisquare(df)), 6)  # joint fat tails (multivariate Student-t), tail-capped
        rets[i] = mu_daily[i] + vol_calm * vmult[r] * z * mix * scale_t
    rets = np.clip(rets, -0.15, 0.15)
    if calibrate_means:  # shift log-returns so each asset's long-run CAGR hits its target; shape is untouched
        logr = np.log1p(rets)
        target = np.log1p(np.array([_TARGET_RETURN[t] for t in tickers])) * n / 252.0
        logr += (target - logr.sum(axis=0)) / n
        rets = np.expm1(logr)
    prices = 100.0 * np.cumprod(1.0 + rets, axis=0)
    return pd.DataFrame(prices, index=idx, columns=tickers)


def _cache_path(cache_dir, tickers, start, end) -> Path:
    key = hashlib.md5(f"{sorted(tickers)}|{start}|{end}".encode()).hexdigest()[:10]
    return Path(cache_dir) / f"prices_{key}.csv"


def load_prices(tickers=None, start="2007-06-01", end="2025-12-31", source="auto",
                csv_path=None, cache_dir="data", seed=7):
    """Return (prices, source_label). `source_label` is carried through to the report."""
    tickers = list(tickers or UNIVERSE)

    if source == "csv":
        px = pd.read_csv(csv_path, index_col=0, parse_dates=True).sort_index()
        return px[[t for t in tickers if t in px.columns]], f"csv:{Path(csv_path).name}"

    if source in ("auto", "yfinance"):
        cache = _cache_path(cache_dir, tickers, start, end)
        if cache.exists():
            return pd.read_csv(cache, index_col=0, parse_dates=True), "yfinance (cached)"
        try:
            import yfinance as yf
            raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False, threads=False)["Close"]
            raw = raw[tickers].dropna(how="all")
            if raw.empty:
                raise RuntimeError("yfinance returned no data")
            Path(cache_dir).mkdir(exist_ok=True)
            raw.to_csv(cache)
            return raw, "yfinance"
        except Exception as exc:  # noqa: BLE001
            if source == "yfinance":
                raise
            warnings.warn(f"yfinance unavailable ({exc}); falling back to SYNTHETIC data. "
                          "Results are not real performance.", stacklevel=2)

    return generate_synthetic_prices(tickers, start, end, seed=seed), "SYNTHETIC"


def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how="all")
