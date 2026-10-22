"""Central configuration: investment universe, stress scenarios and backtest settings."""
from dataclasses import dataclass

# Multi-asset ETF universe: liquid, long-history proxies for the main asset classes.
UNIVERSE = {
    "SPY": "US large-cap equity",
    "QQQ": "US growth / tech equity",
    "IWM": "US small-cap equity",
    "EFA": "Developed ex-US equity",
    "EEM": "Emerging-market equity",
    "VNQ": "US real estate (REITs)",
    "AGG": "US aggregate bonds",
    "TLT": "Long-term US Treasuries",
    "LQD": "Investment-grade credit",
    "HYG": "High-yield credit",
    "GLD": "Gold",
    "DBC": "Broad commodities",
}

# Historical stress windows (start, end). Used for scenario replay on whatever data is loaded.
HISTORICAL_SCENARIOS = {
    "Global financial crisis": ("2008-09-02", "2009-03-09"),
    "US downgrade / euro debt": ("2011-07-25", "2011-10-03"),
    "Q4 2018 sell-off": ("2018-09-20", "2018-12-24"),
    "COVID crash": ("2020-02-19", "2020-03-23"),
    "2022 inflation & rate shock": ("2022-01-03", "2022-10-14"),
}

# Hypothetical shocks (illustrative instantaneous asset returns, not forecasts).
HYPOTHETICAL_SCENARIOS = {
    "Equity crash (-30%)": {
        "SPY": -0.30, "QQQ": -0.34, "IWM": -0.36, "EFA": -0.32, "EEM": -0.38, "VNQ": -0.35,
        "AGG": 0.02, "TLT": 0.08, "LQD": -0.03, "HYG": -0.14, "GLD": 0.03, "DBC": -0.15,
    },
    "Rates +200bp": {
        "SPY": -0.08, "QQQ": -0.14, "IWM": -0.07, "EFA": -0.06, "EEM": -0.07, "VNQ": -0.15,
        "AGG": -0.08, "TLT": -0.30, "LQD": -0.10, "HYG": -0.04, "GLD": -0.05, "DBC": 0.00,
    },
    "Stagflation shock": {
        "SPY": -0.15, "QQQ": -0.20, "IWM": -0.15, "EFA": -0.14, "EEM": -0.12, "VNQ": -0.12,
        "AGG": -0.09, "TLT": -0.20, "LQD": -0.11, "HYG": -0.08, "GLD": 0.15, "DBC": 0.25,
    },
    "Credit event (spreads +400bp)": {
        "SPY": -0.15, "QQQ": -0.17, "IWM": -0.20, "EFA": -0.15, "EEM": -0.15, "VNQ": -0.15,
        "AGG": -0.02, "TLT": 0.05, "LQD": -0.09, "HYG": -0.20, "GLD": 0.03, "DBC": -0.08,
    },
}


@dataclass(frozen=True)
class BacktestConfig:
    start: str = "2007-06-01"
    end: str = "2025-12-31"
    lookback_days: int = 252        # estimation window for every optimizer
    rebalance: str = "monthly"      # monthly | quarterly
    tc_bps: float = 10.0            # one-way transaction cost, basis points of traded notional
    max_weight: float = 0.40        # long-only, single-asset cap
    execution_lag: int = 1          # signal at close t, trade at close t+lag (no same-bar execution)
    risk_free_annual: float = 0.02
    trading_days: int = 252
    featured_strategy: str = "Risk Parity"
