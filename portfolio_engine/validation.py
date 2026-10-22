"""Data-quality checks and an audited cleaning step.

Every check returns PASS / WARN / FAIL with a measurable value, and every change made to the data is
written to an audit log (what, where, why) so the cleaning is fully reproducible and reviewable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class DataQualityReport:
    checks: pd.DataFrame                      # one row per check: check, status, value, detail
    per_ticker: pd.DataFrame                  # one row per ticker: coverage and anomaly counts
    audit_log: pd.DataFrame                   # every modification applied to the raw data
    cleaned: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)

    @property
    def passed(self) -> bool:
        return not (self.checks["status"] == "FAIL").any()


def _max_run(mask: np.ndarray) -> int:
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def validate_prices(prices: pd.DataFrame, min_obs: int = 300, max_daily_move: float = 0.25,
                    stale_days: int = 5, max_missing_pct: float = 0.05, max_gap_days: int = 5,
                    mad_z: float = 12.0, ffill_limit: int = 3) -> DataQualityReport:
    checks, audit = [], []

    def add(name, status, value, detail):
        checks.append({"check": name, "status": status, "value": value, "detail": detail})

    px = prices.copy()

    # 1. index integrity
    dup = int(px.index.duplicated().sum())
    mono = bool(px.index.is_monotonic_increasing)
    add("Index unique & sorted", "PASS" if (dup == 0 and mono) else "FAIL", dup,
        f"{dup} duplicate dates; sorted={mono}")
    if dup or not mono:
        px = px[~px.index.duplicated(keep="last")].sort_index()
        audit.append({"ticker": "*", "date": "", "action": "dedupe+sort",
                      "reason": f"{dup} duplicate dates / unsorted index"})

    # 2. non-positive prices
    nonpos = (px <= 0).sum()
    add("Prices strictly positive", "PASS" if nonpos.sum() == 0 else "FAIL", int(nonpos.sum()),
        "tickers affected: " + (", ".join(nonpos[nonpos > 0].index) or "none"))
    px = px.mask(px <= 0)

    # 3. missing data
    miss = px.isna().mean()
    worst = float(miss.max())
    status = "PASS" if worst == 0 else ("WARN" if worst <= max_missing_pct else "FAIL")
    add("Missing observations", status, worst, f"worst ticker {miss.idxmax()} at {worst:.2%}")

    # 4. stale prices (identical consecutive closes)
    flat = px.diff().eq(0)
    runs = {t: _max_run(flat[t].values) for t in px.columns}
    worst_run = max(runs.values())
    add("Stale prices", "PASS" if worst_run < stale_days else "WARN", worst_run,
        f"longest unchanged run {worst_run} days ({max(runs, key=runs.get)})")

    # 5. extreme returns
    rets = px.pct_change(fill_method=None)
    extreme = (rets.abs() > max_daily_move)
    med = rets.median()
    mad = (rets - med).abs().median() * 1.4826
    robust = ((rets - med).abs() / mad.replace(0, np.nan)) > mad_z
    n_ext = int(extreme.sum().sum())
    add("Extreme daily moves (>|{:.0%}|)".format(max_daily_move), "PASS" if n_ext == 0 else "WARN", n_ext,
        "flagged for review, not removed")
    add(f"Robust outliers (MAD z>{mad_z:g})", "PASS" if robust.sum().sum() == 0 else "WARN",
        int(robust.sum().sum()), "informational: fat tails produce some; clusters merit review")
    for t in px.columns:
        for d in rets.index[extreme[t].fillna(False)]:
            audit.append({"ticker": t, "date": str(d.date()), "action": "flag",
                          "reason": f"|return| {rets.at[d, t]:+.1%} exceeds {max_daily_move:.0%}"})

    # 6. calendar gaps
    gaps = px.index.to_series().diff().dt.days.dropna()
    mg = int(gaps.max()) if len(gaps) else 0
    add("Calendar gaps", "PASS" if mg <= max_gap_days else "WARN", mg, f"largest gap {mg} calendar days")

    # 7. sample size
    add("Sufficient history", "PASS" if len(px) >= min_obs else "FAIL", len(px),
        f"{len(px)} rows (minimum {min_obs})")

    # cleaning: drop tickers with too much missing data, then limited forward-fill (never back-fill)
    drop = miss[miss > max_missing_pct].index.tolist()
    for t in drop:
        audit.append({"ticker": t, "date": "", "action": "drop_ticker",
                      "reason": f"{miss[t]:.1%} missing exceeds {max_missing_pct:.0%}"})
    cleaned = px.drop(columns=drop)
    before = cleaned.isna().sum()
    cleaned = cleaned.ffill(limit=ffill_limit)
    filled = (before - cleaned.isna().sum())
    for t, k in filled[filled > 0].items():
        audit.append({"ticker": t, "date": "", "action": "ffill",
                      "reason": f"{int(k)} gaps forward-filled (limit {ffill_limit} days)"})
    # rows where any asset is still missing are removed so every asset shares one calendar
    bad_rows = cleaned.isna().any(axis=1)
    if bad_rows.any():
        audit.append({"ticker": "*", "date": "", "action": "drop_rows",
                      "reason": f"{int(bad_rows.sum())} rows with unresolved gaps removed (common-calendar rule)"})
    cleaned = cleaned.loc[~bad_rows]

    per_ticker = pd.DataFrame({
        "first_date": px.apply(lambda s: s.first_valid_index()),
        "missing_pct": miss,
        "max_stale_run": pd.Series(runs),
        "extreme_moves": extreme.sum(),
        "max_abs_return": rets.abs().max(),
    })
    return DataQualityReport(pd.DataFrame(checks), per_ticker,
                             pd.DataFrame(audit, columns=["ticker", "date", "action", "reason"]), cleaned)
