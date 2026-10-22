# Multi-Asset Portfolio Construction & Risk Analytics Engine

A research framework that builds portfolios six different ways, backtests them walk-forward with realistic
trading costs, measures their risk, stress-tests them, and then **audits its own results**.
It produces a self-contained HTML research dashboard and an auto-written investment memo.

```
python run_analysis.py --source yfinance     # real data
open outputs/dashboard.html
```

## What it does

| Area | What is implemented |
|---|---|
| **Portfolio construction** | Equal weight, inverse volatility, minimum variance, maximum Sharpe, risk parity (Spinu convex formulation) and Hierarchical Risk Parity (López de Prado), all long-only with position caps |
| **Estimation-error control** | Ledoit-Wolf covariance shrinkage (implemented from scratch and verified against scikit-learn), shrunk expected returns, weight caps |
| **Backtesting** | Walk-forward, monthly or quarterly rebalance, signal at close *t* and trade at *t+1*, weight drift, transaction costs, 60/40 and S&P 500 benchmarks run through the same engine |
| **Performance** | CAGR, volatility, Sharpe, Sortino, Calmar, drawdown tables, alpha/beta, information ratio, up/down capture, **block-bootstrap Sharpe confidence intervals** |
| **Market risk** | VaR and Expected Shortfall (historical, parametric, Cornish-Fisher, Monte Carlo with Student-t), component risk contributions, rolling volatility, stock-bond correlation regimes |
| **Model validation** | Rolling out-of-sample VaR backtest with the **Kupiec POF test** and Basel traffic-light zones |
| **Stress testing** | Replay of five historical crises plus four hypothetical shocks on end-of-sample holdings |
| **Data quality** | Eight automated checks (integrity, missing, stale, extreme, gaps, history) and an **audit log** of every change made to the data |
| **Independent controls** | Weights sum to 100%, long-only, concentration cap, cost reconciliation, independent return recomputation, execution-lag check, and a **future-data perturbation test for look-ahead bias** |

## Quick start

```bash
pip install -r requirements.txt
python run_analysis.py --source yfinance            # downloads 12 ETFs, caches to data/
python run_analysis.py --source synthetic           # offline demo (simulated market, not real results)
python -m unittest discover -s tests -t .           # 62 tests
```

Options: `--start --end --lookback 252 --rebalance monthly|quarterly --tc-bps 10 --max-weight 0.4 --out outputs`

**Outputs** (`outputs/`): `dashboard.html`, `investment_memo.md`, and CSVs for metrics, daily returns, the full rebalance
ledger, controls, data checks, audit log, stress tests, risk contributions and the VaR backtest.

## Project layout

```
portfolio_engine/
  config.py        universe, stress scenarios, backtest settings
  data.py          yfinance / CSV loaders + regime-switching market simulator for offline use
  validation.py    data-quality checks and audited cleaning
  optimizers.py    six construction methods, covariance shrinkage, efficient frontier
  backtest.py      walk-forward engine with execution lag, drift and costs
  metrics.py       performance statistics and bootstrap confidence intervals
  risk.py          VaR/ES, drawdowns, risk decomposition, stress tests, Kupiec test
  controls.py      independent integrity controls for backtest results
  pipeline.py      runs everything once and returns a single Analysis object
  report.py        HTML dashboard         memo.py   auto-generated investment memo
tests/             62 unit tests incl. negative tests that prove the controls catch errors
run_analysis.py    command-line entry point
```

## Design decisions worth knowing

- **No look-ahead, and proof.** A strategy only ever receives `returns[t-L+1 : t]`. The control suite replaces all data after
  a cutoff with random noise and asserts that nothing before the cutoff changes. I also verified the control by deliberately
  injecting a 5-day look-ahead bug into the engine: it failed immediately.
- **Honest statistics.** Differences in Sharpe ratio between neighbouring strategies are usually inside sampling error, so the
  dashboard reports bootstrap intervals rather than just ranking point estimates.
- **Max-Sharpe is included as a cautionary tale.** It maximises estimation error and is expected to show unstable weights and
  high turnover; the report quantifies this rather than hiding it.
- **Fixed parameters.** Lookback, cost and cap are defaults chosen before looking at results, not tuned to the sample.

## Results

Real data (Yahoo Finance adjusted prices), June 2007 to December 2025, net of 10 bp trading costs:

| Strategy | CAGR | Vol | Sharpe | Max drawdown | Calmar | Turnover/yr |
|---|---:|---:|---:|---:|---:|---:|
| Equal Weight | 7.1% | 12.6% | 0.45 | -34.5% | 0.20 | 33% |
| Inverse Volatility | 5.8% | 8.9% | 0.46 | -23.9% | 0.24 | 45% |
| Min Variance | 3.5% | 6.0% | 0.26 | -16.7% | 0.21 | 90% |
| Max Sharpe | 5.4% | 16.2% | 0.28 | -46.8% | 0.12 | 488% |
| Risk Parity | 5.4% | 7.8% | 0.45 | -20.4% | 0.26 | 52% |
| HRP | 4.4% | 6.7% | 0.38 | -19.8% | 0.22 | 131% |
| 60/40 | 8.8% | 12.0% | 0.59 | -30.1% | 0.29 | 22% |
| SPY Buy & Hold | 12.1% | 19.9% | 0.57 | -47.2% | 0.26 | 0% |

**Takeaways:** Risk-based methods roughly halved the worst drawdown relative to the S&P 500, but a simple 60/40 mix had the best risk-adjusted return. Maximum-Sharpe optimisation was the least stable (488% annual turnover). Bootstrap Sharpe intervals overlap across most strategies, so small ranking differences are not statistically reliable.

## Limitations

Universe chosen with hindsight (survivorship/selection bias); flat 10 bp cost model with no spread, impact or tax; long-only and
unlevered; in-sample efficient frontier; hypothetical shocks are round-number assumptions; one ~18-year sample contains few
independent market regimes. The offline simulator exists for development and testing and must never be quoted as performance.
