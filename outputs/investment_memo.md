# Investment memo: multi-asset portfolio construction

**Question.** Which long-only construction method gives the best risk-adjusted outcome across a diversified ETF universe,
and how much of that outcome survives realistic costs and statistical scrutiny?

**Setup.** 12 ETFs, Jun 2007 to Dec 2025. Walk-forward with a 252-day estimation window,
monthly rebalancing, trades executed 1 day after the signal, 10 bp per unit traded, 40% position cap.
Covariance by Ledoit-Wolf shrinkage. Benchmarks (60/40 and SPY) run through the identical engine.

## Results

| Strategy | CAGR | Vol | Sharpe | Max DD | Calmar | Turnover/yr |
|---|---:|---:|---:|---:|---:|---:|
| Equal Weight | 7.1% | 12.6% | 0.45 | -34.5% | 0.20 | 33% |
| Inverse Volatility | 5.8% | 8.9% | 0.46 | -23.9% | 0.24 | 45% |
| Min Variance | 3.5% | 6.0% | 0.26 | -16.7% | 0.21 | 90% |
| Max Sharpe | 5.4% | 16.2% | 0.28 | -46.8% | 0.12 | 488% |
| Risk Parity | 5.4% | 7.8% | 0.45 | -20.4% | 0.26 | 52% |
| HRP | 4.4% | 6.7% | 0.38 | -19.8% | 0.22 | 131% |
| 60/40 | 8.8% | 12.0% | 0.59 | -30.1% | 0.29 | 22% |
| SPY Buy & Hold | 12.1% | 19.9% | 0.57 | -47.2% | 0.26 | 0% |

## Findings

1. **Ranking.** Inverse Volatility had the highest Sharpe ratio (0.46) and Min Variance the shallowest drawdown (-16.7%). The 60/40 benchmark delivered a Sharpe of 0.59 with a -30.1% drawdown.

2. **Statistical caution.** The Risk Parity Sharpe interval (-0.01 to 1.00) overlaps the 60/40 interval (0.18 to 1.09). The ranking between them is therefore not statistically reliable on this sample.

3. **Optimiser instability.** Max Sharpe turned over 488% of the portfolio per year (cost drag 0.52% a year) versus 52% for Risk Parity. Return-chasing optimisers amplify estimation error.

4. **Concentration of risk.** In the 60/40 mix, SPY contributes 94% of portfolio volatility on 60% of the capital. Risk Parity's largest single contribution is 9%.

5. **VaR model validation (Risk Parity, 99%).** Normal-distribution VaR was breached 76 times against 42 expected (Kupiec p = 0.000; model rejected). Historical VaR: 53 breaches (p = 0.086; model accepted).

6. **Stress.** Today's 60/40 holdings lose 27.2% in the 'Global financial crisis' replay, versus 20.2% for Risk Parity.

7. **Diversification regime risk.** Six-month SPY-AGG correlation averaged -0.13 over the full sample but +0.08 during the 2022 rate shock, when bonds stopped hedging equities.

## Data quality and model controls

Data checks: 7 pass, 1 warn, 0 fail. All 62 independent controls passed.
Controls include a future-data perturbation test (all data after a cutoff replaced with noise; earlier weights and returns must not change)
and an independent matrix recomputation of every net return.

## Limitations

Universe chosen with hindsight (survivorship and selection bias); flat cost model with no spread, impact or tax; long-only and unlevered,
so low-volatility strategies cannot match equity returns; the efficient frontier is in-sample; one 19-year sample
contains few independent market regimes, so small Sharpe differences should not be over-interpreted.
