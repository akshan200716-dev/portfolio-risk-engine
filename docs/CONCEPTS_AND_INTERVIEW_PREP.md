# Concepts and interview prep (for you, not necessarily for GitHub)

If you can answer these in your own words, you understand the project well enough to defend it.

## Core ideas in plain English
- **Sharpe ratio**: return above the risk-free rate per unit of volatility. Higher is better, but it is noisy: ~18 years of data
  gives a 95% interval roughly ±0.5 wide, which is why the dashboard shows bootstrap intervals.
- **Sortino / Calmar**: Sortino only penalises downside volatility; Calmar is CAGR divided by max drawdown.
- **Value at Risk (VaR)**: the loss you should only exceed on 1% (or 5%) of days. **Expected Shortfall (ES)**: the *average* loss on
  those bad days. ES sees the tail; VaR only marks its edge. Basel moved from VaR to ES for this reason.
- **Covariance shrinkage (Ledoit-Wolf)**: with 12 assets and 252 days, the sample covariance is noisy and optimisers exploit the noise.
  Shrinking toward a simple target trades a little bias for much less variance.
- **Risk parity**: choose weights so every asset contributes equally to portfolio *risk*, not capital. A 60/40 portfolio holds
  60% of capital in equities but takes ~90%+ of its risk there.
- **HRP**: clusters correlated assets, then splits risk top-down. It never inverts the covariance matrix, so it is stable when
  the matrix is noisy.
- **Look-ahead bias**: using information that would not have been available at decision time. Signals use data through day *t* and
  trade at *t+1*; the controls prove this by scrambling the future.
- **Kupiec test**: checks whether a 99% VaR model is breached ~1% of the time. Too many breaches means it understates risk.
- **Walk-forward**: at each rebalance, estimate on the trailing window only, trade, move on. No peeking, no refitting on the full sample.

## Questions you should expect (and where the answer lives)
1. *Why did you include max Sharpe if it performs badly?* To show estimation error empirically: turnover and unstable weights are in the weights chart and turnover column.
2. *How did you avoid look-ahead bias?* Causal slicing, execution lag, and the perturbation control (`controls.check_no_lookahead`).
3. *Is risk parity better than 60/40?* Lower drawdown and volatility, but unlevered it earns less return; compare Sharpe **intervals** before claiming it is better.
4. *Why does normal VaR fail?* Returns are fat-tailed and volatility clusters; see the Kupiec results and the Cornish-Fisher comparison.
5. *What are the weaknesses?* Hindsight universe, flat cost model, in-sample frontier, short regime history (the Limitations section lists all of them; naming them yourself is a strength).
6. *What would you do next?* Regime-aware covariance (EWMA/DCC), turnover-penalised optimisation, factor-based risk model, out-of-sample universe test.
7. *How do you know the backtest engine is correct?* Independent matrix recomputation of every return, cost reconciliation, and 62 tests including ones that deliberately corrupt results to confirm the controls fail.
