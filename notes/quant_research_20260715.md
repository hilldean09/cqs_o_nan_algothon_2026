# Quant Research Findings — 15 July 2026

## Decision

Use `src/CQS_O_NaN_candidate.py` as the next leaderboard candidate, not the
currently active moving-average code.  It is a self-contained NumPy-only
submission algorithm.  It passed compilation and an exact replay of the
official evaluation loop.

**Important:** its 865.88 local score is a score on the supplied last 250
days, so it is not evidence of an 865.88 score on the unseen General Round.
It should be treated as a diagnostics check, not as a parameter-optimisation
target.  The strategy deliberately uses an online regime selector so that it
can adapt rather than assuming the supplied regime persists.

## Dataset facts

- `prices.txt` has 500 daily closes for 51 assets, with no missing or
  non-positive observations.
- Daily log returns have a common component: average off-diagonal return
  correlation is 0.20 and the first correlation-PC has eigenvalue 11.55
  (22.7% of total standardised variance).  The effective dimensionality is
  about 15, not 51 independent bets.
- This factor structure is persistent: pairwise return correlations in the
  first and second halves have correlation 0.64.
- ALGO is a useful, low-cost factor proxy, but it is not the only one.  Its
  highest contemporaneous correlations include ILVX (0.59), BENI (0.53), and
  CUBO (0.52).
- The common factor's short-horizon behaviour changes regime.  Its lag-one
  autocorrelation is about +0.12 in days 0–249 and -0.07 in days 250–499.
  Fixed momentum, moving-average, or reversal parameters therefore have a
  high regime-overfit risk.

## Backtest discipline

`src/research.py` is a clean walk-forward harness.  It reproduces position
limits, integer shares, the evaluator's delayed commission accounting, and
the official score.  A signal sees only prices available at that close.

The active `CQS_O_NaN.py` had a syntax error in the unused lag helper
(`elif l = 0`); this has been corrected.  Its current moving-average strategy
compiles and scores only **15.64** on the supplied last-250 evaluation.

The following results are all exact local replays of the supplied data, and
are **not** estimates of the next unseen period:

| Strategy | Last-250 score | Annualised Sharpe | Comment |
|---|---:|---:|---|
| Active moving-average crossover | 15.64 | 1.66 | Weak exposure and no regime handling |
| Per-asset 5-day reversal | 381.3 | 2.3 | Good local result, but trades idiosyncratic noise |
| Fixed 2-day standardised market reversal | 785.0 | 2.4 | Very strong but regime-sensitive |
| Online factor-regime ensemble (candidate) | **865.88** | **2.87** | Selected for unseen-data robustness |

The candidate's contiguous walk-forward checks are positive in all four
100-day blocks: 186.63 (days 100–199), 199.12 (200–299), 1110.32 (300–399),
and 671.31 (400–499).  These overlapping checks are a useful stress test but
remain one simulated history, so they are not a substitute for the General
Round data.

## Candidate design

1. Estimate a broad factor each day by standardising every asset's return by
   its past 60-day volatility and averaging across assets.
2. Generate 42 simple, interpretable experts: 1–10 day reversal and momentum
   forecasts from both that broad factor and ALGO, plus permanent long and
   short fallbacks.
3. Re-score experts using only their preceding 30 realised daily PnLs.  PnL is
   measured in the same dollar-limit weighting as the official evaluator.
4. Trade the single best recent expert only when its trailing information ratio
   exceeds 0.30.  Otherwise hold zero exposure.  This avoids forcing a trade
   when the data say the regime is unclear.
5. When confident, hold every asset at its permitted dollar cap in the chosen
   direction.  This is intentional: there is no total portfolio budget, all
   assets load positively on the common factor, and the strategy then earns a
   diversified factor return rather than 51 noisy independent forecasts.

No external data or non-standard package is used.  The exact local official
loop replay completed in about two seconds and generated $96.3m of cumulative
trade volume, comfortably above the activity minimum.

## Do next

1. Submit the candidate to the test leaderboard only after copying it to the
   required registered-team filename and zipping that one Python file.  Do not
   include `requirements-dev.txt`.
2. Record the leaderboard score, code hash, and timestamp.  Treat it as one
   noisy observation; do not tune dozens of parameters to the public score.
3. When the General Round dataset arrives, run the same walk-forward table
   before changing any hyperparameter.  Keep the online expert selection,
   because it can re-select momentum, reversal, or cash from new history.
4. Compare the candidate against two ablations on the new data: fixed
   factor-reversal and ALGO-only.  Promote a change only if it wins across
   multiple chronological holdout blocks and the leaderboard, net of costs.

## Rules relevant to submission

The competition documentation confirms the 51-asset universe, per-asset
limits, no total portfolio budget, commissions, a 10-minute evaluation limit,
and a $25,000 minimum trade-volume rule.  It also confirms that the next-round
prices are unseen and submissions cannot read external data or use networking.

- [Challenge brief](https://wiki.algothon.au/rules/)
- [Submission guide](https://wiki.algothon.au/submission/)
- [Dataset schedule](https://wiki.algothon.au/schedule/)
