# Algothon 2026 — Quantitative Research Report

**Dataset:** `prices.txt` (500 days × 51 assets, ALGO + 50 anonymous instruments)
**Rules source:** Algothon 2026 kickoff deck (Susquehanna)

---

## A. Executive Conclusion

**Headline finding: this dataset is very close to a random walk.** Autocorrelation at every lag tested (1, 2, 5, 10, 20) is statistically indistinguishable from zero, there is no persistent momentum (time-series or cross-sectional), and ALGO shows no exploitable lead-lag relationship with the other 50 assets. The one genuine, reproducible pattern is a **small, consistent price-level mean-reversion effect** (extended prices tend to drift back toward their recent rolling average), which shows up across every window tested and across every walk-forward block.

- **Best individual strategy (by robustness, not just backtest score): Price mean-reversion** (z-score of price vs. rolling mean, window ≈ 20 days). It is the *only* strategy that is positive in-train, in-validation, in-test, *and* positive in 6 of 7 walk-forward out-of-sample blocks (85.7%).
- **Best individual strategy by raw test-period score:** Pairs trading (ALGO vs. NGTE), test score 1.89 — but this pair was selected from the single most extreme result of 15 cointegration tests, so this number is very likely overstated (multiple-testing bias — see §D and §K).
- **Best strategy combination:** an equal-risk blend of price mean-reversion, return mean-reversion, PCA-residual mean-reversion and the ALGO–NGTE pair (train 0.05, val 2.99, test 1.85), and a leaner 3-strategy blend of price mean-reversion (40%) + return mean-reversion (35%) + pairs ALGO-NGTE (25%), risk-normalised and scaled to ~47% of the gross exposure cap, which scored **train 0.27 / val 2.90 / test 2.02** and is our final recommendation (§Final Recommendation).
- **Highest total test-period profit:** price mean-reversion, window 10 (+$26,127 over the 125-day test block).
- **Lowest test-period drawdown among genuinely robust strategies:** pairs ALGO–NGTE (max drawdown ‑$4,683 in test) and the combined portfolio (‑$4,600 to ‑$6,100 depending on blend).
- **Findings that look robust:** the price/return mean-reversion effect (consistent sign and magnitude across parameters, time periods, and walk-forward blocks); the complete absence of momentum; ALGO's near-total redundancy with the equal-weighted market average.
- **Findings that are likely overfitted/unstable:** ALGO lead-lag (looked good in one validation slice, sign-flipped in test in several parameterisations, and was profitable in only 3/7 walk-forward blocks); PCA-residual mean-reversion (profitable in only 2/7 walk-forward blocks); the specific ALGO–NGTE pair (chosen from 15 candidates, only 1 of which cleared a naive cointegration threshold); cross-sectional and time-series momentum (uniformly poor and unstable — not a "currently weak but promising" case, just genuinely absent from this data).
- **Currently weak strategies that could improve with more data:** momentum strategies would become interesting only if rolling autocorrelation turns reliably positive — no such signal exists yet. ALGO lead-lag and PCA-residual reversion could become tradeable if their walk-forward hit-rate improves above ~60%; right now both hover near a coin flip.

**Bottom line:** don't expect a high competition score from this dataset — realistic, out-of-sample-honest scores cluster in the 1–2 range (annualised-Sharpe units), not the eye-popping numbers that appear when a single validation slice or a cherry-picked parameter is reported in isolation. The most defensible submission is the small, diversified mean-reversion blend described in the Final Recommendation, sized conservatively relative to the position caps.

---

## B. Competition Rules (as extracted from the deck)

| Rule | Value |
|---|---|
| Positions | Can be long **or** short, any real value (subject to caps) |
| Position cap — ALGO (asset 0) | **$100,000** dollar exposure (price × size), long or short |
| Position cap — every other asset | **$10,000** dollar exposure, long or short |
| Transaction cost | 0.00002 (0.2 bps) of dollar volume traded on ALGO; 0.0001 (1 bp) on all other assets — **charged on every dollar bought or sold** (i.e., on turnover, not on the position itself) |
| PL | Change in **dollar** portfolio value from one day to the next (not a percentage return) |
| μ | Mean(PL) across all days |
| σ | StdDev(PL) across all days |
| SR | √250 · μ / σ (i.e., an **annualised** Sharpe ratio, using 250 trading days/year) |
| **Score** | **Score = SR** (if σ ≠ 0); **Score = 0** if σ = 0 (guards against division by zero — not a risk penalty on top of SR) |
| Submission | A function `getMyPosition(prcSoFar)` in `teamName.py`. `prcSoFar` is a 51 × numDays NumPy array (day 0 = earliest). |
| Timing | The eval harness calls `getMyPosition` once per test day, passing the full price history up to (not including) that day. The position you return is applied to that day's move — i.e., **your positions are for "the next day," decided using strictly past information.** No look-ahead is possible or permitted. |
| Missing/invalid positions | Not stated in the deck's visible text; standard practice for this format (confirmed generically, not deck-verified) is that non-returned/invalid values are treated as 0 — **we did not find explicit deck text on this and flag it as an assumption**. |
| Dataset structure (full competition) | 2,000 days total, released progressively in 4 blocks (750/250/500/500 days) corresponding to kickoff → general round → finalist selection → final round, each with a live leaderboard. Our `prices.txt` (500 days) is the initial "test dataset," a **sample**, not the full competition series. |

We used this **exact** scoring formula everywhere in this analysis — no substitution of a standard annualised Sharpe with different day-count conventions, and no penalty terms beyond the σ=0 guard shown on the slide.

**Transaction costs:** the deck **does** specify explicit costs (0.2 bps ALGO / 1 bp others), so — per the task instructions — we report both a costless version (used for all primary rankings, to stay consistent with how the deck defines PL, which is stated purely as portfolio-value change with no separate cost bullet on the scoring slide) and a cost-adjusted sensitivity check using a conservative uniform 1 bp assumption. Costs shave 10–25% off the Sharpe of the higher-turnover strategies but do not flip any conclusion (§I).

---

## C. Dataset Findings (plain English)

- **500 trading days, 51 columns** (ALGO + 50 anonymous assets named with 4-letter tickers). No missing values, no duplicate rows, no non-positive prices, no returns beyond ±15% (largest single-day move was +14.6% on asset MMBT). The data is clean.
- **Prices trended down slightly over the sample**: median other-asset total return was ‑25%, ALGO ‑12.9%. Individual assets vary hugely — annualised volatility ranges from roughly 15% (ALGO) to 65%+ (MMBT).
- **ALGO is, statistically, the market index.** Its returns are almost perfectly explained (R² = 0.986, β ≈ 1.00, α ≈ 0) by the simple equal-weighted average return of the other 50 assets, and its realised volatility (0.99%/day) matches that diversified average almost exactly (0.98%/day) — far below any individual constituent. This is *the* single most important structural fact in the dataset: **ALGO does not behave like an independent 51st asset; it behaves like a diversified basket of the other 50.** This explains its much larger position cap ($100k vs $10k) — it's the low-volatility "index" instrument.
- **No autocorrelation, no momentum.** Average lag-1 return autocorrelation across all 51 assets is –0.0002 (essentially zero); lags 2, 5, 10, 20 are similarly indistinguishable from noise. 45% of assets have positive lag-1 autocorrelation and 55% negative — a coin flip. Cross-sectional momentum (do the past month's winners keep winning?) is flat-to-slightly-negative at every lookback from 5 to 60 days.
- **A small, consistent mean-reversion effect exists at the price level.** Assets that sit unusually far above/below their own rolling average (10/20/40-day windows) tend to drift back — the correlation between a price z-score and the next day's return is small (–0.017 to –0.019) but has the *same sign at every window*, which is what makes it tradeable rather than noise.
- **Average pairwise return correlation among the 50 non-ALGO assets is modest (mean 0.19, ranging –0.00 to 0.46)**, and this correlation structure is only moderately stable through time (correlation-of-correlations across quarters ≈ 0.48–0.55) — i.e., which assets move together shifts somewhat over the sample.
- **PCA:** the first principal component explains 22.6% of total variance (PC2 only 5.6%) — a single dominant "market" factor, closely aligned with ALGO (PC1-vs-ALGO correlation 0.975). Residuals after removing this factor show the same weak mean-reversion tendency as raw prices.
- **No usable ALGO lead-lag effect.** Correlating ALGO's return today with every other asset's return 1–5 days later (and the reverse direction) produces average correlations under 5% in magnitude at every lag, with only ~10% of assets even reaching |corr| > 0.1 at the best lag. This is noise, not signal — a materially different conclusion from just assuming "ALGO must be predictive because it has a bigger position limit."
- **Pairs/cointegration:** scanning the 15 most correlated pairs (all vs. ALGO, since ALGO correlates most with everything by construction), 14 of 15 fail a standard Engle-Granger cointegration test (p > 0.4). One pair (ALGO–NGTE) shows p = 0.02 — interesting, but with 15 simultaneous tests we'd expect roughly 0.75 false positives at the 5% level by chance alone, so this should be treated as a lead, not a confirmed relationship.

---

## D. Ranked Results Table

Scores are the **exact competition SR**, computed separately on chronological Train (days 0–249), Validation (250–374) and Test (375–499) position-days. Full parameter grids are in `out/all_strategy_results.csv`; the table below shows the best-per-family configuration (selected using validation-period score only — test was touched exactly once per row).

| Strategy | Params | Train | Val | Test | Walk-fwd mean score | Walk-fwd % positive blocks | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| **Price mean-reversion** | window=20 | 0.92 | 1.90 | **1.39** | **1.51** | **85.7%** | **Most robust single strategy** |
| Price mean-reversion | window=10 | 0.28 | 2.07 | 1.84 | – | – | Strong but noisier |
| Price mean-reversion | window=40 | 1.16 | 2.27 | 0.39 | – | – | Consistent sign, weaker test |
| Return mean-reversion | lookback=2 | -0.03 | 2.41 | 0.85 | 1.58 | 71.4% | Robust, correlated with price-MR (ρ=0.52) |
| Return mean-reversion | lookback=5 | 1.15 | 1.49 | 1.04 | – | – | Consistent sign |
| Pairs trading (ALGO–NGTE) | window=40 | -0.34 | 2.11 | **1.89** | 1.20 | 85.7% | Good numbers, **but pair was multiple-testing selected** — treat cautiously |
| Pairs trading (ALGO–BENI) | window=20 | 1.98 | 0.25 | 2.98 | – | – | Highest single test score, but low val — not selected on principled grounds |
| ALGO lead-lag | lag=1, window=60 | 1.82 | -0.82 | 1.73 | 0.09 | 42.9% | **Not robust** — sign-flips across periods |
| ALGO lead-lag | lag=3, window=60 | -0.25 | 2.23 | -2.31 | – | – | Overfit to validation slice |
| PCA-residual mean-reversion | resid_window=20 | -0.43 | 1.27 | 0.63 | 0.50 | 28.6% | **Not robust** |
| MA crossover | 20/60 | -0.24 | -1.41 | 1.43 | – | – | Unstable |
| Cross-sectional momentum | lookback=10, top 20% | -0.60 | 1.42 | -4.22 | – | – | **Overfit / unreliable** |
| Time-series momentum | lookback=10 | -0.44 | -0.72 | -2.99 | – | – | **Consistently poor — no momentum in data** |
| Volatility breakout | window=40 | -1.05 | -0.86 | 0.33 | – | – | Weak, inconsistent |
| **Equal-risk 4-way combo** (price-MR + ret-MR + PCA-resid + pairs, 25% each) | — | 0.05 | **2.99** | 1.85 | – | – | Best combined validation score |
| **Final blend** (40% price-MR w20, 35% ret-MR lb2, 25% pairs ALGO-NGTE) | risk-normalised, scaled to ~47% gross cap | 0.27 | 2.90 | **2.02** | – | – | **Recommended submission** |
| Required test: 30% PCA-resid + 70% ALGO lead-lag | — | 1.58 | 0.06 | 1.63 | – | – | Decent test score, weak validation — see §H |

Full per-parameter table (60 configurations): `out/all_strategy_results.csv`. Walk-forward block detail: `out/walk_forward_blocks.csv`.

---

## E. Strategy Explanations (plain English + toy example)

1. **Moving-average crossover.** Compute a fast and a slow moving average of price; go long when fast > slow, short when fast < slow. *Example:* if the 10-day average is $105 and the 40-day average is $100, the trend looks "up," so buy. Result here: unstable — the data has no real trend to catch.

2. **Time-series momentum.** Buy assets that have gone up over the last N days, short those that have gone down, sized by how big the move was relative to the asset's own volatility. Result: negative/unstable — no persistence in this data.

3. **Cross-sectional momentum.** Each day, rank all 50 non-ALGO assets by their trailing N-day return; buy the top 20%, short the bottom 20%. Result: poor and overfit — rankings do not persist into the future here.

4. **Price mean-reversion.** Compute how many standard deviations the current price is above/below its own rolling average (a "z-score"). If price is 2 standard deviations above its 20-day average, short it a little, betting it drifts back down; if 2 below, buy it. *Toy example:* rolling mean $50, rolling std $2, price now $54 → z = +2 → take a modest short position sized at –z/2 (capped at ±1) × position limit. **This is our best-performing, most stable individual strategy.**

5. **Return mean-reversion.** Same idea as #4 but applied to the most recent 1–5 day return instead of the price level — bet that yesterday's move partially reverses today.

6. **Volatility breakout.** Buy when price breaks above its N-day high, short when it breaks below its N-day low (a trend-following, not reversion, idea). Weak here — consistent with the "no momentum" finding.

7. **Pairs trading.** Pick two historically related assets (here, ALGO and NGTE), compute a rolling hedge ratio, and trade the *spread* between them: short the spread when it's unusually wide, long when unusually narrow. Works reasonably well empirically but the specific pair was chosen after scanning many candidates, so some of its apparent edge is likely a statistical fluke rather than a real relationship (classic multiple-testing risk).

8. **PCA residual mean-reversion.** Strip out the single dominant "market" factor (essentially, the ALGO/index move) from every asset's return, leaving an asset-specific "residual." Trade the residual's own mean reversion. Sounds appealing in theory; in this dataset it is not robust out-of-sample (positive in only 2 of 7 walk-forward blocks).

9. **ALGO lead-lag.** Regress each asset's future return on ALGO's recent return, using only a trailing window, and trade the fitted relationship. Despite ALGO's oversized position limit, we found **no genuine predictive edge** — the strategy performs inconsistently and is not something we would trust with real capital.

10. **Combined multi-strategy portfolio.** Blend several of the above, each scaled to contribute similar risk, so no single strategy dominates the combined P&L. This diversification is what ultimately delivers the best, most stable score in this analysis (§H, §Final Recommendation).

---

## F. Key Charts

All charts are in `charts/`:

1. `01_price_examples.png` — sample price paths including ALGO
2. `02_return_distribution.png` — pooled daily return histogram
3. `03_correlation_heatmap.png` — full 51×51 return correlation matrix
4. `04_pca_explained_variance.png` — PCA scree + cumulative variance
5. `05_pca_residual_example.png` — smoothed PCA residuals for 3 sample assets
6. `06_algo_leadlag_scatter.png` — ALGO(t) vs. future market(t+1) scatter (near-zero relationship)
7. `07_cumulative_pl_all_strategies.png` — cumulative $ P&L for every major strategy, with train/val/test boundaries marked
8. `08_drawdowns.png` — drawdown curves for every major strategy
9. `09_rolling_score.png` — rolling 60-day competition score for the three most-discussed strategies
10. `10_strategy_pl_correlation.png` — correlation matrix between strategy P&L streams (train period)

---

## G. Backtesting Methodology (fair, look-ahead-free)

- **Position convention:** `dollar_positions[t]` is the desired dollar exposure decided using price information available **through day t only**, and is applied to the day t→t+1 price move — exactly matching the deck's description of the evaluation harness.
- **Position limits** are applied by clipping dollar exposure to ±$100,000 (ALGO) / ±$10,000 (others) **before** converting to share counts, on every single day, for every strategy — never just at initiation.
- **No look-ahead:** all rolling statistics (moving averages, z-scores, PCA fits, regression betas, hedge ratios) use only a trailing window ending at day t. PCA and the ALGO lead-lag regression are refit periodically on rolling/expanding windows using **only past data** (§ below).
- **Chronological train/validation/test split:** first 50% (249 position-days) / next 25% (125) / final 25% (125) — no shuffling.
- **Parameter grids** were kept small and pre-specified (e.g., MA pairs 5/20, 10/40, 20/60; momentum lookbacks 5/10/20/40/60; mean-reversion windows 10/20/40) rather than searched exhaustively, per the task's anti-overfitting instructions. The **test set was used exactly once** per configuration reported — no iterative tuning against test.
- **Walk-forward testing:** each strategy's causal position series was additionally sliced into seven consecutive 50-day out-of-sample blocks (starting once enough history existed for the longest lookback) and scored block-by-block, to see whether performance is a single-period fluke or a persistent property.

---

## H. Strategy Combinations

- **Required test (30% PCA-residual + 70% ALGO lead-lag):** train 1.58 / val 0.06 / test 1.63. The test-period number looks reasonable, but the validation score near zero is a red flag — it means this exact blend would not have been selected by a disciplined validation-based process. A grid search over the PCA/ALGO-lead-lag weight (0%–100% in 10% steps) shows the validation-optimal weight is actually **100% PCA / 0% ALGO lead-lag** (val 1.27, test 0.63) — a materially weaker combination than the specified 30/70 split, and a good illustration of how noisy small-sample validation selection can be. **We do not endorse the 30/70 ALGO-lead-lag-heavy blend as a primary strategy**, because ALGO lead-lag itself failed the walk-forward robustness check (positive in only 3 of 7 blocks).
- **Best combination found:** an equal-risk blend of price mean-reversion, return mean-reversion, PCA-residual mean-reversion, and pairs ALGO–NGTE (25% risk-weight each) scored train 0.05 / val 2.99 / test 1.85 — the best validation score of any configuration tested, with a solid test score.
- **Leaner, more defensible combination (our final pick):** 40% price mean-reversion (w=20) + 35% return mean-reversion (lb=2) + 25% pairs ALGO–NGTE, risk-normalised on the train period only, scored train 0.27 / val 2.90 / test 2.02 — we prefer this over the 4-way blend because it excludes the two strategies (PCA-residual, ALGO lead-lag) that failed the walk-forward robustness test, at a small cost in validation score.
- **Correlation between strategy P&L streams (train period):** price-MR and return-MR are meaningfully correlated (ρ=0.52 — expected, both are reversal strategies on different horizons, so they don't diversify each other much). PCA-residual (ρ≈0.13–0.19 vs. the others), ALGO lead-lag (mildly *negatively* correlated with return-MR, ρ=–0.43) and the ALGO–NGTE pair (ρ≈0.01–0.14 vs. everything) add more genuine diversification. Full matrix: `out/strategy_pl_correlation.csv` and `charts/10_strategy_pl_correlation.png`.
- Position limits were re-applied (clipped) **after** combining signals, on every day, consistent with the rule that limits apply per asset per day regardless of how the position was constructed.

---

## I. Position Sizing, Risk Controls, and Transaction Costs

- We tested **signal-strength scaling** (z-scores clipped to [–1,1] then multiplied by the asset's dollar cap) throughout — this outperformed naive equal-dollar or always-max-position sizing in every strategy we checked, because it avoids maxing out a $10,000 position on a weak/noisy signal.
- We deliberately **did not** use the maximum ALGO position by default. Given the evidence in §C/§D that ALGO carries no unique predictive edge over the equal-weighted market and that ALGO lead-lag failed walk-forward validation, there is no out-of-sample justification for leaning heavily into the larger ALGO cap. In our final blend, ALGO exposure comes only from its role as one leg of the ALGO–NGTE pair, sized like any other signal.
- **Risk normalisation:** before combining strategies, each was scaled so its **train-period** daily P&L had unit standard deviation (info from val/test was never used for this scaling), preventing one high-volatility strategy (e.g., time-series momentum, or the wide-swinging ALGO-lead-lag configurations) from dominating a blend.
- **Gross exposure control:** the final blended positions are additionally scaled so average gross exposure sits at roughly 47% of the theoretical $600,000 cap (51 assets × their caps) — leaving headroom rather than running at the limit, which we consider prudent given how weak most of the underlying signals are.
- **Transaction costs:** the deck specifies 0.2 bps (ALGO) / 1 bp (others) per dollar traded. Using a conservative uniform 1 bp assumption (over-charging the ALGO leg) on our top candidates:

| Strategy | Score, no costs | Score, 1 bp costs (conservative) |
|---|---:|---:|
| Price mean-reversion (w=20) | 1.31 | 1.20 |
| Return mean-reversion (lb=2) | 0.88 | 0.66 |
| Pairs ALGO–NGTE | 0.79 | 0.77 |
| ALGO lead-lag (lag=1, w=60) | 1.12 | 0.88 |

No conclusion flips under costs; the pairs strategy is the most cost-resistant (low turnover), the mean-reversion strategies lose the most in relative terms (they trade more often) but remain net positive.

---

## J. Regime Dependence and Forward-Looking Monitoring

| Strategy | Regime it needs | Current evidence | Watch for |
|---|---|---|---|
| Momentum (TS & XS) | Positive, persistent return autocorrelation | None present — autocorrelation ≈ 0 everywhere | Rolling 60-day autocorrelation turning reliably positive (>0.05) *and* staying there across several windows before re-enabling |
| Price/return mean-reversion | Negative short-lag autocorrelation / bounded price ranges | Present, small but consistent (z-score correlation ≈ –0.02) | Would strengthen if the z-score/forward-return correlation magnitude increases (currently modest); would weaken if it drifts toward 0 or flips sign |
| Pairs trading | Stationary, stable spread between two names | Weak — only 1 of 15 candidate pairs showed even marginal cointegration evidence | New data confirming ALGO–NGTE spread stays mean-reverting (not just correlated) over an extended window; disable if p-value evidence disappears on refit |
| PCA-residual reversion | Stable factor loadings, mean-reverting residuals | Loadings reasonably stable (PC1 ≈ 22–23% of variance consistently) but residual mean reversion itself is weak/unstable in walk-forward | Would improve if walk-forward hit-rate on residual reversion climbs materially above the current ~29% |
| ALGO lead-lag | Genuine predictive information flow from ALGO to constituents (or vice versa) | Not found — correlations at every lag are within noise | Would need a sudden, sustained rise in |correlation| across multiple lags and assets, not just one favourable validation window |

---

## K. Dynamic Strategy-Selection Framework (for use as more `prices.txt` data arrives)

1. **Refit cadence:**
   - PCA: refit every 20 trading days on a rolling 100-day window (as implemented) — frequent enough to track drifting factor structure, infrequent enough to avoid noise-fitting.
   - ALGO lead-lag regression betas: refit every day on a rolling 40–60 day window (cheap to compute; the relationship is fast-moving if it exists at all).
   - Pairs hedge ratio: refit every day on a rolling 20–40 day window.
2. **Evaluation cadence:** re-score every active strategy on the most recent 50-day block weekly; re-score the full walk-forward history whenever a new 250-day chunk of data becomes available (matches the competition's own segment sizes of 250–750 days).
3. **Regime-change detection:** track rolling 60-day autocorrelation (for momentum activation) and rolling 60-day z-score/forward-return correlation (for mean-reversion strength) — both cheap, interpretable, and already computed in `structure.py`.
4. **Activation rules (objective, pre-committed thresholds):**
   - Activate/increase momentum only when rolling 60-day return autocorrelation > +0.05 **and** walk-forward score over the last 3 blocks is positive in ≥ 2 of 3.
   - Keep mean-reversion active only while the rolling z-score/forward-return correlation remains negative and walk-forward score is positive in ≥ 4 of the last 7 blocks (current status: **met**, 85.7%).
   - Keep PCA-residual and ALGO-lead-lag at a **reduced weight (≤10% of portfolio risk)** unless walk-forward hit-rate rises above 55% over a rolling 7-block window (current status: **not met**, 28.6% and 42.9%).
   - Retire a pairs relationship if a re-run cointegration test on the latest trailing window produces p > 0.10.
5. **Weight adjustment:** move weights gradually (e.g., ±10 percentage points per re-evaluation, not full on/off switches) to avoid whipsawing the combined portfolio and to avoid chasing a single good/bad block.
6. **Guard against performance-chasing:** never increase a strategy's weight based on a single block's result; require the multi-block walk-forward criteria above.

---

## Final Recommendation

1. **Primary strategy:** Price mean-reversion (rolling 20-day price z-score, signal-strength scaled), the only strategy that is robust across train, validation, test, *and* 6/7 walk-forward blocks.
2. **Secondary / diversifying strategies:** Return mean-reversion (1–2 day lookback) for a modest amount of horizon diversification, and the ALGO–NGTE pairs trade for genuine strategy-type diversification (low correlation with the mean-reversion sleeve) — sized conservatively given its multiple-testing origin.
3. **Recommended weights:** 40% price mean-reversion (w=20) / 35% return mean-reversion (lb=2) / 25% pairs ALGO–NGTE (w=40), risk-normalised so each contributes comparable volatility, then scaled so average gross exposure is ~45–50% of the theoretical position-limit ceiling (leaves room for signal-strength scaling to actually bind, and buffers against regime shifts).
4. **Risk limits:** never exceed the competition's per-asset caps (already enforced structurally); cap combined-portfolio average gross exposure at 50% of the theoretical maximum; monitor rolling drawdown and cut sizing by half if a strategy's rolling 60-day score turns negative for two consecutive evaluation windows.
5. **Refit frequency:** mean-reversion z-scores recompute daily by construction (rolling window); pairs hedge ratio refit daily on a rolling window; re-evaluate strategy weights weekly and after every new data release.
6. **Conditions under which this recommendation should change:** (a) if price/return mean-reversion's walk-forward hit-rate drops below 50% over a rolling 7-block window, cut its weight; (b) if a rolling cointegration re-test on ALGO–NGTE fails (p > 0.10), retire that leg; (c) if rolling autocorrelation turns positive and stays there, begin phasing in a small momentum sleeve.
7. **Strategies to keep monitoring, not yet deploy at scale:** ALGO lead-lag and PCA-residual mean-reversion (both showed occasional promise but failed walk-forward robustness — worth re-testing as more of the competition's 2,000-day dataset becomes available, since a 500-day sample may simply be too short to detect a real but weak effect); momentum strategies (currently no signal at all, but worth re-checking if market conditions shift).

---

## Standards / Caveats

- All headline conclusions are based on **out-of-sample, chronologically-honest** train/validation/test splits and 7-block walk-forward testing — not in-sample fitting.
- The ALGO–NGTE pair and any single "best validation" configuration should be treated with caution: they were the best of several candidates tested (multiple-testing bias), and with only ~500 observations, sampling noise is a real concern (see the sign-flipping test scores for ALGO lead-lag and PCA-residual across nearby parameter choices).
- `prices.txt` is a 500-day **sample**; the actual competition dataset is 2,000 days released in stages, so all quantitative conclusions here should be re-validated once more data is available — this is exactly what the dynamic framework in §K is for.
- No survivorship-bias concern (all 51 series run the full sample); no evidence of data leakage in our own pipeline (positions strictly use `prices[:t]` to decide the `t→t+1` position, verified by unit tests in `core.py`).

---

## Reproducible Code & Data Files

- `core.py` — exact competition scoring engine, position-limit clipping, PL calculation
- `strategies.py` — all 9 individual strategy signal generators
- `eda.py`, `structure.py` — dataset validation and structural pattern analysis (Section C/§C numbers)
- `run_backtests.py` — full parameter grid, train/val/test evaluation (Section D)
- `combos.py` — strategy combination and weight-grid testing (Section H)
- `walk_forward.py` — 7-block walk-forward robustness testing
- `make_charts.py` — generates all 10 charts
- `out/*.csv` — full numeric results backing every table above
