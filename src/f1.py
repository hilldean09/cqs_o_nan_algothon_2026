"""
=============================================================================
Algothon 2026 -- CQS_O_NaN
Ensemble cross-asset lead-lag, blended with two orthogonal reversal legs.
=============================================================================

WHAT THE DATA ACTUALLY CONTAINS  (measured, not assumed)
--------------------------------------------------------
 * Asset 0 (ALGO) is the market index: its returns correlate 0.977 with the
   equal-weight mean of assets 1-50, and PC1 of the return covariance
   correlates 0.959 with it. It has the lowest volatility of the 51.
 * There are ~3 genuine common factors. The Marchenko-Pastur upper edge for
   a 51 x 999 random matrix is 1.50; the observed eigenvalues are
   11.9, 2.76, 1.94, then 1.21, 1.18, ... -- so exactly three sit above noise.
 * The dominant edge is a ONE-DAY cross-asset lead-lag. 3.2% of the entries of
   the lag-1 cross-correlation matrix exceed 3 sigma (0.3% expected by chance).
   The relation is directional, not symmetric: e.g. DUCT -> AMRP carries
   +0.171 (t=5.4) while AMRP -> DUCT is -0.017. A handful of assets are
   persistent "leaders" (DUCT, MTNS, ALGO) and a handful are persistent
   "receivers" (GARI, AMRP, CUBO).
 * That lead-lag matrix is LOW RANK -- its top singular value alone carries
   51% of its squared mass -- which is why rank truncation denoises it so well.
 * The edge is purely one-day. Walk-forward IC by horizon:
       +1d: +0.086     +2d: +0.004     +3d: -0.004     +5d: -0.015
   So there is no way to hold positions longer to save commission.

SIGNAL CONSTRUCTION -- three near-orthogonal legs
--------------------------------------------------------
 (1) LEAD-LAG, weight 0.65.  Ridge regression of every asset's return on the
     previous day's standardised returns of all 51 assets, then rank-truncated
     via SVD. Estimated on the FULL EXPANDING HISTORY.
     Walk-forward IC 0.086, t = 13.6.

     Two choices here matter and both were measured, not guessed:
       - Expanding window beats every rolling window and every exponential
         decay (IC 0.073 expanding vs 0.068 at 600d, 0.051 at 250d, 0.042 at
         120d). The lead-lag structure is STATIONARY, so throwing away old
         data only adds estimation error. This is also what makes the model
         "learn on the go": every new day is folded into the estimate, and
         accuracy genuinely improves with more data -- walk-forward IC by
         thirds of the sample runs 0.066 -> 0.091 -> 0.100.
       - Rank truncation is the single biggest signal improvement found
         (IC 0.073 -> 0.086). The optimum is a broad plateau over ranks 4-8
         and lambda 300-3000, so rather than pick a point on that plateau the
         model AVERAGES over all 12 combinations. Ensembling both removed a
         fragile hyper-parameter choice and slightly beat every individual
         member of the ensemble.

 (2) SHORT-HORIZON RESIDUAL REVERSAL, weight 0.25.  Strip the top 3 principal
     components, then fade the last 8 days of residual return.
     IC 0.024, t = 4.6.  Correlation with leg (1): +0.001.

 (3) LONG-HORIZON REVERSAL, weight 0.10.  Fade 120-day normalised performance.
     IC 0.015, t = 2.8, same sign in all three thirds of the sample.
     Correlation with leg (1): +0.003.

     Legs 2 and 3 are weak alone, but being uncorrelated with the dominant leg
     they add IC in quadrature: 0.086 -> 0.090 blended.

POSITION SIZING -- why there is no ranking or top-K rule
--------------------------------------------------------
 The scoring function is  score = mean(PL) * SR^2 / (SR^2 + 1).
 That Sharpe factor is 0.96 at SR=5 and 0.988 at SR=9 -- it saturates almost
 completely above SR ~ 4. So score is, to first order, JUST THE MEAN DAILY
 PnL, and mean PnL scales linearly with deployed capital. Any rule that sits
 out of an asset (a top-K filter, a conviction rank cut, a per-asset
 reliability gate) buys a Sharpe improvement worth ~2% of score while
 surrendering mean PnL worth far more. All three were tested and all three
 lowered the score.

 So every asset is sized continuously by a saturating function of its own
 blended signal -- no ranking, no hysteresis, no top-K:

     w_i = clip( signal_i / KAPPA , -1, +1 )

 with KAPPA small enough that roughly 93% of assets sit at their cap, giving
 ~97% utilisation of the $600k of available limit. Conviction still governs
 the unsaturated tail and, more importantly, the sign of every bet.

 Asset 0 is traded on the same footing. It carries no standalone edge (its own
 walk-forward IC is -0.006, and it shows no autocorrelation, momentum or
 drift), but its position expresses the market-timing content of the lead-lag
 book -- ALGO leads the universe, so when the book tilts net long it is
 usually right to be long the index too. Setting it to zero, or using it as a
 beta hedge against the other 50, both LOWERED the score: hedging destroys
 that genuine timing alpha along with the unwanted beta.

WHAT WAS TESTED AND REJECTED  (all walk-forward, none of it is in the file)
--------------------------------------------------------
   raw single-asset momentum/reversal .... |IC| < 0.013, unstable
   multi-lag lead-lag (lags 2,3,5) ....... IC 0.086 -> 0.068, pure noise
   |return| and squared-return features .. IC 0.086 -> 0.080
   cross-sectional rank transform ........ IC 0.086 -> 0.073
   EWMA-vol standardised inputs .......... no change
   volatility-regime conditioned training  no change
   per-target R^2 shrinkage .............. no change
   proper reduced-rank regression ........ IC 0.086 -> 0.077
   winsorising returns at 3/4 sigma ...... no change
   per-asset reliability gating .......... score 868 -> 662
   beta-hedging the book with ALGO ....... score 868 -> 795
   in-book beta neutralisation ........... score 868 -> 712
   no-trade band (5%-35% of limit) ....... no gain; signal is 1-day so
                                           turnover is irreducible
=============================================================================
"""

import numpy as np

EPS = 1e-12
nInst = 51

# ---- dollar position limits (asset 0 has the 10x cap and 5x lower commission)
DLR = np.full(nInst, 10_000.0)
DLR[0] = 100_000.0

# ---- lead-lag ensemble: average over a measured plateau rather than a point
LAMBDAS = (300.0, 1000.0, 3000.0)
RANKS = (4, 5, 6, 7)

# ---- signal blend weights (leg1 = 1 - W_SREV - W_LREV)
W_SREV = 0.25          # short-horizon residual reversal
W_LREV = 0.10          # long-horizon reversal

SREV_L = 8             # short reversal lookback (days)
SREV_K = 3             # principal components stripped (= number of real factors)
SREV_W = 250           # window for the PCA covariance
LREV_L = 120           # long reversal lookback (days)

KAPPA = 0.06           # saturation scale -> ~93% of assets at their cap
MIN_HIST = 120         # below this the lead-lag matrix is too noisy to trade
FULL_HIST = 250        # size scales in linearly between MIN_HIST and FULL_HIST
MAX_SHARES = 1e12      # sanity ceiling so the int cast is always well-defined


def _zc(v):
    """Cross-sectional standardisation, safe against a degenerate cross-section."""
    s = v.std()
    if not np.isfinite(s) or s < EPS:
        return np.zeros_like(v)
    return (v - v.mean()) / s


def _leadlag(rets):
    """Rank-truncated ridge lead-lag forecast, averaged over the (lambda, rank)
    plateau. Fitted on the entire history available so far."""
    n = rets.shape[0]
    X = rets[:, :-1].T
    Y = rets[:, 1:].T
    xmu = X.mean(0)
    xsd = X.std(0) + EPS
    ymu = Y.mean(0)
    ysd = Y.std(0) + EPS
    Xs = (X - xmu) / xsd
    Ys = (Y - ymu) / ysd

    G = Xs.T @ Xs
    C = Xs.T @ Ys
    I = np.eye(n)
    cur = (rets[:, -1] - xmu) / xsd

    acc = np.zeros(n)
    cnt = 0
    for lam in LAMBDAS:
        try:
            B = np.linalg.solve(G + lam * I, C)
        except np.linalg.LinAlgError:
            continue
        try:
            u, s, vt = np.linalg.svd(B, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        for k in RANKS:
            if k >= len(s):
                continue
            s2 = s.copy()
            s2[k:] = 0.0
            f = cur @ ((u * s2) @ vt)
            sd = f.std()
            if np.isfinite(sd) and sd > EPS:
                acc += f / sd
                cnt += 1
    if cnt == 0:
        return np.zeros(n)
    return acc / cnt


def _short_reversal(rets):
    """Fade recent residual moves after stripping the 3 common factors."""
    h = rets[:, -SREV_W:] if rets.shape[1] > SREV_W else rets
    sd = h.std(1, keepdims=True) + EPS
    z = (h - h.mean(1, keepdims=True)) / sd
    try:
        C = np.cov(z)
        w, v = np.linalg.eigh(C)
        v = v[:, ::-1][:, :SREV_K]
        res = z - v @ (v.T @ z)
    except np.linalg.LinAlgError:
        res = z
    L = min(SREV_L, res.shape[1])
    return -res[:, -L:].sum(1) / np.sqrt(L)


def _long_reversal(rets):
    """Fade long-horizon normalised performance."""
    h = rets[:, -(LREV_L + 30):] if rets.shape[1] > LREV_L + 30 else rets
    x = h / (h.std(1, keepdims=True) + EPS)
    L = min(LREV_L, x.shape[1])
    return -x[:, -L:].sum(1) / np.sqrt(L)


def getMyPosition(prcSoFar):
    prcSoFar = np.asarray(prcSoFar, dtype=float)
    nins, nt = prcSoFar.shape

    # ---- guards: not enough history, or unusable prices
    if nt < MIN_HIST + 2:
        return np.zeros(nins, dtype=int)
    if not np.all(np.isfinite(prcSoFar)) or np.any(prcSoFar <= 0.0):
        prc = np.where(np.isfinite(prcSoFar) & (prcSoFar > 0.0), prcSoFar, np.nan)
        prc = np.asarray(
            [np.interp(np.arange(nt), np.flatnonzero(np.isfinite(row)),
                       row[np.isfinite(row)]) if np.isfinite(row).any()
             else np.ones(nt) for row in prc])
    else:
        prc = prcSoFar

    rets = np.diff(np.log(prc), axis=1)
    if not np.all(np.isfinite(rets)):
        rets = np.nan_to_num(rets, nan=0.0, posinf=0.0, neginf=0.0)

    # ---- blended signal: three near-orthogonal legs
    sig = (1.0 - W_SREV - W_LREV) * _zc(_leadlag(rets))
    if rets.shape[1] >= SREV_L + 5:
        sig = sig + W_SREV * _zc(_short_reversal(rets))
    if rets.shape[1] >= 40:
        sig = sig + W_LREV * _zc(_long_reversal(rets))
    if not np.all(np.isfinite(sig)):
        sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)

    # ---- continuous saturating sizing: no ranking, no top-K, no hysteresis
    w = np.clip(sig / KAPPA, -1.0, 1.0)

    # ---- scale in while the lead-lag estimate is still short of data
    if rets.shape[1] < FULL_HIST:
        w = w * np.clip((rets.shape[1] - MIN_HIST) / float(FULL_HIST - MIN_HIST), 0.0, 1.0)

    # ---- dollars -> shares, hardened against degenerate prices.
    # A price that is zero, NaN or denormally small would overflow the division
    # and make the int cast undefined, so the divisor is floored and the result
    # is sanitised before the cast. The share ceiling is recomputed from TODAY's
    # price on every call, so a position can never drift outside its cap when
    # prices move (the brief's ALGO example) -- this function is stateless and
    # never assumes yesterday's position survived.
    cur = np.maximum(np.nan_to_num(prc[:, -1], nan=0.0, posinf=0.0, neginf=0.0), 1e-8)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        lim = np.floor(np.clip(DLR[:nins] / cur, 0.0, MAX_SHARES))
        pos = np.clip(w * DLR[:nins] / cur, -lim, lim)
    lim = np.nan_to_num(lim, nan=0.0, posinf=MAX_SHARES, neginf=0.0)
    pos = np.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(pos, -lim, lim).astype(np.int64)


