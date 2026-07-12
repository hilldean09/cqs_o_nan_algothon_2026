""" 
thing to note: the reason why this algorithm, heavily edited version from my previous code,
doesnt yield such a high score is because it only trades with veryu high confidence.
i.e. only trades top 10, and only holds them if they are still in the top 16. this is a very conservative approach, and it is not necessarily the best approach for maximizing score
there definitely loys of ways to improve the score, but this version is a safe version that guarantees a positive :D
for days 251-500, this scored 153.67. haha 67
"""


import numpy as np

nInst = 51
dlrLimit = np.full(nInst, 10_000.0) # limits
dlrLimit[0] = 100_000.0
EPS = 1e-9 # prevent divide-by-zero in standardization

MIN_HIST = 60      # minimum days of return history before trading -> estimates r/s between all 51 instruments
ENTER_K = 10        # an asset must rank in the top ENTER_K |signal| to open a new position
EXIT_K = 16          # an already-held asset stays as long as it's still in the top EXIT_K
                     # (hysteresis - cuts needless flip-flopping / commission drag)

_heldSet = set()     # persists across calls: which assets currently carry conviction bets


def _leadlag_signal(rets_hist):
    """
    cross-sectional lead-lag forecast: regress each asset's return on *all*
    assets' previous-day standardized returns (via covariance-based
    projection), then apply that r/s to today's returns to forecast
    tomorrow's cross-section of returns. Captures the modest but genuine
    (statistically significant out-of-sample) lead-lag structure between
    instruments in this universe, as opposed to single-asset momentum/reversal
    which carries no real edge here.
    """
    X = rets_hist[:, :-1].T   # returns at t-1, shape (T-1, nInst) -> coontains the returns of all instruments for all days except the last day
    Y = rets_hist[:, 1:].T    # returns at t,   shape (T-1, nInst) -> 1 day after
    xmu, xsd = X.mean(0), X.std(0) + EPS # historical mean and stand dev of every X asset
    ymu, ysd = Y.mean(0), Y.std(0) + EPS # historical mean and stand dev of every Y asset
    Xs = (X - xmu) / xsd # standardize X and Y to have mean 0 and std 1
    Ys = (Y - ymu) / ysd # standardize X and Y to have mean 0 and std 1
    LL = (Xs.T @ Ys) / Xs.shape[0]     # (nInst, nInst) lead-lag coefficient matrix
    np.fill_diagonal(LL, 0.0)          # use cross-asset relationships
    last = rets_hist[:, -1] # most recent return for every instrument
    last_std = (last - xmu) / xsd # standardize the most recent return for every instrument
    return last_std @ LL               # forecast for tomorrow's standardized return, per asset


def _regime_scale(rets_hist):
    """shrink exposure modestly when the market is in an abnormally turbulent
    regime (short-term vol well above its longer-run level); this doesn't
    change direction, only overall aggressiveness."""
    mkt = rets_hist.mean(axis=0)
    if mkt.shape[0] < 60: # not enough history to estimate short/long vol, so don't scale down
        return 1.0
    short_vol = mkt[-10:].std() + EPS # short-term volatility of the market
    long_vol = mkt[-60:].std() + EPS # long term
    ratio = short_vol / long_vol # ratio of short-term to long-term volatility -> if ratio > 1, then short-term volatility is higher than long-term volatility, indicating a turbulent regime
    return float(np.clip(1.15 - 0.35 * max(ratio - 1.0, 0.0), 0.6, 1.15))


""" 
getmyposition() first checks data avail, verifies that at least 60 days of historical returns are avail,
then calculate its returns, converts it into logarithmic prices, which are used as the input for forecasting model
( logarithmic returns as my friend claude and chatgpt says thisb is the standard practice in quant firms :D )
it then calls the trading signals, ranks intruments, and selects which to hold. (top 10), after that it assigns 
position direction, and converts desired dollar exposure to required numbner of shares
"""
def getMyPosition(prcSoFar):
    global _heldSet
    nins, nt = prcSoFar.shape # number of instruments and time steps

    if nt < MIN_HIST + 1: # not enough history to estimate r/s between all instruments, so don't trade yet
        return np.zeros(nins, dtype=int)

    logp = np.log(prcSoFar) # compute log prices from prices
    rets = np.diff(logp, axis=1) # compute returns from log prices

    sig = _leadlag_signal(rets)  # compute lead-lag signals
    scale = _regime_scale(rets) # compute regime scaling factor

    order = np.argsort(-np.abs(sig)) # sort the signals in descending order of absolute value, so that the most extreme signals are first
    ranks = np.empty(nins, dtype=int)
    ranks[order] = np.arange(nins)

    newHeld = set()
    for i in range(nins): 
        threshold = EXIT_K if i in _heldSet else ENTER_K
        if ranks[i] < threshold:
            newHeld.add(i)
    _heldSet = newHeld

    conf = np.zeros(nins)
    for i in newHeld:
        conf[i] = np.sign(sig[i]) * scale

    curPrices = prcSoFar[:, -1]
    targetDollars = conf * dlrLimit
    targetShares = targetDollars / curPrices

    posLimitShares = (dlrLimit / curPrices).astype(int)
    newPos = np.clip(targetShares, -posLimitShares, posLimitShares)
    return newPos.astype(int)


def setGlobalVariable( name, value ):
    globals()[ name ] = value

