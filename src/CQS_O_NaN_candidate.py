"""Candidate submission: online factor-regime ensemble.

This is intentionally self-contained and uses NumPy only.  It is designed to
be copied to the required submission filename after the team has completed its
own validation.
"""

import numpy as np


# The values match the published limits and commission asymmetry.  We only use
# commission in the model-selection proxy; evaluation itself applies the true
# fees after this function returns the desired position.
_EPS = 1e-12
_VOL_WINDOW = 60
_PERFORMANCE_WINDOW = 30
_LOOKBACKS = tuple(range(1, 11))
# Do not trade when even the best expert has weak recent evidence.  This guards
# against forcing a forecast during a regime transition.
_MIN_INFORMATION_RATIO = 0.30


def _rolling_sum(values, window):
    """Trailing sums with shorter windows at the beginning of the history."""
    cumulative = np.cumsum(values)
    answer = cumulative.copy()
    if window < len(values):
        answer[window:] -= cumulative[:-window]
    return answer


def _cross_sectional_factor(returns):
    """Past-only, volatility-normalised estimate of the common market move."""
    n_days, n_inst = returns.shape
    factor = np.empty(n_days)
    for day in range(n_days):
        recent = returns[max(0, day - _VOL_WINDOW + 1) : day + 1]
        scale = np.std(recent, axis=0) + _EPS
        factor[day] = np.mean(returns[day] / scale)
    return factor


def _expert_directions(returns):
    """Return directional forecasts, one per row, for every observed day.

    Each value at index t is a forecast for the return following t.  The pool
    includes reversal and momentum at 1--10 day horizons for two independent
    factor estimates (the broad standardised factor and ALGO itself), plus
    permanent long/short fallbacks.  The selectors below decide which expert is
    currently credible; no outcome after t is used to form that day's forecast.
    """
    common_factor = _cross_sectional_factor(returns)
    factors = (common_factor, returns[:, 0])
    experts = []
    for factor in factors:
        for window in _LOOKBACKS:
            reversal = -np.sign(_rolling_sum(factor, window))
            experts.append(reversal)
            experts.append(-reversal)  # momentum counterpart
    experts.append(np.ones(len(returns)))
    experts.append(-np.ones(len(returns)))
    return np.asarray(experts)


def _choose_direction(returns, dollar_limits):
    """Choose the best recent expert by realised, risk-adjusted PnL.

    At the latest return index L-1, expert forecasts through L-2 already have
    known outcomes.  The latest forecast is therefore selected without any
    look-ahead.  We use a winner-take-all rule rather than averaging conflicting
    experts, which retains useful exposure when the current regime is clear.
    """
    n_days = len(returns)
    if n_days < _PERFORMANCE_WINDOW + 2:
        return 0.0

    experts = _expert_directions(returns)

    # A proxy for the PnL earned by holding every instrument at its dollar cap
    # over each next day.  It weights expert performance in the same direction
    # as the official objective, including ALGO's deliberately larger limit.
    full_long_pnl = np.sum(dollar_limits * np.expm1(returns), axis=1)

    # Forecast k was made after return k and earns the known return k+1.
    known_end = n_days - 1  # exclusive expert index; ends at n_days - 2
    known_start = max(0, known_end - _PERFORMANCE_WINDOW)
    pnl = experts[:, known_start:known_end] * full_long_pnl[known_start + 1 : known_end + 1]
    if pnl.shape[1] < 10:
        return 0.0
    risk_adjusted = np.mean(pnl, axis=1) / (np.std(pnl, axis=1) + _EPS)
    chosen = int(np.argmax(risk_adjusted))
    if risk_adjusted[chosen] < _MIN_INFORMATION_RATIO:
        return 0.0
    return float(experts[chosen, -1])


def getMyPosition(prcSoFar):
    """Return desired integer share holdings for the current close."""
    n_inst, n_times = prcSoFar.shape
    if n_inst != 51 or n_times < _PERFORMANCE_WINDOW + 3:
        return np.zeros(n_inst, dtype=int)

    returns = np.diff(np.log(prcSoFar), axis=1).T
    dollar_limits = np.full(n_inst, 10_000.0)
    dollar_limits[0] = 100_000.0
    direction = _choose_direction(returns, dollar_limits)

    target_shares = direction * dollar_limits / prcSoFar[:, -1]
    return target_shares.astype(int)
