import numpy as np

LOOKBACKS = (5, 20)
DEFAULT_DOLLAR_LIMIT = 10_000
ALGO_DOLLAR_LIMIT = 100_000
VOLATILITY_FLOOR = 1e-12
REBALANCE_BAND = 0.515
REGIME_VOL_SHORT = 10
REGIME_VOL_LONG = 60
REGIME_THRESHOLD = 1.15
REGIME_SCALE = 1.0
MPAIRS_LOOKBACK = 40
MPAIRS_WEIGHT = 0.0
MPAIRS_TOP_K = 3
MPAIRS_ENTRY_Z = 1.5
MPAIRS_MIN_CORR = 0.65
OLS_LOOKBACK = 40
OLS_WEIGHT = 0.0
OLS_ENTRY_Z = 2.0
LEADLAG_LOOKBACK = 330
LEADLAG_MIN_OBS = 80
LEADLAG_WEIGHT = 1.0
LEADLAG_TOP_K = 2
LEADLAG_ENTRY_Z = 0.42
LEADLAG_MIN_CORR = 0.0
LEADLAG_PREFIX = True
LEADLAG_SHRINK = 0.022
LEADLAG_POWER = 1.30
EDGEPAIRS_SELECT_LOOKBACK = 330
EDGEPAIRS_FIT_LOOKBACK = 30
EDGEPAIRS_WEIGHT = 1.0
EDGEPAIRS_TOP_K = 5
EDGEPAIRS_ENTRY_Z = 1.0
EDGEPAIRS_MIN_CORR = 0.25
EDGEPAIRS_PREFIX = True
EDGEPAIRS_INCLUDE_ALGO = False
EDGEPAIRS_TREND = False
EDGEPAIRS_BETA_RIDGE = 1e-6
FACTOR_LOOKBACK = 20
FACTOR_WEIGHT = 0.08
FACTOR_ENTRY_Z = 1.30
FACTOR_INCLUDE_ALGO = False
FACTOR_BETA_RIDGE = 1e-6
ADAPTIVE_BAND_VOL_LB = 10
ADAPTIVE_BAND_SCALE = 2.5
VOL_TARGET_LOOKBACK = 20
VOL_TARGET_FLOOR = 2.75
VOL_TARGET_CAP = 3.0

_current_positions = np.zeros(0, dtype=int)
_last_history = np.zeros((0, 0))


def _rolling_ols_beta(y, x, lb):
    yy = np.asarray(y[-lb:], dtype=float)
    xx = np.asarray(x[-lb:], dtype=float)
    denom = float(xx @ xx)
    if denom < VOLATILITY_FLOOR:
        return 0.0
    return float(xx @ yy) / denom


def _spread_z(spread, lb):
    window = np.asarray(spread[-lb:], dtype=float)
    mu = float(window.mean())
    sd = float(window.std()) + VOLATILITY_FLOOR
    return (float(window[-1]) - mu) / sd


def _mpairs_signal(log_prices, daily_returns):
    nins = log_prices.shape[0]
    sig = np.zeros(nins)
    rets = daily_returns[1:, -MPAIRS_LOOKBACK:]
    n = rets.shape[0]
    if n < 2 or MPAIRS_LOOKBACK < 3:
        return sig

    # Pairwise abs corr using vectorized corrcoef for speed
    stds = np.std(rets, axis=1)
    valid = stds >= VOLATILITY_FLOOR
    corr_matrix = np.corrcoef(rets)
    candidates = []
    for i in range(n):
        if not valid[i]:
            continue
        for j in range(i + 1, n):
            if not valid[j]:
                continue
            corr = corr_matrix[i, j]
            if np.isnan(corr):
                continue
            if abs(corr) >= MPAIRS_MIN_CORR:
                candidates.append((abs(corr), i + 1, j + 1))
    candidates.sort(reverse=True)

    active = 0
    for _, i, j in candidates[:MPAIRS_TOP_K]:
        beta = _rolling_ols_beta(
            log_prices[i, :],
            log_prices[j, :],
            MPAIRS_LOOKBACK,
        )
        spread = log_prices[i, :] - beta * log_prices[j, :]
        z = _spread_z(spread, MPAIRS_LOOKBACK)
        if abs(z) < MPAIRS_ENTRY_Z:
            continue
        zc = float(np.clip(z, -1.0, 1.0))
        sig[i] += -zc
        sig[j] += zc * beta
        active += 1

    if active == 0:
        return np.zeros(nins)
    sig /= active
    max_abs = np.max(np.abs(sig)) + VOLATILITY_FLOOR
    return np.clip(sig / max_abs, -1.0, 1.0)


def _leadlag_signal(daily_returns):
    nins, nret = daily_returns.shape
    if nret < LEADLAG_MIN_OBS + 1:
        return np.zeros(nins)

    if LEADLAG_LOOKBACK > 0:
        lb = min(LEADLAG_LOOKBACK, nret - 1)
    else:
        lb = nret - 1
    if lb < LEADLAG_MIN_OBS:
        return np.zeros(nins)

    if LEADLAG_PREFIX:
        leader = daily_returns[:, :lb]
        follower = daily_returns[:, 1 : lb + 1]
    else:
        leader = daily_returns[:, -(lb + 1) : -1]
        follower = daily_returns[:, -lb:]

    leader_mu = leader.mean(axis=1, keepdims=True)
    follower_mu = follower.mean(axis=1, keepdims=True)
    leader_sd = leader.std(axis=1, keepdims=True)
    follower_sd = follower.std(axis=1, keepdims=True)
    valid_leader = leader_sd[:, 0] >= VOLATILITY_FLOOR
    valid_follower = follower_sd[:, 0] >= VOLATILITY_FLOOR
    if not np.any(valid_leader) or not np.any(valid_follower):
        return np.zeros(nins)

    leader_z = (leader - leader_mu) / np.maximum(leader_sd, VOLATILITY_FLOOR)
    follower_z = (follower - follower_mu) / np.maximum(follower_sd, VOLATILITY_FLOOR)
    corr = np.dot(leader_z, follower_z.T) / lb
    corr[~valid_leader, :] = 0.0
    corr[:, ~valid_follower] = 0.0
    np.fill_diagonal(corr, 0.0)

    latest = daily_returns[:, -1]
    latest_z = (latest - leader_mu[:, 0]) / np.maximum(leader_sd[:, 0], VOLATILITY_FLOOR)
    if LEADLAG_ENTRY_Z > 0:
        latest_z = np.where(np.abs(latest_z) >= LEADLAG_ENTRY_Z, latest_z, 0.0)
    latest_z[~valid_leader] = 0.0

    weights = corr.copy()
    if LEADLAG_SHRINK > 0:
        weights = np.sign(weights) * np.maximum(np.abs(weights) - LEADLAG_SHRINK, 0.0)
    if LEADLAG_POWER != 1.0:
        weights = np.sign(weights) * (np.abs(weights) ** LEADLAG_POWER)
    abs_weights = np.abs(weights)
    if LEADLAG_TOP_K > 0 and LEADLAG_TOP_K < nins:
        keep = np.zeros_like(weights, dtype=bool)
        idx = np.argpartition(abs_weights, -LEADLAG_TOP_K, axis=0)[
            -LEADLAG_TOP_K :,
            :,
        ]
        keep[idx, np.arange(nins)[None, :]] = True
        weights[~keep] = 0.0
    weights[abs_weights < LEADLAG_MIN_CORR] = 0.0

    denom = np.sum(np.abs(weights), axis=0)
    sig = np.zeros(nins)
    active = denom > VOLATILITY_FLOOR
    if np.any(active):
        sig[active] = np.dot(latest_z, weights[:, active]) / denom[active]

    return np.clip(sig, -1.0, 1.0)


def _edgepairs_signal(log_prices, daily_returns):
    nins, nret = daily_returns.shape
    select_lb = min(EDGEPAIRS_SELECT_LOOKBACK, nret)
    fit_lb = min(EDGEPAIRS_FIT_LOOKBACK, log_prices.shape[1])
    if select_lb < 3 or fit_lb < 5:
        return np.zeros(nins)

    if EDGEPAIRS_PREFIX:
        select_rets = daily_returns[:, :select_lb]
    else:
        select_rets = daily_returns[:, -select_lb:]
    if EDGEPAIRS_INCLUDE_ALGO:
        inst_offset = 0
    else:
        inst_offset = 1
        select_rets = select_rets[1:, :]

    n = select_rets.shape[0]
    if n < 2:
        return np.zeros(nins)

    stds = select_rets.std(axis=1)
    valid = stds >= VOLATILITY_FLOOR
    valid_idx = np.flatnonzero(valid)
    if valid_idx.size < 2:
        return np.zeros(nins)
    corr_matrix = np.full((n, n), np.nan)
    corr_matrix[np.ix_(valid_idx, valid_idx)] = np.corrcoef(select_rets[valid_idx, :])
    candidates = []
    for a in range(n):
        if not valid[a]:
            continue
        for b in range(a + 1, n):
            if not valid[b]:
                continue
            corr = corr_matrix[a, b]
            if np.isnan(corr) or corr < EDGEPAIRS_MIN_CORR:
                continue
            candidates.append((corr, a + inst_offset, b + inst_offset))
    if not candidates:
        return np.zeros(nins)
    candidates.sort(reverse=True)

    sig = np.zeros(nins)
    active = 0
    yx = log_prices[:, -fit_lb:]
    for _, i, j in candidates[:EDGEPAIRS_TOP_K]:
        y = yx[i, :]
        x = yx[j, :]
        x_mu = float(x.mean())
        y_mu = float(y.mean())
        xc = x - x_mu
        yc = y - y_mu
        beta = float(xc @ yc) / (float(xc @ xc) + EDGEPAIRS_BETA_RIDGE)
        if beta <= 0 or not np.isfinite(beta):
            continue
        beta = float(np.clip(beta, 0.2, 5.0))
        alpha = y_mu - beta * x_mu
        spread = y - (alpha + beta * x)
        z = _spread_z(spread, fit_lb)
        if abs(z) < EDGEPAIRS_ENTRY_Z:
            continue
        zc = float(np.clip(z, -1.0, 1.0))
        direction = 1.0 if EDGEPAIRS_TREND else -1.0
        sig[i] += direction * zc
        sig[j] += -direction * zc * beta
        active += 1

    if active == 0:
        return np.zeros(nins)
    sig /= active
    max_abs = np.max(np.abs(sig)) + VOLATILITY_FLOOR
    return np.clip(sig / max_abs, -1.0, 1.0)


def _factor_residual_signal(log_prices):
    """ETF-style residual: each instrument versus a rolling synthetic basket."""
    nins, nt = log_prices.shape
    lb = min(FACTOR_LOOKBACK, nt)
    if lb < 5:
        return np.zeros(nins)

    sig = np.zeros(nins)
    window = log_prices[:, -lb:]
    tradable = range(nins) if FACTOR_INCLUDE_ALGO else range(1, nins)
    market = np.mean(window[1:, :], axis=0)
    m_mu = float(market.mean())
    mc = market - m_mu
    denom = float(mc @ mc) + FACTOR_BETA_RIDGE

    for i in tradable:
        y = window[i, :]
        y_mu = float(y.mean())
        beta = float(mc @ (y - y_mu)) / denom
        alpha = y_mu - beta * m_mu
        spread = y - (alpha + beta * market)
        z = _spread_z(spread, lb)
        if abs(z) < FACTOR_ENTRY_Z:
            continue
        sig[i] = -float(np.clip(z, -1.0, 1.0))

    return sig


def getMyPosition(prcSoFar):
    global _current_positions, _last_history

    nins, nt = prcSoFar.shape
    longest_lookback = max(max(LOOKBACKS), REGIME_VOL_LONG, MPAIRS_LOOKBACK, OLS_LOOKBACK)
    if ADAPTIVE_BAND_VOL_LB > 0:
        longest_lookback = max(longest_lookback, ADAPTIVE_BAND_VOL_LB)
    if VOL_TARGET_LOOKBACK > 0:
        longest_lookback = max(longest_lookback, VOL_TARGET_LOOKBACK)
    if LEADLAG_WEIGHT > 0:
        lag_need = LEADLAG_MIN_OBS + 1
        if LEADLAG_LOOKBACK > 0:
            lag_need = max(lag_need, LEADLAG_LOOKBACK + 1)
        longest_lookback = max(longest_lookback, lag_need)
    if EDGEPAIRS_WEIGHT > 0 and EDGEPAIRS_SELECT_LOOKBACK > 0:
        longest_lookback = max(
            longest_lookback,
            EDGEPAIRS_SELECT_LOOKBACK + 1,
            EDGEPAIRS_FIT_LOOKBACK,
        )
    if FACTOR_WEIGHT > 0 and FACTOR_LOOKBACK > 0:
        longest_lookback = max(longest_lookback, FACTOR_LOOKBACK)

    same_history = _last_history.shape == prcSoFar.shape and np.array_equal(
        _last_history,
        prcSoFar,
    )
    if same_history:
        return _current_positions.copy()

    is_continuation = (
        _current_positions.size == nins
        and _last_history.shape == (nins, nt - 1)
        and np.array_equal(_last_history, prcSoFar[:, :-1])
    )
    if not is_continuation:
        _current_positions = np.zeros(nins, dtype=int)

    if nt <= longest_lookback:
        _last_history = prcSoFar.copy()
        return _current_positions.copy()

    if (
        (LEADLAG_WEIGHT > 0 and LEADLAG_PREFIX)
        or (EDGEPAIRS_WEIGHT > 0 and EDGEPAIRS_PREFIX)
    ):
        recent_prices = prcSoFar
    else:
        recent_prices = prcSoFar[:, -(longest_lookback + 1) :]
    log_prices = np.log(recent_prices)
    daily_returns = np.diff(log_prices, axis=1)

    leadlag_replaces = LEADLAG_WEIGHT >= 1.0
    signal = np.zeros(nins)
    if not leadlag_replaces:
        for lookback in LOOKBACKS:
            cumulative_return = log_prices[:, -1] - log_prices[:, -(lookback + 1)]
            volatility = daily_returns[:, -lookback:].std(axis=1) * np.sqrt(lookback)
            standardized_return = cumulative_return / np.maximum(
                volatility, VOLATILITY_FLOOR
            )
            signal -= np.clip(standardized_return, -1.0, 1.0) / len(LOOKBACKS)

        if OLS_WEIGHT > 0 and nt > OLS_LOOKBACK:
            basket = np.nanmean(log_prices[1:, :], axis=0)
            algo = log_prices[0, :]
            beta = _rolling_ols_beta(algo, basket, OLS_LOOKBACK)
            spread = algo - beta * basket
            z = _spread_z(spread, OLS_LOOKBACK)
            if abs(z) >= OLS_ENTRY_Z:
                ols_sig = np.zeros(nins)
                ols_sig[0] = -np.clip(z, -1.0, 1.0)
                ols_sig[1:] = -ols_sig[0] / (nins - 1)
                signal = (1.0 - OLS_WEIGHT) * signal + OLS_WEIGHT * ols_sig

        if MPAIRS_WEIGHT > 0:
            mpairs_signal = _mpairs_signal(log_prices, daily_returns)
            if np.any(np.abs(mpairs_signal) > VOLATILITY_FLOOR):
                signal = (1.0 - MPAIRS_WEIGHT) * signal + MPAIRS_WEIGHT * mpairs_signal

    if LEADLAG_WEIGHT > 0:
        leadlag_signal = _leadlag_signal(daily_returns)
        if np.any(np.abs(leadlag_signal) > VOLATILITY_FLOOR):
            signal = (1.0 - LEADLAG_WEIGHT) * signal + LEADLAG_WEIGHT * leadlag_signal

    if EDGEPAIRS_WEIGHT > 0 and EDGEPAIRS_SELECT_LOOKBACK > 0:
        edgepairs_signal = _edgepairs_signal(log_prices, daily_returns)
        if np.any(np.abs(edgepairs_signal) > VOLATILITY_FLOOR):
            signal = np.clip(signal + EDGEPAIRS_WEIGHT * edgepairs_signal, -1.0, 1.0)

    if FACTOR_WEIGHT > 0 and FACTOR_LOOKBACK > 0:
        factor_signal = _factor_residual_signal(log_prices)
        if np.any(np.abs(factor_signal) > VOLATILITY_FLOOR):
            signal = np.clip(signal + FACTOR_WEIGHT * factor_signal, -1.0, 1.0)

    short_vol = daily_returns[:, -REGIME_VOL_SHORT:].std(axis=1)
    long_vol = daily_returns[:, -REGIME_VOL_LONG:].std(axis=1)
    vol_ratio = short_vol / np.maximum(long_vol, VOLATILITY_FLOOR)
    exposure = np.where(vol_ratio >= REGIME_THRESHOLD, REGIME_SCALE, 1.0)

    dollar_limits = np.full(nins, DEFAULT_DOLLAR_LIMIT, dtype=float)
    dollar_limits[0] = ALGO_DOLLAR_LIMIT
    # Vol-targeted position sizing: scale dollar limit by inverse rolling vol per instrument
    if VOL_TARGET_LOOKBACK > 0 and daily_returns.shape[1] >= VOL_TARGET_LOOKBACK:
        inst_vol = daily_returns[:, -VOL_TARGET_LOOKBACK:].std(axis=1)
        median_vol = float(np.median(inst_vol))
        vt_scale = median_vol / np.maximum(inst_vol, VOLATILITY_FLOOR)
        vt_scale = np.clip(vt_scale, VOL_TARGET_FLOOR, VOL_TARGET_CAP)
        dollar_limits *= vt_scale
    target_dollars = dollar_limits * np.clip(signal, -1.0, 1.0) * exposure
    current_prices = prcSoFar[:, -1]
    desired_positions = np.trunc(target_dollars / current_prices).astype(int)

    rebalance_notional = np.abs(desired_positions - _current_positions) * current_prices
    band = REBALANCE_BAND
    if ADAPTIVE_BAND_VOL_LB > 0 and daily_returns.shape[1] >= 2 * ADAPTIVE_BAND_VOL_LB:
        ab_short = daily_returns[:, -ADAPTIVE_BAND_VOL_LB:].std(axis=1)
        ab_long = daily_returns[:, -2 * ADAPTIVE_BAND_VOL_LB:].std(axis=1)
        ab_ratio = ab_short / np.maximum(ab_long, VOLATILITY_FLOOR)
        if float(np.median(ab_ratio)) >= 1.0:
            band *= ADAPTIVE_BAND_SCALE
    should_rebalance = rebalance_notional >= band * dollar_limits
    new_positions = np.where(
        should_rebalance,
        desired_positions,
        _current_positions,
    )

    # The evaluator clips submitted positions at these fixed limits. Keep the
    # local band state on the same bounds even when vol targeting scales a
    # signal's provisional allocation above them.
    position_limits = np.full(nins, DEFAULT_DOLLAR_LIMIT, dtype=float)
    position_limits[0] = ALGO_DOLLAR_LIMIT
    max_shares = np.trunc(position_limits / current_prices).astype(int)
    _current_positions = np.clip(new_positions, -max_shares, max_shares)
    _last_history = prcSoFar.copy()
    return _current_positions.copy()
