"""Walk-forward research harness for the Algothon price data.

This file deliberately reimplements the scoring mechanics relevant to a
strategy: a position chosen at close d earns the price change from d to d+1,
and the fee for a rebalance is charged on the following mark.  Signals only
receive prices through the current close, so it is safe to use this for
walk-forward analysis.

It is an analysis tool, not a submission file.  Run from ``src`` with
``python research.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


EPS = 1e-12


def load_prices(filename: str = "prices.txt") -> np.ndarray:
    return pd.read_csv(filename, sep=r"\s+").to_numpy(dtype=float).T


def score(pnl: np.ndarray) -> tuple[float, float, float]:
    """Return official score, mean daily PnL, and annualised Sharpe."""
    mu = float(np.mean(pnl))
    sigma = float(np.std(pnl))
    sharpe = np.sqrt(250.0) * mu / sigma if sigma > EPS else 0.0
    official = mu * sharpe**2 / (sharpe**2 + 1.0) if mu > 0.0 and sigma > EPS else mu
    return official, mu, sharpe


@dataclass(frozen=True)
class BacktestResult:
    official_score: float
    mean_pnl: float
    annual_sharpe: float
    mean_turnover: float
    pnl: np.ndarray


Signal = Callable[[np.ndarray], np.ndarray]


def backtest(prices: np.ndarray, signal: Signal, start: int, end: int) -> BacktestResult:
    """Backtest on price marks ``start`` through ``end - 1``.

    The first position is chosen after observing close ``start - 1``.  No
    information at or after the following marked day reaches the signal.
    """
    n_inst = prices.shape[0]
    limits = np.full(n_inst, 10_000.0)
    limits[0] = 100_000.0
    commission = np.full(n_inst, 0.0001)
    commission[0] = 0.00002

    def target_position(history: np.ndarray) -> np.ndarray:
        raw = np.asarray(signal(history), dtype=float)
        if raw.shape != (n_inst,):
            raise ValueError(f"signal has shape {raw.shape}, expected {(n_inst,)}")
        # A signal represents the fraction of the allowed dollar exposure.
        fractions = np.nan_to_num(np.clip(raw, -1.0, 1.0))
        return (fractions * limits / history[:, -1]).astype(int)

    position = target_position(prices[:, :start])
    opening_price = prices[:, start - 1]
    fee = np.sum(np.abs(position) * opening_price * commission)
    turnover = [float(np.sum(np.abs(position) * opening_price))]
    daily_pnl: list[float] = []

    for day in range(start, end):
        # This is the exact one-day mark-to-market and delayed fee convention
        # used by eval.py.  The final iteration marks but does not rebalance.
        daily_pnl.append(float(position @ (prices[:, day] - prices[:, day - 1]) - fee))
        if day == end - 1:
            continue
        new_position = target_position(prices[:, : day + 1])
        trade_dollars = np.abs(new_position - position) * prices[:, day]
        fee = float(np.sum(trade_dollars * commission))
        turnover.append(float(np.sum(trade_dollars)))
        position = new_position

    pnl = np.asarray(daily_pnl)
    official, mean, sharpe = score(pnl)
    return BacktestResult(official, mean, sharpe, float(np.mean(turnover)), pnl)


def _returns(history: np.ndarray) -> np.ndarray:
    return np.diff(np.log(history), axis=1).T  # observations x instruments


def constant(direction: float) -> Signal:
    return lambda _: np.full(51, direction)


def return_reversal(lookback: int, scale: float = 1.0) -> Signal:
    """Trade against the sign of the most recent cumulative log return."""
    def signal(history: np.ndarray) -> np.ndarray:
        width = min(lookback + 1, history.shape[1])
        move = np.log(history[:, -1] / history[:, -width])
        return -scale * np.sign(move)
    return signal


def normalised_return_reversal(lookback: int, cap_z: float) -> Signal:
    """A continuous, volatility-normalised reversal signal.

    The signal only reaches maximum exposure after a move of ``cap_z`` recent
    standard deviations.  It is less sensitive to tiny moves than sign-only
    reversal and can materially reduce turnover.
    """
    def signal(history: np.ndarray) -> np.ndarray:
        returns = _returns(history)
        width = min(lookback, len(returns))
        recent = returns[-width:]
        move = recent.sum(axis=0)
        vol = returns[-min(60, len(returns)):].std(axis=0) * np.sqrt(width) + EPS
        return -np.clip(move / vol / cap_z, -1.0, 1.0)
    return signal


def pooled_ar_signal(window: int, lookback: int, ridge: float = 0.0) -> Signal:
    """Fit one pooled AR coefficient and use it on a cumulative return.

    Pooling is intentional: estimating 51 separate AR coefficients from a
    short rolling window is high variance.  The coefficient's sign lets the
    model adapt between momentum and mean-reversion regimes.
    """
    def signal(history: np.ndarray) -> np.ndarray:
        returns = _returns(history)
        train = returns[-min(window, len(returns)):]
        x, y = train[:-1], train[1:]
        beta = float(np.sum(x * y) / (np.sum(x * x) + ridge + EPS))
        width = min(lookback, len(returns))
        move = returns[-width:].sum(axis=0)
        return np.sign(beta) * np.sign(move)
    return signal


def ridge_lead_lag(window: int, ridge: float, top_k: int | None = None) -> Signal:
    """Regularised multivariate one-day return forecast.

    Fits standardised ``r[t+1] = r[t] B + e`` only on the rolling history.
    Ridge regularisation is essential here: the unregularised 51x51 lag matrix
    is very noisy when the available window is only a few hundred days.
    """
    def signal(history: np.ndarray) -> np.ndarray:
        returns = _returns(history)
        train = returns[-min(window, len(returns)):]
        x, y = train[:-1], train[1:]
        x_mean, x_std = x.mean(axis=0), x.std(axis=0) + EPS
        y_mean, y_std = y.mean(axis=0), y.std(axis=0) + EPS
        xs = (x - x_mean) / x_std
        ys = (y - y_mean) / y_std
        gram = xs.T @ xs + ridge * np.eye(xs.shape[1])
        coefficients = np.linalg.solve(gram, xs.T @ ys)
        prediction = ((returns[-1] - x_mean) / x_std) @ coefficients
        if top_k is None:
            return np.tanh(prediction)
        answer = np.zeros_like(prediction)
        chosen = np.argsort(np.abs(prediction))[-top_k:]
        answer[chosen] = np.sign(prediction[chosen])
        return answer
    return signal


def results_table(prices: np.ndarray, candidates: dict[str, Signal]) -> pd.DataFrame:
    """Evaluate fixed, non-overlapping out-of-sample blocks plus last 250 days."""
    ranges = {"125-249": (125, 250), "250-374": (250, 375), "375-499": (375, 500), "250-499": (250, 500)}
    rows = []
    for name, candidate in candidates.items():
        row: dict[str, float | str] = {"strategy": name}
        for label, (start, end) in ranges.items():
            result = backtest(prices, candidate, start, end)
            row[f"{label} score"] = result.official_score
            row[f"{label} sharpe"] = result.annual_sharpe
        rows.append(row)
    return pd.DataFrame(rows).set_index("strategy")


if __name__ == "__main__":
    prices = load_prices()
    candidates: dict[str, Signal] = {
        "always short": constant(-1.0),
        **{f"sign reversal {days}d": return_reversal(days) for days in (1, 2, 3, 5, 10, 20)},
        **{f"continuous reversal {days}d": normalised_return_reversal(days, 1.5) for days in (1, 2, 3, 5)},
        **{f"pooled adaptive AR{window}/{lookback}": pooled_ar_signal(window, lookback) for window in (20, 40, 60, 90) for lookback in (1, 3, 5)},
        **{f"ridge LL {window}/{ridge:g}/K{k}": ridge_lead_lag(window, ridge, k) for window in (60, 120, 240) for ridge in (10.0, 30.0, 100.0) for k in (5, 10, 20)},
    }
    pd.set_option("display.max_columns", None)
    print(results_table(prices, candidates).round(1).sort_values("250-499 score", ascending=False))
