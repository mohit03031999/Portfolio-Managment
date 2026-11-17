"""Return signals computed point in time from a daily returns panel.

A *signal* is one number per stock per day, computed from that day and the
days before it, that we hope ranks stocks by their return over the following
weeks. The functions here take the pack's daily simple-returns panel
(dates × tickers, missing where a stock had no price) and return a panel of the
same shape. Every value at date ``t`` uses rows ``<= t`` only, so a signal row
can be read on its own date without looking ahead; the test-suite checks this
by changing the future and asserting the past does not move.

Windows are counted in trading sessions (252 a year, 21 a month). A stock gets
a value when at least ``min_fraction`` of the window's sessions are present.

The registry :data:`SIGNALS` carries each signal's orientation: momentum is
"higher is better", trailing volatility is "lower is better". :func:`oriented`
flips the sign so that a higher oriented value always means "expect a higher
return", which is what the lab and, later, the optimiser consume.

:func:`forward_return` is the outcome the lab scores against: the return over
the sessions *after* ``t``. It is the one function here that looks forward, and
it must never be fed to a signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

YEAR = 252
MONTH = 21


# --------------------------------------------------------------------------- helpers


def _window(panel: pd.DataFrame, lookback: int, min_fraction: float):
    """Rolling view over ``lookback`` sessions needing ``min_fraction`` of them."""
    min_periods = max(2, int(np.ceil(min_fraction * lookback)))
    return panel.rolling(lookback, min_periods=min_periods)


def equal_weight_market(returns: pd.DataFrame) -> pd.Series:
    """The equal-weight average of every stock present each day."""
    market = returns.mean(axis=1, skipna=True)
    market.name = "market"
    return market


# --------------------------------------------------------------------------- signals


def momentum(
    returns: pd.DataFrame, lookback: int = YEAR, skip: int = MONTH, min_fraction: float = 0.9
) -> pd.DataFrame:
    """Cumulative return over the ``lookback`` sessions ending ``skip`` sessions ago.

    The classic "12-1" construction: the past year's return with the most
    recent month left out, because the most recent month reverses on average
    (Jegadeesh and Titman, 1993).
    """
    growth = np.log1p(returns.astype(float))
    past = _window(growth, lookback, min_fraction).sum()
    return np.expm1(past.shift(skip))


def reversal(returns: pd.DataFrame, lookback: int = MONTH, min_fraction: float = 0.9) -> pd.DataFrame:
    """The return over the most recent ``lookback`` sessions (one month).

    Last month's winners tend to give part of it back, so this is a
    "lower is better" signal in the registry.
    """
    growth = np.log1p(returns.astype(float))
    return np.expm1(_window(growth, lookback, min_fraction).sum())


def trailing_volatility(
    returns: pd.DataFrame, lookback: int = YEAR, min_fraction: float = 0.9
) -> pd.DataFrame:
    """Annualised standard deviation of daily returns over the past ``lookback`` sessions."""
    return _window(returns.astype(float), lookback, min_fraction).std() * np.sqrt(YEAR)


def market_beta(
    returns: pd.DataFrame,
    lookback: int = YEAR,
    market: pd.Series | None = None,
    min_fraction: float = 0.9,
) -> pd.DataFrame:
    """Slope of each stock on the equal-weight market over the past ``lookback`` sessions.

    Computed from rolling means of r, m, r·m and m², with the market masked to
    the sessions where the stock has a return so both sides use the same days.
    """
    panel = returns.astype(float)
    if market is None:
        market = equal_weight_market(panel)
    m = market.reindex(panel.index).to_numpy(float)[:, None]
    masked = pd.DataFrame(np.where(panel.isna(), np.nan, m), index=panel.index, columns=panel.columns)
    mean_r = _window(panel, lookback, min_fraction).mean()
    mean_m = _window(masked, lookback, min_fraction).mean()
    mean_rm = _window(panel * masked, lookback, min_fraction).mean()
    mean_mm = _window(masked * masked, lookback, min_fraction).mean()
    covariance = mean_rm - mean_r * mean_m
    variance = mean_mm - mean_m * mean_m
    return covariance / variance


def historical_mean(
    returns: pd.DataFrame, lookback: int = YEAR, horizon: int = MONTH, min_fraction: float = 0.9
) -> pd.DataFrame:
    """Average daily return over the past ``lookback`` sessions, scaled to ``horizon`` sessions.

    This is the textbook expected-return estimate and the lab's negative
    control: the record says it carries almost no information about the next
    month. Scaling by ``horizon`` puts it in the same units as the outcome, so
    the calibration slope reads directly as "the fraction of the forecast that
    survives".
    """
    return _window(returns.astype(float), lookback, min_fraction).mean() * horizon


# --------------------------------------------------------------------------- outcome


def forward_return(returns: pd.DataFrame, horizon: int = MONTH, offset: int = 0) -> pd.DataFrame:
    """Return over sessions ``t+offset+1 .. t+offset+horizon``, missing unless all are present.

    ``offset=0`` is the next ``horizon`` sessions (the holding period);
    ``offset=horizon`` is the month after that, and so on, which is how the
    lab measures how fast a signal's information decays.
    """
    growth = np.log1p(returns.astype(float))
    ahead = growth.rolling(horizon, min_periods=horizon).sum().shift(-(offset + horizon))
    return np.expm1(ahead)


# --------------------------------------------------------------------------- registry


@dataclass(frozen=True)
class Signal:
    key: str
    label: str
    higher_is_better: bool
    description: str
    compute: Callable[[pd.DataFrame], pd.DataFrame]

    @property
    def sign(self) -> float:
        return 1.0 if self.higher_is_better else -1.0


SIGNALS: dict[str, Signal] = {
    "momentum": Signal(
        "momentum",
        "Momentum (12-1)",
        True,
        "return over the past year, skipping the latest month",
        momentum,
    ),
    "reversal": Signal(
        "reversal",
        "Reversal (1 month)",
        False,
        "last month's return; recent winners tend to give some back",
        reversal,
    ),
    "low_vol": Signal(
        "low_vol",
        "Low volatility",
        False,
        "annualised daily volatility over the past year",
        trailing_volatility,
    ),
    "low_beta": Signal(
        "low_beta",
        "Low beta",
        False,
        "slope on the equal-weight market over the past year",
        market_beta,
    ),
    "mean_1y": Signal(
        "mean_1y",
        "Historical mean (1 year)",
        True,
        "average daily return over the past year, scaled to a month",
        historical_mean,
    ),
    "mean_3y": Signal(
        "mean_3y",
        "Historical mean (3 years)",
        True,
        "average daily return over the past three years, scaled to a month",
        lambda returns: historical_mean(returns, lookback=3 * YEAR),
    ),
}


def oriented(signal: Signal, panel: pd.DataFrame) -> pd.DataFrame:
    """The panel with its sign set so that higher means 'expect a higher return'."""
    return panel * signal.sign


def compute_signals(
    returns: pd.DataFrame, keys: tuple[str, ...] | list[str] | None = None
) -> dict[str, pd.DataFrame]:
    """Oriented panels for the requested registry keys (all by default)."""
    chosen = list(SIGNALS) if keys is None else list(keys)
    return {key: oriented(SIGNALS[key], SIGNALS[key].compute(returns)) for key in chosen}
