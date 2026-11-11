"""Summary statistics for a user-defined return matrix."""

from __future__ import annotations

import pandas as pd

_TRADING_DAYS = 252


def summarize(
    returns: pd.DataFrame,
    periods_per_year: int = _TRADING_DAYS,
) -> pd.DataFrame:
    """Per-ticker total return, geometric annualized return, and volatility.

    Statistics are computed on the return matrix itself, so they describe the
    same dates you will analyze later. Annualized volatility is the sample
    standard deviation of daily simple returns, scaled by
    ``sqrt(periods_per_year)``. The default of 252 is the usual equity
    convention.
    """
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")

    gross = 1 + returns
    observations = returns.count()
    total_return = gross.prod(min_count=1) - 1
    ann_return = _geometric_annualized(gross, observations, periods_per_year)
    vol_daily = returns.std()
    summary = pd.DataFrame(
        {
            "observations": observations.astype(int),
            "total_return": total_return,
            "ann_return": ann_return,
            "ann_volatility": vol_daily * (periods_per_year ** 0.5),
            "mean_daily": returns.mean(),
            "vol_daily": vol_daily,
            "max_drawdown": returns.apply(_max_drawdown, axis=0),
        }
    )
    summary.index.name = "ticker"
    return summary


def correlation(returns: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation of the columns in the return matrix."""
    return returns.corr()


def _geometric_annualized(
    gross: pd.DataFrame,
    observations: pd.Series,
    periods_per_year: int,
) -> pd.Series:
    invalid = (gross <= 0).any()
    growth = gross.prod(min_count=1)
    annualized = growth ** (periods_per_year / observations) - 1
    return annualized.mask(invalid | observations.eq(0))


def _max_drawdown(returns: pd.Series) -> float:
    clean = returns.dropna()
    if clean.empty:
        return float("nan")
    wealth = pd.concat([pd.Series([1.0]), (1 + clean).cumprod()], ignore_index=True)
    return float((wealth / wealth.cummax() - 1).min())
